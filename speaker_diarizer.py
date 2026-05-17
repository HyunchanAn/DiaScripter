import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import warnings

# 경고 메시지 무시 설정 (SpeechBrain 로드 시의 사소한 UserWarning 배제)
warnings.filterwarnings("ignore")

class SpeakerDiarizer:
    """
    정제된 보컬 오디오 트랙을 분석하여 화자를 분리하고,
    사용자 터미널 대화 검증을 통해 화자 수 및 이름 매핑을 정밀 갱신하는 모듈입니다.
    """
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.classifier = None
        self.embedding_mode = "speechbrain"
        
        # 1차 시도: SpeechBrain ECAPA-TDNN 화자 인식 모델 초기화
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            print(f"[화자 분리] SpeechBrain 화자 임베딩 모델(ECAPA-TDNN) 로드 시도 중... (디바이스: {self.device})")
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            print("[화자 분리] SpeechBrain 임베딩 모델 로드 성공.")
        except Exception as e:
            print(f"[화자 분리] SpeechBrain 초기화 실패. 오프라인 MFCC 통계 분석 모드로 자동 전환됩니다. 원인: {e}")
            self.embedding_mode = "mfcc"
            
    def extract_segments(self, y, sr, min_segment_len=1.5, top_db=22):
        """
        오디오 신호에서 묵음이 아닌 실제 발화 구간(Speech Segments)을 찾아냅니다.
        min_segment_len: 감지할 발화 구간의 최소 길이 (초 단위, 너무 짧은 잡음 배제)
        """
        print("[화자 분리] 발화 구간(Speech Segment) 추출 중...")
        intervals = librosa.effects.split(y, top_db=top_db)
        
        segments = []
        min_samples = int(min_segment_len * sr)
        
        for start, end in intervals:
            if (end - start) >= min_samples:
                segments.append({
                    "start_sample": start,
                    "end_sample": end,
                    "start_time": start / sr,
                    "end_time": end / sr,
                    "length": (end - start) / sr
                })
                
        print(f"[화자 분리] 총 {len(segments)}개의 유의미한 발화 구간이 감지되었습니다.")
        return segments

    def extract_embedding(self, segment_audio, sr):
        """
        주어진 발화 구간 오디오에서 화자의 특징 벡터(임베딩)를 추출합니다.
        """
        if self.embedding_mode == "speechbrain" and self.classifier is not None:
            try:
                # SpeechBrain은 float32 torch 텐서 입력을 기대합니다.
                audio_tensor = torch.tensor(segment_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embeddings = self.classifier.encode_batch(audio_tensor)
                    # [1, 1, 192] -> [192]
                    emb_np = embeddings.squeeze().cpu().numpy()
                return emb_np
            except Exception as e:
                # SpeechBrain 런타임 중 에러가 날 경우 MFCC로 우아하게 Fallback
                pass
                
        # MFCC 기반 Fallback 피처 연산 (MFCC 20차원 + Delta MFCC 20차원 등의 통계량 결합 = 80차원)
        mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        delta_mean = np.mean(mfcc_delta, axis=1)
        delta_std = np.std(mfcc_delta, axis=1)
        
        emb_mfcc = np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta_std])
        # 코사인 유사도 측정을 위해 임베딩 L2 정규화
        norm = np.linalg.norm(emb_mfcc)
        if norm > 0:
            emb_mfcc = emb_mfcc / norm
            
        return emb_mfcc

    def estimate_optimal_speakers(self, embeddings, min_k=2, max_k=8):
        """
        실루엣 스코어를 기반으로 오디오 내의 가장 적절한 1차 화자 수를 자동으로 추정합니다.
        """
        n_samples = len(embeddings)
        if n_samples < min_k:
            return 1
            
        adjusted_max_k = min(max_k, n_samples - 1)
        if adjusted_max_k < min_k:
            return min_k
            
        best_score = -1
        best_k = min_k
        
        for k in range(min_k, adjusted_max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(embeddings)
            
            # 실루엣 계수 계산으로 군집의 조밀도 및 분리도 측정
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
                
        return best_k

    def perform_clustering(self, embeddings, num_speakers):
        """
        지정된 화자 수에 맞춰 K-Means 군집화를 수행하고 각 세그먼트의 화자 레이블을 부여합니다.
        """
        kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        return labels, centroids

    def find_representative_segments(self, segments, embeddings, labels, centroids, num_speakers):
        """
        각 화자 군집별로 클러스터 중심(Centroid)에 가장 가까우면서 
        3~5초 범위에 준하는 대표 발화 구간의 인덱스를 선별합니다.
        """
        reps = {}
        for spk_id in range(num_speakers):
            spk_indices = np.where(labels == spk_id)[0]
            if len(spk_indices) == 0:
                continue
                
            centroid = centroids[spk_id]
            best_idx = None
            min_dist = float('inf')
            
            # 1순위: 길이가 3~5초 범위인 세그먼트 중 중심에 가장 가까운 세그먼트 탐색
            for idx in spk_indices:
                seg = segments[idx]
                emb = embeddings[idx]
                dist = np.linalg.norm(emb - centroid)
                
                # 3초~5초 내외 발화 구간 우대
                if 2.5 <= seg["length"] <= 6.0:
                    dist = dist * 0.7  # 가중치 혜택을 줌
                    
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
                    
            if best_idx is None:
                # 2순위: 범위 내 세그먼트가 없는 경우 거리 기준으로만 최적 탐색
                best_idx = spk_indices[np.argmin([np.linalg.norm(embeddings[i] - centroid) for i in spk_indices])]
                
            reps[spk_id] = best_idx
            
        return reps

    def format_timestamp(self, seconds):
        """
        초 단위 시간을 00:00:00 포맷의 문자열로 변환합니다.
        """
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def run_diarization_pipeline(self, vocal_audio_path):
        """
        화자 분리 메인 파이프라인을 동작시키고 사용자 검증 대화 루프를 가동합니다.
        """
        # 16kHz 모노 보컬 오디오 로드
        y, sr = librosa.load(vocal_audio_path, sr=16000)
        
        # 발화 구간 추출
        segments = self.extract_segments(y, sr)
        if not segments:
            print("[화자 분리] 경고: 감지된 발화 구간이 없습니다. 전체를 단일 화자로 처리합니다.")
            return [{"start": 0.0, "end": len(y)/sr, "speaker": "Speaker_01"}], {"Speaker_01": "화자_01"}
            
        # 발화 구간별 특징 임베딩 추출
        print("[화자 분리] 발화 구간별 특징 벡터(임베딩) 추출 중...")
        embeddings = []
        valid_segments = []
        
        for seg in segments:
            # 해당 구간 오디오 슬라이싱
            seg_audio = y[seg["start_sample"]:seg["end_sample"]]
            # 임베딩 추출
            emb = self.extract_embedding(seg_audio, sr)
            if emb is not None:
                embeddings.append(emb)
                valid_segments.append(seg)
                
        if not embeddings:
            print("[화자 분리] 특징 벡터 추출에 실패하였습니다. 전체를 단일 화자로 처리합니다.")
            return [{"start": 0.0, "end": len(y)/sr, "speaker": "Speaker_01"}], {"Speaker_01": "화자_01"}
            
        embeddings = np.array(embeddings)
        
        # 1차 최적 화자 수 자동 감지
        inferred_k = self.estimate_optimal_speakers(embeddings)
        print(f"[화자 분리] 알고리즘 추정 최적 화자 수: {inferred_k}명")
        
        # 대화형 검증 및 조정 루프 (Interactive Loop)
        labels = None
        centroids = None
        current_k = inferred_k
        
        while True:
            labels, centroids = self.perform_clustering(embeddings, current_k)
            reps = self.find_representative_segments(valid_segments, embeddings, labels, centroids, current_k)
            
            print("\n==================================================")
            print("DiaScript 화자 검증 단계 (Interactive Verification)")
            print("==================================================")
            print(f"알고리즘 분석 결과, 총 {current_k}명의 화자가 감지되었습니다.")
            print("\n[화자별 검증용 대표 발화 샘플 구간]")
            
            for spk_id in range(current_k):
                rep_idx = reps.get(spk_id)
                if rep_idx is not None:
                    seg = valid_segments[rep_idx]
                    t_start = self.format_timestamp(seg["start_time"])
                    t_end = self.format_timestamp(seg["end_time"])
                    print(f"- Speaker_{spk_id + 1:02d}: 구간 [{t_start} - {t_end}] (길이: {seg['length']:.1f}초)")
                else:
                    print(f"- Speaker_{spk_id + 1:02d}: 샘플 구간 없음")
                    
            print("--------------------------------------------------")
            print("선택 1: 감지된 화자 수가 맞다면 그냥 엔터(Enter)를 입력하세요.")
            print("선택 2: 화자 수가 맞지 않다면, 정확한 화자 수(숫자)를 입력하고 엔터를 누르세요.")
            user_input = input("입력: ").strip()
            
            if user_input == "":
                # 화자 수 확정
                break
            else:
                try:
                    new_k = int(user_input)
                    if 1 <= new_k <= len(valid_segments):
                        current_k = new_k
                        print(f"[화자 분리] 화자 수를 {current_k}명으로 재설정하여 다시 군집화합니다.")
                    else:
                        print(f"[화자 분리] 잘못된 입력입니다. 1에서 {len(valid_segments)} 사이의 정수를 입력하십시오.")
                except ValueError:
                    print("[화자 분리] 숫자 또는 엔터(공백)를 입력해야 합니다.")
                    
        # 화자 이름 매핑 입력
        print("\n--------------------------------------------------")
        print("이제 각 화자 번호에 대응하는 실제 이름을 설정해주십시오.")
        print("방법: 콤마(,)로 이름을 나열해 주시거나 개별 입력하십시오.")
        print("예시: 홍길동, 김철수, 이영희 (순서대로 Speaker_01, Speaker_02...에 매핑)")
        print("--------------------------------------------------")
        
        names_map = {}
        while True:
            name_input = input(f"화자 이름 목록 ({current_k}명 분량): ").strip()
            if name_input:
                names = [n.strip() for n in name_input.split(",")]
                if len(names) == current_k:
                    for idx, name in enumerate(names):
                        names_map[f"Speaker_{idx + 1:02d}"] = name
                    break
                else:
                    print(f"[화자 분리] 입력된 이름은 {len(names)}개이나, 확정된 화자 수는 {current_k}명입니다. 다시 입력하십시오.")
            else:
                # 이름 미입력 시 기본 레이블 사용
                for idx in range(current_k):
                    names_map[f"Speaker_{idx + 1:02d}"] = f"화자_{idx + 1:02d}"
                break
                
        print("\n[화자 분리] 이름 매핑이 확정되었습니다:")
        for spk_label, name in names_map.items():
            print(f"- {spk_label} => {name}")
            
        # 화자 타임라인 세그먼트 리스트 구축
        timeline = []
        for idx, seg in enumerate(valid_segments):
            spk_label = f"Speaker_{labels[idx] + 1:02d}"
            timeline.append({
                "start": seg["start_time"],
                "end": seg["end_time"],
                "speaker": spk_label
            })
            
        # 가독성을 위해 시작 시간 기준으로 정렬
        timeline = sorted(timeline, key=lambda x: x["start"])
        
        return timeline, names_map
