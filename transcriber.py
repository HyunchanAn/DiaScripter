import os
import torch
from faster_whisper import WhisperModel

class WhisperTranscriber:
    """
    정제된 보컬 오디오 트랙을 faster-whisper로 전사(Transcription)하고,
    검증 완료된 화자 타임라인 정보와 이름을 매핑하여 최종 텍스트 스크립트 및 SRT 자막을 생성하는 모듈입니다.
    """
    def __init__(self, model_size="base", use_gpu=True):
        self.model_size = model_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # GPU 가속이 가능하면 cuda, 그렇지 않으면 cpu
        self.device = "cuda" if self.use_gpu else "cpu"
        # float16은 CUDA 장치에서만 지원됩니다. CPU에서는 int8 또는 float32 권장
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        print(f"[음성 인식] WhisperModel({model_size}) 초기화 중... (디바이스: {self.device}, 연산 타입: {self.compute_type})")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        print("[음성 인식] Whisper 모델 로드 성공.")

    def match_speaker(self, seg_start, seg_end, timeline, names_map):
        """
        Whisper 전사 세그먼트와 가장 많이 겹치는(Overlap) 화자 이름을 타임라인 데이터를 참조하여 찾아냅/니다.
        """
        best_speaker = None
        max_overlap = 0.0
        
        # 1순위: 가장 큰 겹침(Overlap) 영역을 갖는 화자 선정
        for t_seg in timeline:
            overlap = max(0.0, min(seg_end, t_seg["end"]) - max(seg_start, t_seg["start"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = t_seg["speaker"]
                
        # 2순위: 겹치는 구간이 전혀 없는 경우, 세그먼트 중간 시점과 가장 근접한 화자 탐색
        if best_speaker is None and timeline:
            seg_mid = (seg_start + seg_end) / 2.0
            min_dist = float('inf')
            for t_seg in timeline:
                dist_start = abs(seg_mid - t_seg["start"])
                dist_end = abs(seg_mid - t_seg["end"])
                dist = min(dist_start, dist_end)
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = t_seg["speaker"]
                    
        # 최종 이름 치환 (이름 맵에 없으면 기본 화자 ID 유지)
        speaker_name = names_map.get(best_speaker, best_speaker if best_speaker else "알 수 없는 화자")
        return speaker_name

    def format_time_txt(self, seconds):
        """
        초 단위를 00:00:00 포맷의 문자열로 변환 (텍스트 파일용)
        """
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def format_time_srt(self, seconds):
        """
        초 단위를 HH:MM:SS,mmm 포맷의 문자열로 변환 (SRT 자막 파일용)
        """
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def transcribe_and_align(self, vocal_audio_path, timeline, names_map, language_mode="ko"):
        """
        보컬 트랙에 대해 전사를 실행하고, 화자를 매핑한 스크립트 데이터를 생성합니다.
        language_mode: 'ko' (한국어 고정), 'auto' (자동 감지)
        """
        print(f"[음성 인식] 음성 전사(Transcription)를 시작합니다. (언어 모드: {language_mode})")
        
        # 언어 매개변수 바인딩 (기본 한국어 고정 / auto 설정 시 자동 감지)
        lang_param = "ko" if language_mode == "ko" else None
        
        # Whisper 전사 실행 (Silero VAD 활성화로 오디오 정합도 최대화)
        segments, info = self.model.transcribe(
            vocal_audio_path, 
            language=lang_param,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_speech_duration_ms=250)
        )
        
        if lang_param is None:
            print(f"[음성 인식] 자동 감지된 언어: {info.language} (신뢰도: {info.language_probability:.2f})")
            
        aligned_segments = []
        
        # segments는 generator이므로 순회하며 리스트화
        for segment in segments:
            # 텍스트 양 끝 공백 제거
            text = segment.text.strip()
            if not text:
                continue
                
            # 화자 매칭 수행
            speaker_name = self.match_speaker(segment.start, segment.end, timeline, names_map)
            
            aligned_segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker_name,
                "text": text
            })
            
        print(f"[음성 인식] 전사 완료. 총 {len(aligned_segments)}개의 발화 자막 세그먼트 생성됨.")
        return aligned_segments

    def save_results(self, aligned_segments, base_output_path):
        """
        최종 스크립트를 구조화된 텍스트(.txt) 및 표준 자막 파일(.srt)로 저장합니다.
        """
        txt_path = f"{base_output_path}.txt"
        srt_path = f"{base_output_path}.srt"
        
        print(f"[음성 인식] 결과 저장 중...")
        
        # 1. 텍스트 스크립트(.txt) 저장
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            for seg in aligned_segments:
                t_start = self.format_time_txt(seg["start"])
                t_end = self.format_time_txt(seg["end"])
                f_txt.write(f"[{t_start} - {t_end}] {seg['speaker']}: {seg['text']}\n")
                
        # 2. 표준 자막 파일(.srt) 저장
        with open(srt_path, "w", encoding="utf-8") as f_srt:
            for idx, seg in enumerate(aligned_segments):
                t_start = self.format_time_srt(seg["start"])
                t_end = self.format_time_srt(seg["end"])
                f_srt.write(f"{idx + 1}\n")
                f_srt.write(f"{t_start} --> {t_end}\n")
                f_srt.write(f"[{seg['speaker']}] {seg['text']}\n\n")
                
        print(f"[음성 인식] 텍스트 파일 저장 완료: {txt_path}")
        print(f"[음성 인식] 자막 SRT 파일 저장 완료: {srt_path}")
        return txt_path, srt_path
