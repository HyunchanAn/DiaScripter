import os
import gc
import numpy as np
import soundfile as sf
import onnxruntime as ort
import av

class VocalSeparator:
    """
    Demucs v4 ONNX 모델을 사용하여 오디오/비디오 파일에서 배경음을 제거하고 깨끗한 보컬 트랙을 분리하는 모듈입니다.
    """
    def __init__(self, model_path, use_gpu=True):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {model_path}")
        
        # ONNX Runtime 실행 공급자(Providers) 설정
        available_providers = ort.get_available_providers()
        providers = []
        if use_gpu and 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        print(f"[보컬 분리] ONNX 런타임 초기화 중... 사용 공급자: {providers}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 모델의 입출력 명세 확인
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]
        
        # Demucs v4 고정 규격 설정
        self.sample_rate = 44100
        self.chunk_size = 344064  # 약 7.8초
        self.channels = 2         # 스테레오
        
    def load_audio(self, file_path):
        """
        비디오(MP4, MKV) 또는 오디오(WAV, MP3) 파일로부터 오디오 스트림을 추출하여
        44.1kHz 스테레오 float32 numpy 배열로 반환합니다.
        """
        print(f"[보컬 분리] 오디오 트랙 추출 및 디코딩 중: {os.path.basename(file_path)}")
        
        # 1차 시도: PyAV를 활용한 비디오/오디오 범용 디코딩
        try:
            container = av.open(file_path)
            if not container.streams.audio:
                raise ValueError("파일 내에 오디오 스트림이 존재하지 않습니다.")
                
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format='flt',  # float32
                layout='stereo',
                rate=self.sample_rate,
            )
            
            chunks = []
            for frame in container.decode(stream):
                resampled_frames = resampler.resample(frame)
                for rf in resampled_frames:
                    # PyAV flt/stereo 형태는 (2, samples) 형태의 float32 numpy array를 반환
                    arr = rf.to_ndarray()
                    chunks.append(arr)
            
            container.close()
            
            if chunks:
                audio = np.concatenate(chunks, axis=1)
                if audio.shape[0] == 1:
                    print("[보컬 분리] PyAV 디코딩 중 모노(1채널) 오디오가 감지되어 스테레오(2채널)로 복제 확장합니다.")
                    audio = np.concatenate([audio, audio], axis=0)
                print(f"[보컬 분리] 디코딩 완료 (PyAV) - 샘플 수: {audio.shape[1]}, 채널 수: {audio.shape[0]}")
                return audio
        except Exception as e:
            print(f"[보컬 분리] PyAV 디코딩 실패. Fallback으로 soundfile/librosa 시도. 원인: {e}")
            
        # 2차 시도: soundfile을 활용한 오디오 디코딩
        try:
            data, sr = sf.read(file_path, dtype='float32')
            # (samples, channels) -> (channels, samples)
            if data.ndim == 1:
                # 모노 오디오를 스테레오로 복제
                data = np.stack([data, data], axis=0)
            else:
                data = data.T
                
            if sr != self.sample_rate:
                import librosa
                print(f"[보컬 분리] 샘플레이트 리샘플링 중: {sr}Hz -> {self.sample_rate}Hz")
                # librosa는 (channels, samples)를 지원
                data_resampled = []
                for ch in range(data.shape[0]):
                    res = librosa.resample(data[ch], orig_sr=sr, target_sr=self.sample_rate)
                    data_resampled.append(res)
                data = np.stack(data_resampled, axis=0)
                
            if data.shape[0] == 1:
                print("[보컬 분리] soundfile 디코딩 중 모노(1채널) 오디오가 감지되어 스테레오(2채널)로 복제 확장합니다.")
                data = np.concatenate([data, data], axis=0)
                
            print(f"[보컬 분리] 디코딩 완료 (soundfile/librosa) - 샘플 수: {data.shape[1]}")
            return data
        except Exception as e:
            raise RuntimeError(f"오디오 파일을 디코딩하는 데 모두 실패하였습니다: {e}")

    def separate(self, file_path, output_path=None):
        """
        오디오를 7.8초 단위 슬라이딩 윈도우 및 Overlap-Add 방식으로 보컬 분리를 수행합니다.
        OOM을 방지하고 아주 매끄러운 보컬 오디오 신호를 획득합니다.
        """
        # 오디오 로드 (stereo, 44100Hz)
        audio = self.load_audio(file_path)
        num_samples = audio.shape[1]
        
        # 75% 오버랩 적용 (hop size = chunk_size // 4)
        hop_size = self.chunk_size // 4
        
        # 경계 처리를 위해 양쪽에 패딩을 적용합니다 (오버랩-애드 에지 노이즈 완전 제거)
        # 앞뒤로 chunk_size 만큼 0 패딩을 추가
        padded_audio = np.pad(audio, ((0, 0), (self.chunk_size, self.chunk_size)), mode='constant')
        padded_len = padded_audio.shape[1]
        
        # Hanning window 생성 및 누적 버퍼 준비
        window = np.hanning(self.chunk_size).astype(np.float32)
        vocal_out = np.zeros_like(padded_audio, dtype=np.float32)
        window_sum = np.zeros(padded_len, dtype=np.float32)
        
        # 슬라이딩 윈도우 추론 시작
        print("[보컬 분리] 슬라이딩 윈도우 보컬 분리 추론을 시작합니다...")
        
        # zInput은 상태를 전이하지 않고 매 홉마다 0으로 초기화하여 사용
        z_input = np.zeros((1, 4, 2048, 336), dtype=np.float32)
        
        # 총 스텝 수 계산
        steps = range(0, padded_len - self.chunk_size + 1, hop_size)
        total_steps = len(steps)
        
        last_pct = -1
        for idx, offset in enumerate(steps):
            # 입력 청크 추출 [1, 2, 344064]
            chunk = padded_audio[:, offset:offset+self.chunk_size]
            # 만약 마지막 청크가 chunk_size보다 작다면 0으로 패딩 (슬라이딩 로직상 padded_len 범주 내이므로 항상 chunk_size와 일치함)
            if chunk.shape[1] < self.chunk_size:
                chunk = np.pad(chunk, ((0, 0), (0, self.chunk_size - chunk.shape[1])), mode='constant')
                
            chunk_input = chunk[np.newaxis, :, :].astype(np.float32)
            
            # ONNX 추론 실행
            outputs = self.session.run(
                ["timeOutput"],
                {
                    "timeInput": chunk_input,
                    "zInput": z_input
                }
            )
            
            # timeOutput: [1, 4, 2, 344064] -> 4개 stem 중 3번째(index 3)가 vocals
            vocal_chunk = outputs[0][0, 3, :, :]  # [2, 344064]
            
            # 윈도우 함수 적용하여 가중 합산
            vocal_out[:, offset:offset+self.chunk_size] += vocal_chunk * window
            window_sum[offset:offset+self.chunk_size] += window
            
            # 진행률 표시
            pct = int((idx + 1) / total_steps * 100)
            if pct % 10 == 0 and pct != last_pct:
                print(f"[보컬 분리] 진행 상황: {pct}% 완료")
                last_pct = pct
                
        # 가중 정규화 (0 나누기 방지)
        vocal_out = vocal_out / np.maximum(window_sum, 1e-8)
        
        # 양 끝의 패딩을 제거하고 원래 길이로 복원
        vocal_clean = vocal_out[:, self.chunk_size:self.chunk_size+num_samples]
        
        # 메모리 정리
        del padded_audio, vocal_out, window_sum, z_input
        gc.collect()
        
        # 모노로 변환 (STT 및 화자 분리에는 16kHz 모노 오디오가 효율적이므로 저장 및 후속 처리에 맞춤)
        # 스테레오 채널 평균으로 모노 트랙 생성
        vocal_mono = np.mean(vocal_clean, axis=0)
        
        # 16kHz로 리샘플링하여 중간 보컬 오디오 파일 저장
        import librosa
        print("[보컬 분리] 16kHz Mono 보컬 오디오로 리샘플링 중...")
        vocal_16k = librosa.resample(vocal_mono, orig_sr=self.sample_rate, target_sr=16000)
        
        if output_path is None:
            output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else "."
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_vocal_clean.wav")
            
        print(f"[보컬 분리] 정제된 Vocal 오디오 트랙 저장 중: {output_path}")
        sf.write(output_path, vocal_16k, 16000, subtype='PCM_16')
        
        print("[보컬 분리] 보컬 분리 처리가 완료되었습니다!")
        return output_path
