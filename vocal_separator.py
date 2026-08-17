import os
import gc
import torch
import numpy as np
import soundfile as sf
import demucs.api
import librosa

class VocalSeparator:
    """
    공식 Demucs PyTorch 라이브러리를 사용하여 오디오/비디오 파일에서
    배경음을 제거하고 깨끗한 보컬 트랙을 분리하는 모듈입니다.
    """
    def __init__(self, model_name: str = "htdemucs", use_gpu: bool = True):
        """
        보컬 분리 초기화
        :param model_name: 사용할 Demucs 모델 이름 (기본값: htdemucs)
        :param use_gpu: GPU 사용 여부
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        print(f"[보컬 분리] 모델 초기화 중... (디바이스: {self.device})")
        
        # Demucs API Separator 초기화 (자동으로 모델을 다운로드하거나 로드합니다)
        self.separator = demucs.api.Separator(
            model=self.model_name,
            device=self.device,
            shifts=1,
            overlap=0.25,
            split=True,
            segment=None,
            jobs=0
        )
        print("[보컬 분리] 초기화 완료.")

    def separate(self, file_path: str, output_path: str = None, progress_callback=None) -> str:
        """
        오디오 파일에서 보컬을 분리하여 16kHz Mono WAV로 저장합니다.
        
        :param file_path: 원본 오디오/비디오 파일 경로
        :param output_path: 결과물을 저장할 경로 (지정하지 않으면 원본 경로+_vocals.wav)
        :param progress_callback: 진행률 콜백 함수 (pct: float, msg: str)
        :return: 분리된 보컬 파일의 절대 경로
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
        if output_path is None:
            base, _ = os.path.splitext(file_path)
            output_path = f"{base}_vocals.wav"
            
        print(f"[보컬 분리] 처리를 시작합니다: {file_path}")
        if progress_callback: progress_callback(0.05, "보컬 분리 준비 중...")

        # Demucs API의 콜백을 사용해 진행률을 대략적으로 추적합니다.
        def demucs_callback(args):
            if progress_callback:
                try:
                    # state: 'start', 'step', 'end'
                    if args.get('state') == 'step':
                        # 진행 상황을 대략적으로 추적하기 위해 모델 내 chunk 처리 과정을 시뮬레이션
                        progress_callback(0.5, "AI 보컬 분리 진행 중... (약간의 시간이 소요됩니다)")
                except Exception:
                    pass

        self.separator.update_parameter(callback=demucs_callback)

        try:
            import av
            target_sr = self.separator.samplerate
            
            if progress_callback: progress_callback(0.1, "PyAV로 오디오/비디오 디코딩 중 (시스템 FFmpeg 독립 처리)...")
            print(f"[보컬 분리] PyAV를 이용한 내장 오디오 디코딩 시작 (타겟 SR: {target_sr})")
            
            container = av.open(file_path)
            audio_stream = None
            for stream in container.streams:
                if stream.type == 'audio':
                    audio_stream = stream
                    break
            if not audio_stream:
                raise ValueError("오디오 스트림을 찾을 수 없습니다.")
                
            resampler = av.AudioResampler(
                format='fltp',
                layout='stereo',
                rate=target_sr
            )
            
            frames = []
            for packet in container.demux(audio_stream):
                # PyAV 12+ decode() can return a list of frames
                for frame in packet.decode():
                    r_frames = resampler.resample(frame)
                    for r_frame in r_frames:
                        frames.append(r_frame.to_ndarray())
                        
            # 플러시
            for r_frame in resampler.resample(None):
                frames.append(r_frame.to_ndarray())
                
            if not frames:
                raise ValueError("오디오 프레임 디코딩에 실패했습니다.")
                
            audio_np = np.concatenate(frames, axis=1)
            mix_tensor = torch.from_numpy(audio_np).float()
            
            print(f"[보컬 분리] 디코딩 완료. Tensor 형태: {mix_tensor.shape}")
            if progress_callback: progress_callback(0.2, "AI 보컬 분리 연산 진행 중... (약간의 시간이 소요됩니다)")
            
            # Demucs에 텐서 직접 전달 (시스템 FFmpeg 의존성 완전 제거)
            res = self.separator.separate_tensor(mix_tensor, target_sr)
            
            if isinstance(res, tuple):
                _, stems = res
            else:
                stems = res
                
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f"[FATAL VOCAL ERROR]\n{err_msg}")
            raise RuntimeError(f"보컬 분리 실패 (오디오 로드 및 분리 오류): {str(e)}")

        if 'vocals' not in stems:
            raise ValueError(f"모델 출력에 'vocals' 트랙이 없습니다. (출력 트랙: {list(stems.keys())})")
            
        if progress_callback: progress_callback(0.8, "보컬 트랙 추출 완료, 후처리 중...")
        print("[보컬 분리] 보컬 트랙 추출 완료.")
        
        # vocal_tensor 형태: (channels, samples), float32
        vocal_tensor = stems['vocals']
        sample_rate = self.separator.model.samplerate
        
        # PyTorch Tensor를 Numpy Array로 변환
        vocal_np = vocal_tensor.cpu().numpy()
        
        if progress_callback: progress_callback(0.85, "16kHz Mono로 변환 중...")
        print("[보컬 분리] 16kHz Mono 다운믹스 리샘플링 시작...")
        
        # 스테레오를 모노로 변환 (채널 평균)
        if vocal_np.ndim == 2 and vocal_np.shape[0] > 1:
            vocal_mono = np.mean(vocal_np, axis=0)
        else:
            vocal_mono = vocal_np.squeeze()
            
        # 16kHz로 리샘플링 (화자 분리 및 STT 입력용)
        if sample_rate != 16000:
            vocal_16k = librosa.resample(vocal_mono, orig_sr=sample_rate, target_sr=16000)
        else:
            vocal_16k = vocal_mono
            
        if progress_callback: progress_callback(0.95, "오디오 저장 중...")
        print(f"[보컬 분리] 정제된 보컬 오디오 트랙 저장 중: {output_path}")
        
        # WAV 파일로 저장 (16bit PCM)
        sf.write(output_path, vocal_16k, 16000, subtype='PCM_16')
        
        if progress_callback: progress_callback(1.0, "보컬 분리 완료!")
        print(f"[보컬 분리] 완료! 저장 경로: {output_path}")
        
        # 메모리 정리
        del vocal_tensor, stems, vocal_np, vocal_mono, vocal_16k
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return os.path.abspath(output_path)
