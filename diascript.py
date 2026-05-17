import os
import sys
import time
import argparse
from vocal_separator import VocalSeparator
from speaker_diarizer import SpeakerDiarizer
from transcriber import WhisperTranscriber

def parse_args():
    parser = argparse.ArgumentParser(
        description="DiaScript: 대용량 오디오/비디오 배경음 분리, 화자 다이어리화 및 이름 매핑 자막 생성 프로그램"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="입력 멀티미디어 파일 경로 (MP4, MKV 등의 영상 또는 MP3, WAV 등의 음성)"
    )
    parser.add_argument(
        "--model", "-m",
        default=r"e:\Github\DiaScripter\demucs4_htdemucs_ft_cac_voice.onnx",
        help="Demucs ONNX 보컬 분리 모델 경로"
    )
    parser.add_argument(
        "--whisper-model", "-w",
        default="base",
        help="faster-whisper 모델 크기 (tiny, base, small, medium, large-v3 등)"
    )
    parser.add_argument(
        "--language", "-l",
        default="ko",
        choices=["ko", "auto"],
        help="전사 언어 모드 (ko: 한국어 고정, auto: 자동 감지)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="최종 결과물이 저장될 디렉토리 경로 (기본값: 입력 파일과 동일 경로)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="GPU 가속을 사용하지 않고 CPU로 강제 구동"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 0. 경로 검증 및 출력 디렉토리 설정
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[DiaScript 에러] 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
        
    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"[DiaScript 에러] ONNX 모델 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)
        
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(input_path) if os.path.dirname(input_path) else "."
        
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    final_output_base = os.path.join(output_dir, base_name)
    
    use_gpu = not args.no_gpu
    
    print("\n==================================================")
    print("DiaScript 오디오 신호 분리 및 스크립트 작성 시스템")
    print("==================================================")
    print(f"- 입력 파일: {input_path}")
    print(f"- ONNX 모델: {model_path}")
    print(f"- Whisper 크기: {args.whisper_model}")
    print(f"- 언어 모드: {'한국어 고정' if args.language == 'ko' else '자동 감지'}")
    print(f"- 구동 장치: {'GPU 가속 사용 시도' if use_gpu else 'CPU 강제 구동'}")
    print("==================================================\n")
    
    start_time = time.time()
    
    # 1단계: 보컬 분리 (Demucs ONNX 연동)
    vocal_separator_start = time.time()
    vocal_audio_path = None
    try:
        separator = VocalSeparator(model_path=model_path, use_gpu=use_gpu)
        # 중간 생성 보컬 트랙 경로 설정
        temp_vocal_path = os.path.join(output_dir, f"{base_name}_vocal_temp.wav")
        vocal_audio_path = separator.separate(input_path, output_path=temp_vocal_path)
    except Exception as e:
        print(f"[DiaScript 에러] 1단계 보컬 분리 도중 오류가 발생했습니다: {e}")
        sys.exit(1)
        
    vocal_separator_end = time.time()
    print(f"[1단계 완료] 보컬 분리 소요 시간: {vocal_separator_end - vocal_separator_start:.2f}초\n")
    
    # 2단계: 화자 분리 및 대화형 검증 루프 (Speaker Diarization & Verification)
    diarizer_start = time.time()
    timeline = None
    names_map = None
    try:
        diarizer = SpeakerDiarizer(use_gpu=use_gpu)
        timeline, names_map = diarizer.run_diarization_pipeline(vocal_audio_path)
    except Exception as e:
        print(f"[DiaScript 에러] 2단계 화자 분리 도중 오류가 발생했습니다: {e}")
        # 임시 보컬 오디오가 유효하면 임시 화자 맵으로 강제 진행 유도
        print("[DiaScript 복구] 단일 화자로 간주하여 3단계를 시도합니다.")
        timeline = [{"start": 0.0, "end": 99999.0, "speaker": "Speaker_01"}]
        names_map = {"Speaker_01": "화자_01"}
        
    diarizer_end = time.time()
    print(f"[2단계 완료] 화자 다이어리화 소요 시간: {diarizer_end - diarizer_start:.2f}초\n")
    
    # 3단계: Whisper 전사 및 자막 생성
    transcriber_start = time.time()
    try:
        # argparse의 하이픈 치환 처리
        w_model_size = getattr(args, "whisper_model", "base")
        transcriber = WhisperTranscriber(model_size=w_model_size, use_gpu=use_gpu)
        
        # 전사 및 화자 타임라인 합병
        aligned_segments = transcriber.transcribe_and_align(
            vocal_audio_path=vocal_audio_path,
            timeline=timeline,
            names_map=names_map,
            language_mode=args.language
        )
        
        # 결과 파일 자동 저장
        txt_out, srt_out = transcriber.save_results(aligned_segments, final_output_base)
        print("\n==================================================")
        print("DiaScript 파이프라인 처리가 성공적으로 완료되었습니다!")
        print(f"- 텍스트 스크립트: {txt_out}")
        print(f"- SRT 자막 파일: {srt_out}")
        print("==================================================")
    except Exception as e:
        print(f"[DiaScript 에러] 3단계 음성 전사 도중 오류가 발생했습니다: {e}")
        sys.exit(1)
    finally:
        # 사용이 끝난 임시 정제 보컬 오디오 삭제 처리 (저장 공간 확보)
        if vocal_audio_path and os.path.exists(vocal_audio_path):
            try:
                os.remove(vocal_audio_path)
                print("[DiaScript] 임시 정제 보컬 오디오 파일이 안전하게 제거되었습니다.")
            except Exception as ex:
                print(f"[DiaScript] 임시 파일 제거 실패: {ex}")
                
    transcriber_end = time.time()
    total_end_time = time.time()
    
    print(f"[3단계 완료] 음성 전사 소요 시간: {transcriber_end - transcriber_start:.2f}초")
    print(f"[전체 프로세스 완료] 총 소요 시간: {total_end_time - start_time:.2f}초\n")

if __name__ == "__main__":
    main()
