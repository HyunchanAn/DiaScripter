import os
import shutil
import time
import io
import base64
import soundfile as sf
import librosa
import numpy as np
import gradio as gr

from vocal_separator import VocalSeparator
from speaker_diarizer import SpeakerDiarizer
from transcriber import WhisperTranscriber

# 전역 모듈 캐시 (중복 인스턴스화 및 리소스 낭비 원천 배제)
MODEL_CACHE = {
    "separator": None,
    "diarizer": None,
    "transcriber": None,
    "last_separator_model": None,
    "last_whisper_model": None
}

def get_vocal_separator(model_path, use_gpu):
    if MODEL_CACHE["separator"] is None or MODEL_CACHE["last_separator_model"] != model_path:
        MODEL_CACHE["separator"] = VocalSeparator(model_path=model_path, use_gpu=use_gpu)
        MODEL_CACHE["last_separator_model"] = model_path
    return MODEL_CACHE["separator"]

def get_speaker_diarizer(use_gpu):
    if MODEL_CACHE["diarizer"] is None:
        MODEL_CACHE["diarizer"] = SpeakerDiarizer(use_gpu=use_gpu)
    return MODEL_CACHE["diarizer"]

def get_whisper_transcriber(model_size, use_gpu):
    if MODEL_CACHE["transcriber"] is None or MODEL_CACHE["last_whisper_model"] != model_size:
        MODEL_CACHE["transcriber"] = WhisperTranscriber(model_size=model_size, use_gpu=use_gpu)
        MODEL_CACHE["last_whisper_model"] = model_size
    return MODEL_CACHE["transcriber"]

# --- 백그라운드 이벤트 처리 함수군 ---

def process_vocal_separation(file_input, local_path_input, model_path, whisper_model, language, no_gpu):
    """
    1단계: 보컬 분리 처리를 수행하고 중간 보컬 오디오 경로를 반환합니다.
    """
    use_gpu = not no_gpu
    
    # 입력 파일 확인 (드래그앤드롭 파일 우선, 없으면 로컬 절대 경로 사용)
    input_file = file_input if file_input else local_path_input
    if not input_file or not os.path.exists(input_file):
        raise gr.Error("유효한 입력 파일을 선택하거나 로컬 절대 경로를 기입하십시오.")
        
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else "."
    
    # 보컬 정제 파일 최종 저장 경로 바인딩
    vocal_output_path = os.path.join(output_dir, f"{base_name}_vocal_temp.wav")
    
    gr.Info("1단계 보컬 분리를 가동합니다... (배경음 제거 진행 중)")
    start_t = time.time()
    
    try:
        separator = get_vocal_separator(model_path, use_gpu)
        actual_vocal_path = separator.separate(input_file, output_path=vocal_output_path)
        actual_vocal_path = actual_vocal_path.replace("\\", "/")
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[FATAL VOCAL ERROR]\n{err_msg}")
        raise gr.Error(f"보컬 분리 처리 상세 백트레이스 로그:\n{err_msg}")
        
    end_t = time.time()
    gr.Info(f"보컬 분리 완료! 소요 시간: {end_t - start_t:.2f}초")
    
    # 2단계 준비를 위해 오디오 정보를 gr.State와 UI 컴포넌트에 넘겨줌
    return (
        actual_vocal_path,  # state_vocal_path
        actual_vocal_path,  # ui_vocal_audio
        gr.update(visible=True),  # step2_panel 가시화
        gr.update(value=f"보컬 트랙 정제 완료: {os.path.basename(actual_vocal_path)}"),
        gr.update(visible=True, value="보컬 분리 완료! 다음 단계로 이동하기") # btn_go_to_step2 가시화 및 문구 갱신
    )

def run_diarization_pipeline(vocal_path, no_gpu):
    """
    2단계 화자 식별, K-Means 군집화, 대표 샘플 슬라이싱 및 플레이어 매핑을 
    단 하나의 파이썬 함수 흐름 안에서 일괄 완료하여 데이터 무결성을 보장합니다.
    """
    if not vocal_path or not os.path.exists(vocal_path):
        raise gr.Error("보컬 분리 오디오 파일이 준비되지 않았습니다.")
        
    use_gpu = not no_gpu
    gr.Info("통합 화자 감지 및 발화 특징 추출을 기동합니다...")
    print(f"[DEBUG LOG] Starting run_diarization_pipeline with vocal_path: {vocal_path}")
    
    try:
        diarizer = get_speaker_diarizer(use_gpu)
        
        # 16kHz 모노 오디오 로드 및 발화 슬라이싱
        print("[DEBUG LOG] Loading audio for speaker diarization...")
        y, sr = librosa.load(vocal_path, sr=16000)
        segments = diarizer.extract_segments(y, sr)
        
        if not segments:
            raise gr.Error("오디오에서 말소리(발화 구간)를 전혀 찾아내지 못했습니다.")
            
        # 발화 구간별 특징 추출
        print(f"[DEBUG LOG] Extracting embeddings for {len(segments)} segments...")
        embeddings = []
        valid_segments = []
        for seg in segments:
            seg_audio = y[seg["start_sample"]:seg["end_sample"]]
            emb = diarizer.extract_embedding(seg_audio, sr)
            if emb is not None:
                embeddings.append(emb)
                valid_segments.append(seg)
                
        if not embeddings:
            raise gr.Error("화자 음성 특징 임베딩을 정량화하지 못했습니다.")
            
        embeddings = np.array(embeddings)
        
        # 1차 최적 화자 수(K) 자동 추정
        inferred_k = diarizer.estimate_optimal_speakers(embeddings)
        print(f"[DEBUG LOG] Auto-detected K speakers: {inferred_k}")
        
        # K-Means 군집화
        labels, centroids = diarizer.perform_clustering(embeddings, inferred_k)
        reps = diarizer.find_representative_segments(valid_segments, embeddings, labels, centroids, inferred_k)
        
        html_blocks = []
        ui_text_updates = []
        
        for i in range(8):
            if i < inferred_k:
                rep_idx = reps.get(i)
                if rep_idx is not None:
                    seg = valid_segments[rep_idx]
                    seg_audio = y[seg["start_sample"]:seg["end_sample"]]
                    
                    buffer = io.BytesIO()
                    sf.write(buffer, seg_audio, sr, format='WAV', subtype='PCM_16')
                    b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    t_start = diarizer.format_timestamp(seg["start_time"])
                    t_end = diarizer.format_timestamp(seg["end_time"])
                    
                    html_blocks.append(f"""
                    <div style='margin-bottom: 10px; padding: 10px; background: #2a2a2b; border-radius: 8px;'>
                        <strong style='color: #4da6ff;'>Speaker_{i + 1:02d}</strong> 목소리 샘플 [{t_start} - {t_end}]<br/>
                        <audio controls style='width: 100%; margin-top: 5px;' src='data:audio/wav;base64,{b64_data}'></audio>
                    </div>
                    """)
                    text_update = gr.update(value=f"화자_{i + 1:02d}", visible=True, label=f"Speaker_{i + 1:02d} 실제 이름 매핑")
                else:
                    html_blocks.append(f"""
                    <div style='margin-bottom: 10px; padding: 10px; background: #2a2a2b; border-radius: 8px;'>
                        <strong style='color: #ff4d4d;'>Speaker_{i + 1:02d}</strong> 샘플 없음
                    </div>
                    """)
                    text_update = gr.update(value=f"화자_{i + 1:02d}", visible=True, label=f"Speaker_{i + 1:02d} 실제 이름 매핑")
            else:
                text_update = gr.update(visible=False)
                
            ui_text_updates.append(text_update)
            
        final_html = "".join(html_blocks)
        if not final_html:
            final_html = "<div>표시할 화자 샘플이 없습니다.</div>"
            
        # 타임라인 생성
        timeline = []
        for idx, seg in enumerate(valid_segments):
            spk_label = f"Speaker_{labels[idx] + 1:02d}"
            timeline.append({
                "start": seg["start_time"],
                "end": seg["end_time"],
                "speaker": spk_label
            })
        timeline = sorted(timeline, key=lambda x: x["start"])
        print(f"[DEBUG LOG] Successfully generated timeline with {len(timeline)} events.")
        
        # 최종 리턴값 매핑
        return (
            valid_segments,  # state_segments
            embeddings,      # state_embeddings
            inferred_k,      # state_current_k
            gr.update(value=inferred_k),  # ui_speaker_count_slider
            gr.update(value=f"알고리즘 1차 감지 화자 수: {inferred_k}명. 아래 샘플을 듣고 실시간 조율하십시오."),
            timeline,        # state_timeline
            gr.update(value=final_html, visible=True) # ui_audio_html
        ) + tuple(ui_text_updates)
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[FATAL DIARIZATION ERROR]\n{err_msg}")
        raise gr.Error(f"화자 감지 분석 상세 백트레이스 로그:\n{err_msg}")

def run_recluster_pipeline(vocal_path, segments, embeddings, num_speakers, no_gpu):
    """
    사용자가 화자 수를 수동 조정했을 때, 메모리 내 세션 데이터를 바탕으로 
    플레이어 리스트와 최종 타임라인을 고속 갱신합니다.
    """
    if not vocal_path or segments is None or embeddings is None:
        raise gr.Error("세션 정보가 유효하지 않아 화자 수 변경이 불가능합니다.")
        
    use_gpu = not no_gpu
    diarizer = get_speaker_diarizer(use_gpu)
    print(f"[DEBUG LOG] Reclustering speakers to: {num_speakers}")
    
    try:
        labels, centroids = diarizer.perform_clustering(embeddings, num_speakers)
        reps = diarizer.find_representative_segments(segments, embeddings, labels, centroids, num_speakers)
        
        y, sr = librosa.load(vocal_path, sr=16000)
        
        html_blocks = []
        ui_text_updates = []
        
        for i in range(8):
            if i < num_speakers:
                rep_idx = reps.get(i)
                if rep_idx is not None:
                    seg = segments[rep_idx]
                    seg_audio = y[seg["start_sample"]:seg["end_sample"]]
                    
                    buffer = io.BytesIO()
                    sf.write(buffer, seg_audio, sr, format='WAV', subtype='PCM_16')
                    b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    t_start = diarizer.format_timestamp(seg["start_time"])
                    t_end = diarizer.format_timestamp(seg["end_time"])
                    
                    html_blocks.append(f"""
                    <div style='margin-bottom: 10px; padding: 10px; background: #2a2a2b; border-radius: 8px;'>
                        <strong style='color: #4da6ff;'>Speaker_{i + 1:02d}</strong> 목소리 샘플 [{t_start} - {t_end}]<br/>
                        <audio controls style='width: 100%; margin-top: 5px;' src='data:audio/wav;base64,{b64_data}'></audio>
                    </div>
                    """)
                    text_update = gr.update(value=f"화자_{i + 1:02d}", visible=True, label=f"Speaker_{i + 1:02d} 실제 이름 매핑")
                else:
                    html_blocks.append(f"""
                    <div style='margin-bottom: 10px; padding: 10px; background: #2a2a2b; border-radius: 8px;'>
                        <strong style='color: #ff4d4d;'>Speaker_{i + 1:02d}</strong> 샘플 없음
                    </div>
                    """)
                    text_update = gr.update(value=f"화자_{i + 1:02d}", visible=True, label=f"Speaker_{i + 1:02d} 실제 이름 매핑")
            else:
                text_update = gr.update(visible=False)
                
            ui_text_updates.append(text_update)
            
        final_html = "".join(html_blocks)
        if not final_html:
            final_html = "<div>표시할 화자 샘플이 없습니다.</div>"
            
        timeline = []
        for idx, seg in enumerate(segments):
            spk_label = f"Speaker_{labels[idx] + 1:02d}"
            timeline.append({
                "start": seg["start_time"],
                "end": seg["end_time"],
                "speaker": spk_label
            })
        timeline = sorted(timeline, key=lambda x: x["start"])
        
        return (gr.update(value=final_html, visible=True),) + tuple(ui_text_updates) + (timeline,)
        
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[FATAL RECLUSTER ERROR]\n{err_msg}")
        raise gr.Error(f"화자 수 변경 상세 백트레이스 로그:\n{err_msg}")

def run_transcription_and_export(vocal_path, timeline, whisper_model, language, no_gpu, *names_list):
    """
    3단계: 사용자가 설정한 한글 이름 목록을 받아 Whisper 음성 전사를 완료하고 최종 스크립트/자막 파일을 생성합니다.
    """
    if not vocal_path or not os.path.exists(vocal_path):
        raise gr.Error("보컬 분리 오디오 트랙을 찾을 수 없습니다.")
    if not timeline:
        raise gr.Error("화자 타임라인 정보가 유효하지 않습니다.")
        
    use_gpu = not no_gpu
    
    # 8개 텍스트박스 입력에서 화자 수만큼의 이름 추출 및 매핑 맵핑 맵 빌드
    names_map = {}
    
    # timeline에서 실제 검출된 화자 목록 탐색
    detected_speakers = sorted(list(set([t["speaker"] for t in timeline])))
    
    for idx, spk in enumerate(detected_speakers):
        if idx < len(names_list) and names_list[idx]:
            names_map[spk] = names_list[idx].strip()
        else:
            names_map[spk] = f"화자_{idx + 1:02d}"
            
    gr.Info("Whisper 음성 인식을 시작합니다... (텍스트 고속 변환 중)")
    start_t = time.time()
    
    try:
        transcriber = get_whisper_transcriber(whisper_model, use_gpu)
        
        # 전사 및 이름 치환 매핑
        aligned_segments = transcriber.transcribe_and_align(
            vocal_audio_path=vocal_path,
            timeline=timeline,
            names_map=names_map,
            language_mode=language
        )
        
        # 최종 결과 파일 자동 저장 경로 빌드
        base_dir = os.path.dirname(vocal_path)
        # _vocal_temp.wav 가 포함된 중간 이름을 원본 명칭으로 회귀
        vocal_base = os.path.basename(vocal_path).replace("_vocal_temp.wav", "")
        final_base_path = os.path.join(base_dir, vocal_base)
        
        txt_path, srt_path = transcriber.save_results(aligned_segments, final_base_path)
        
        # 화면 출력을 위해 txt 및 srt 내용 로드
        with open(txt_path, "r", encoding="utf-8") as f:
            txt_content = f.read()
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read()
            
        end_t = time.time()
        gr.Info(f"자막 전사 최종 생성 완료! 소요 시간: {end_t - start_t:.2f}초")
        
        # 웹 브라우저 캐시 임시 파일 제거 (diascript.py 명세와 결을 맞춤)
        try:
            if os.path.exists(vocal_path):
                os.remove(vocal_path)
                print("[Gradio 웹앱] 임시 정제 보컬 오디오 파일 안전 제거 성공.")
        except Exception as ex:
            print(f"[Gradio 웹앱] 임시 정제 보컬 파일 제거 오류: {ex}")
            
        return (
            gr.update(visible=True),  # step3_panel (결과창 활성화)
            txt_content,              # ui_txt_preview
            srt_content,              # ui_srt_preview
            [txt_path, srt_path]     # ui_download_files (gr.File 연동)
        )
    except Exception as e:
        raise gr.Error(f"음성 인식 및 매핑 도중 실패: {e}")

# --- UI 레이아웃 선언 및 조립 (Premium Dark / Ocean Theme 적용) ---

custom_css = """
body, .gradio-container {
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
}
.main-title {
    text-align: center;
    margin-bottom: 5px;
    font-weight: 800;
}
.sub-title {
    text-align: center;
    margin-bottom: 25px;
    color: #888;
}
"""

def create_ui():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="zinc"
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        block_background_fill="*neutral_900",
        block_background_fill_dark="*neutral_900",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500"
    )
    
    with gr.Blocks(theme=theme, css=custom_css, title="DiaScript AI 화자 분리 자막 시스템") as demo:
        # --- 세션 상태 스토리지 선언 ---
        state_vocal_path = gr.State(value="")
        state_segments = gr.State(value=None)
        state_embeddings = gr.State(value=None)
        state_current_k = gr.State(value=2)
        state_timeline = gr.State(value=None)
        
        # 타이틀 및 개요
        gr.HTML("<h1 class='main-title'>DiaScript AI</h1>")
        gr.HTML("<p class='sub-title'>대용량 동영상 및 음성 배경음악 분리, 인공지능 화자 식별 및 청음 보정 통합 스크립트 작성 시스템</p>")
        
        with gr.Tabs() as tabs:
            # --- 탭 1: 파일 입력 및 아키텍처 환경 설정 ---
            with gr.TabItem("프로젝트 설정 및 파일 입력", id="tab_setting") as tab1:
                gr.Markdown("### 1. 입력 파일 지정을 행해주십시오.")
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="동영상 또는 오디오 파일 업로드 (MP4, MKV, WAV, MP3 등)",
                            file_types=["video", "audio"]
                        )
                    with gr.Column(scale=1):
                        local_path_input = gr.Textbox(
                            label="대용량 서버 절대 경로 직접 입력 (선택사항, 업로드 우회용)",
                            placeholder="예시: E:\\Github\\DiaScripter\\samples\\long_interview.mp4",
                            lines=2
                        )
                        
                gr.Markdown("### 2. 가속 연산 및 아키텍처 튜닝 설정")
                with gr.Row():
                    model_path = gr.Textbox(
                        label="Demucs ONNX 보컬 분리 가중치 경로",
                        value=r"e:\Github\DiaScripter\demucs4_htdemucs_ft_cac_voice.onnx"
                    )
                    whisper_model = gr.Dropdown(
                        label="faster-whisper 음성 인식 모델 크기",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        value="base"
                    )
                    language = gr.Dropdown(
                        label="전사 언어 모드",
                        choices=[("한국어 고정 강제 (권장)", "ko"), ("다국어 자동 감지 및 혼용", "auto")],
                        value="ko"
                    )
                    no_gpu = gr.Checkbox(
                        label="GPU 가속 대신 CPU 연산 강제 고정 (CUDA 오류 대비)",
                        value=False
                    )
                    
                btn_start_separation = gr.Button("보컬 분리 가동 시작 (배경음 제거)", variant="primary", size="lg")
                # 신규 추가: 보컬 분리 완료 시 우뚝 서는 대형 다음 단계 이동 점프 버튼
                btn_go_to_step2 = gr.Button("보컬 분리 진행 중...", variant="primary", visible=False, size="lg")
                
            # --- 탭 2: 배경음 분리 결과 청음 및 실시간 화자 보정 ---
            with gr.TabItem("보컬 분리 및 화자 검증", id="tab_diarization") as tab2:
                gr.Markdown("### 1단계: 정제 보컬 오디오 트랙")
                ui_vocal_audio = gr.Audio(label="정제된 보컬 오디오 스트림 (16kHz Mono)", interactive=False)
                
                # 2단계 패널 (기본 비활성, 보컬 분리 끝난 후 가시화)
                with gr.Group(visible=False) as step2_panel:
                    gr.Markdown("### 2단계: 실시간 인공지능 화자 식별 및 한글 이름 매핑")
                    ui_status_label = gr.Label(value="화자 식별을 진행하기 위해 아래 버튼을 누르십시오.")
                    
                    with gr.Row(equal_height=True):
                        btn_start_diarization = gr.Button("화자 분리 연산 시작 (여기를 클릭하십시오)", variant="primary", size="lg")
                        
                    with gr.Row(visible=True) as speaker_recluster_row:
                        ui_speaker_count_slider = gr.Slider(
                            label="실제 화자 수 재조정 (샘플 청음 후 수치가 틀리다면 보정하십시오)",
                            minimum=1, maximum=8, step=1, value=2, interactive=True
                        )
                        btn_recluster = gr.Button("화자 수 재조정 반영", variant="secondary")
                        
                    gr.Markdown("#### 아래 각 화자별 대표 발화 샘플의 재생 단추를 눌러 목소리를 판별하고 한글 이름을 입력해 주십시오.")
                    
                    # 단일 HTML 컴포넌트를 이용한 오디오 플레이어 서빙 (Errno 13 원천 차단)
                    with gr.Row():
                        with gr.Column(scale=2):
                            ui_audio_html = gr.HTML(visible=False)
                        with gr.Column(scale=1):
                            text_boxes = []
                            for i in range(8):
                                txt = gr.Textbox(label=f"Speaker_{i + 1:02d} 이름", visible=False, value=f"화자_{i + 1:02d}")
                                text_boxes.append(txt)
                                
                    btn_start_transcription = gr.Button("3단계 음성 전사 및 스크립트 자막 추출 시작", variant="primary", size="lg")
                    
            # --- 탭 3: 최종 텍스트 및 SRT 자막 다운로드 ---
            with gr.TabItem("최종 결과 및 자막 다운로드", id="tab_download") as tab3:
                with gr.Group(visible=False) as step3_panel:
                    gr.Markdown("### 3단계: 최종 화자 매핑 자막 파일 생성 성공")
                    
                    ui_download_files = gr.File(
                        label="최종 생성된 자막 및 텍스트 스크립트 파일 다운로드",
                        file_count="multiple"
                    )
                    
                    with gr.Row():
                        ui_txt_preview = gr.Textbox(
                            label="텍스트 스크립트 미리보기 (.txt)",
                            lines=15, max_lines=30, interactive=False
                        )
                        ui_srt_preview = gr.Textbox(
                            label="표준 SRT 자막 미리보기 (.srt)",
                            lines=15, max_lines=30, interactive=False
                        )
                        
        # --- 이벤트 파이프라인 연결 및 조립 (Event Bindings) ---
        
        # 1. 1단계 [보컬 분리] 시작 버튼 바인딩
        btn_start_separation.click(
            fn=process_vocal_separation,
            inputs=[file_input, local_path_input, model_path, whisper_model, language, no_gpu],
            outputs=[state_vocal_path, ui_vocal_audio, step2_panel, ui_status_label, btn_go_to_step2]
        )
        
        # 1-A. 신규 추가: 1단계 완료 버튼 클릭 시, 2번 탭 점프 + 화자 분리 일괄 파이프라인 즉시 자동 기동
        btn_go_to_step2.click(
            fn=lambda: gr.update(selected="tab_diarization"),
            outputs=tabs
        ).then(
            fn=run_diarization_pipeline,
            inputs=[state_vocal_path, no_gpu],
            outputs=[state_segments, state_embeddings, state_current_k, ui_speaker_count_slider, ui_status_label, state_timeline, ui_audio_html] + text_boxes
        )
        
        # 2. 2단계 [화자 분리 연산 시작] 버튼 바인딩 (단일 오케스트레이션 구동으로 deepcopy 원천 근절)
        btn_start_diarization.click(
            fn=run_diarization_pipeline,
            inputs=[state_vocal_path, no_gpu],
            outputs=[state_segments, state_embeddings, state_current_k, ui_speaker_count_slider, ui_status_label, state_timeline, ui_audio_html] + text_boxes
        )
        
        # 3. 2단계 [화자 수 재조정 반영] 버튼 바인딩 (통합 캐시 기반 0.1초 갱신)
        btn_recluster.click(
            fn=run_recluster_pipeline,
            inputs=[state_vocal_path, state_segments, state_embeddings, ui_speaker_count_slider, no_gpu],
            outputs=[ui_audio_html] + text_boxes + [state_timeline]
        ).then(
            fn=lambda k: gr.Info(f"화자 수를 {k}명으로 재조정하여 군집 배치를 완료하였습니다."),
            inputs=[ui_speaker_count_slider]
        )
        
        # 4. 3단계 [최종 음성 전사 시작] 버튼 바인딩
        # 8개 텍스트 입력 박스의 실시간 데이터 수집 및 timeline(state)과 매핑
        btn_start_transcription.click(
            fn=run_transcription_and_export,
            inputs=[state_vocal_path, state_timeline, whisper_model, language, no_gpu] + text_boxes,
            outputs=[step3_panel, ui_txt_preview, ui_srt_preview, ui_download_files]
        ).then(
            fn=lambda: gr.update(selected="tab_download"),
            outputs=tabs
        ).then(
            fn=lambda: gr.Info("자막 변환 최종 완료! 결과 확인 및 자막 다운로드 탭으로 즉시 안전하게 스위칭되었습니다.")
        )
        
    return demo

if __name__ == "__main__":
    ui_app = create_ui()
    # 로컬 네트워크 공유 기본 비활성화, 로컬 포트 7860 바인딩 실행
    ui_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
