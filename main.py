import os, uuid, asyncio, subprocess
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
JOBS = {}

app = FastAPI(title="BhashaAI API", version="0.2.0")
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/files", StaticFiles(directory=DATA), name="files")

def run_cmd(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "FFmpeg failed")[-1800:])

def split_text(text, limit=1800):
    # Split translated text into manageable TTS chunks.
    words = text.split()
    chunks, current = [], []
    for word in words:
        if len(" ".join(current + [word])) > limit and current:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks

async def process_job(job_id, input_path, target_language, voice, make_subtitles, keep_background):
    job = JOBS[job_id]
    work = DATA / job_id
    work.mkdir(exist_ok=True)
    try:
        job.update(status="processing", progress=5, message="Extracting audio…")
        wav = work / "source.wav"
        run_cmd(["ffmpeg", "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", str(wav)])

        job.update(progress=18, message="Transcribing speech with Whisper…")
        from faster_whisper import WhisperModel
        model_name = os.getenv("WHISPER_MODEL", "small")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(wav), beam_size=5, vad_filter=True)
        source_segments = list(segments)
        transcript = " ".join(s.text.strip() for s in source_segments).strip()
        if not transcript:
            raise RuntimeError("No speech was detected. Try a clearer video with spoken audio.")

        job.update(progress=38, message=f"Translating into {target_language}…")
        from deep_translator import GoogleTranslator
        lang_codes = {"Hindi":"hi", "English":"en", "German":"de", "Spanish":"es",
                      "French":"fr", "Japanese":"ja", "Arabic":"ar"}
        target_code = lang_codes.get(target_language)
        if not target_code:
            raise RuntimeError("This target language is not configured.")
        translator = GoogleTranslator(source="auto", target=target_code)

        # Google Translate can rate-limit long transcripts if sent as one request.
        # Split into smaller chunks and send them sequentially with a delay/retry.
        def split_for_translation(text, limit=3000):
            import re
            sentences = re.split(r"(?<=[.!?।])\\s+", text.strip())
            chunks, current = [], ""
            for sentence in sentences:
                if len(sentence) > limit:
                    words = sentence.split()
                    piece = ""
                    for word in words:
                        if len(piece) + len(word) + 1 > limit and piece:
                            chunks.append(piece)
                            piece = word
                        else:
                            piece = (piece + " " + word).strip()
                    if piece:
                        if current:
                            chunks.append(current)
                            current = ""
                        chunks.append(piece)
                elif len(current) + len(sentence) + 1 > limit and current:
                    chunks.append(current)
                    current = sentence
                else:
                    current = (current + " " + sentence).strip()
            if current:
                chunks.append(current)
            return chunks or [text]

        translated_parts = []
        for index, part in enumerate(split_for_translation(transcript)):
            result = None
            for attempt in range(4):
                try:
                    result = await asyncio.to_thread(translator.translate, part)
                    if result:
                        break
                except Exception:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2 ** (attempt + 1))
                if not result and attempt < 3:
                    await asyncio.sleep(2 ** (attempt + 1))
            if not result:
                raise RuntimeError("Translation service returned an empty result.")
            translated_parts.append(result)
            if index < len(split_for_translation(transcript)) - 1:
                await asyncio.sleep(1.3)  # stay below Google's request-rate limit
        translated = " ".join(translated_parts)
        if not translated:
            raise RuntimeError("Translation service returned an empty result.")

        job.update(progress=55, message="Generating dubbed voice…")
        import edge_tts
        voice_map = {
            "Hindi": {"Natural voice (default)":"hi-IN-SwaraNeural", "Warm and friendly":"hi-IN-SwaraNeural",
                      "Professional":"hi-IN-MadhurNeural", "Energetic":"hi-IN-MadhurNeural"},
            "English": {"Natural voice (default)":"en-IN-NeerjaNeural", "Warm and friendly":"en-IN-NeerjaNeural",
                        "Professional":"en-IN-PrabhatNeural", "Energetic":"en-IN-PrabhatNeural"},
            "German": {"Natural voice (default)":"de-DE-KatjaNeural", "Warm and friendly":"de-DE-KatjaNeural",
                       "Professional":"de-DE-ConradNeural", "Energetic":"de-DE-ConradNeural"}
        }
        selected_voice = voice_map.get(target_language, {}).get(voice)
        if not selected_voice:
            selected_voice = {"Spanish":"es-ES-ElviraNeural", "French":"fr-FR-DeniseNeural",
                              "Japanese":"ja-JP-NanamiNeural", "Arabic":"ar-EG-SalmaNeural"}.get(target_language)
        if not selected_voice:
            raise RuntimeError("No voice configured for this language.")
        chunks = split_text(translated)
        audio_parts = []
        for i, chunk in enumerate(chunks):
            part = work / f"voice_{i:03}.mp3"
            communicate = edge_tts.Communicate(chunk, selected_voice)
            await communicate.save(str(part))
            audio_parts.append(part)
        concat_file = work / "concat.txt"
        concat_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in audio_parts), encoding="utf-8")
        dubbed_audio = work / "dubbed.mp3"
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(dubbed_audio)])

        job.update(progress=75, message="Creating dubbed video…")
        output = DATA / f"{job_id}_dubbed.mp4"
        if keep_background:
            run_cmd(["ffmpeg", "-y", "-i", str(input_path), "-i", str(dubbed_audio),
                     "-filter_complex", "[0:a]volume=0.18[bg];[1:a]volume=1.0[voice];[bg][voice]amix=inputs=2:duration=first:dropout_transition=2[a]",
                     "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                     "-shortest", str(output)])
        else:
            run_cmd(["ffmpeg", "-y", "-i", str(input_path), "-i", str(dubbed_audio),
                     "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                     "-shortest", str(output)])

        subtitle_url = None
        if make_subtitles:
            job.update(progress=88, message="Writing translated subtitles…")
            srt = DATA / f"{job_id}_translated.srt"
            # Basic subtitle file: translated transcript as one caption.
            def stamp(seconds):
                ms = int(seconds * 1000)
                h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"
            duration = float(subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)], text=True).strip())
            srt.write_text(f"1\n{stamp(0)} --> {stamp(duration)}\n{translated}\n", encoding="utf-8")
            subtitle_url = f"/files/{srt.name}"

        job.update(status="completed", progress=100, message="Your dubbed video is ready.",
                   output_url=f"/files/{output.name}", subtitle_url=subtitle_url,
                   transcript=transcript, translation=translated)
    except Exception as exc:
        job.update(status="failed", message=str(exc), progress=100)
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass

@app.get("/api/health")
def health():
    return {"ok": True, "service": "BhashaAI API"}

@app.post("/api/dub")
async def create_dub(background_tasks: BackgroundTasks,
                     video: UploadFile = File(...),
                     target_language: str = Form("Hindi"),
                     voice_style: str = Form("Natural voice (default)"),
                     subtitles: bool = Form(True),
                     keep_background: bool = Form(False)):
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a video file.")
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    if suffix.lower() not in {".mp4", ".mov", ".webm", ".m4v", ".avi"}:
        raise HTTPException(400, "Supported formats: MP4, MOV, WEBM, M4V, AVI.")
    job_id = uuid.uuid4().hex[:12]
    input_path = DATA / f"{job_id}_input{suffix}"
    content = await video.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(413, "Maximum upload size is 500 MB.")
    input_path.write_bytes(content)
    JOBS[job_id] = {"status":"queued", "progress":0, "message":"Job queued"}
    background_tasks.add_task(process_job, job_id, input_path, target_language, voice_style, subtitles, keep_background)
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Project not found.")
    return JOBS[job_id]
