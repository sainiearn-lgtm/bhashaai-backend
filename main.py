import os
import uuid
import json
import asyncio
import sqlite3
import subprocess
import urllib.request
import urllib.error
import threading
import re
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
DATA = Path(os.getenv("DATA_DIR", str(BASE / "data")))
DATA.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "jobs.sqlite3"

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "2"))
TTS_CONCURRENCY = max(1, int(os.getenv("TTS_CONCURRENCY", "3")))

JOBS = {}
JOBS_LOCK = threading.Lock()
WHISPER_MODEL = None
WHISPER_LOCK = threading.Lock()
TTS_SEMAPHORE = asyncio.Semaphore(TTS_CONCURRENCY)

LANG_NAMES = {
    "hi": "Hindi", "en": "English", "de": "German", "es": "Spanish",
    "fr": "French", "ja": "Japanese", "ar": "Arabic"
}
TARGET_CODES = {
    "Hindi": "hi", "English": "en", "German": "de", "Spanish": "es",
    "French": "fr", "Japanese": "ja", "Arabic": "ar"
}
VOICE_MAP = {
    "Hindi": {
        "Natural voice (default)": "hi-IN-SwaraNeural",
        "Warm and friendly": "hi-IN-SwaraNeural",
        "Professional": "hi-IN-MadhurNeural",
        "Energetic": "hi-IN-MadhurNeural",
    },
    "English": {
        "Natural voice (default)": "en-IN-NeerjaNeural",
        "Warm and friendly": "en-IN-NeerjaNeural",
        "Professional": "en-IN-PrabhatNeural",
        "Energetic": "en-IN-PrabhatNeural",
    },
    "German": {
        "Natural voice (default)": "de-DE-KatjaNeural",
        "Warm and friendly": "de-DE-KatjaNeural",
        "Professional": "de-DE-ConradNeural",
        "Energetic": "de-DE-ConradNeural",
    },
}
EXTRA_VOICES = {
    "Spanish": "es-ES-ElviraNeural",
    "French": "fr-FR-DeniseNeural",
    "Japanese": "ja-JP-NanamiNeural",
    "Arabic": "ar-EG-SalmaNeural",
}
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".m4v", ".avi"}


def db():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                message TEXT NOT NULL,
                output_url TEXT,
                subtitle_url TEXT,
                transcript TEXT,
                translation TEXT,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Jobs left processing during a restart cannot resume automatically.
        con.execute("""
            UPDATE jobs SET status='failed',
            message='Server restarted before this job finished. Please upload again.',
            error='Interrupted by server restart.', progress=100,
            updated_at=CURRENT_TIMESTAMP
            WHERE status IN ('queued','processing')
        """)


def save_job(job_id, **changes):
    allowed = {
        "status", "progress", "message", "output_url", "subtitle_url",
        "transcript", "translation", "error"
    }
    changes = {k: v for k, v in changes.items() if k in allowed}
    with db() as con:
        current = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if current is None:
            return
        values = dict(current)
        values.update(changes)
        con.execute("""
            UPDATE jobs SET status=?, progress=?, message=?, output_url=?,
            subtitle_url=?, transcript=?, translation=?, error=?,
            updated_at=CURRENT_TIMESTAMP WHERE job_id=?
        """, (
            values["status"], values["progress"], values["message"],
            values["output_url"], values["subtitle_url"], values["transcript"],
            values["translation"], values["error"], job_id
        ))
    with JOBS_LOCK:
        JOBS[job_id] = get_job(job_id)


def get_job(job_id):
    with db() as con:
        row = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def run_cmd(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout or "FFmpeg failed")[-1500:]
        raise RuntimeError(detail)


def split_for_translation(text, limit=2600):
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            cut = sentence.rfind(" ", 0, limit)
            if cut < 1:
                cut = limit
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            if current and len(current) + len(sentence) + 1 > limit:
                chunks.append(current)
                current = sentence
            else:
                current = (current + " " + sentence).strip()
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


def split_text(text, limit=1800):
    words, chunks, current = text.split(), [], []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > limit and current:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))
    return chunks


def gemini_translate(part, source_language, target_language):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in Render > Environment.")

    models = [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        os.getenv("GEMINI_FALLBACK_MODEL", "").strip(),
    ]
    models = list(dict.fromkeys(m for m in models if m))
    prompt = (
        f"Translate the text below from {source_language} into {target_language}. "
        "Preserve meaning, names, numbers and tone. Return only the translation.\n\n"
        f"TEXT:\n{part}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1}
    }
    last_error = "Gemini translation failed."
    for model in models:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                result = json.loads(response.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            translated = candidates[0]["content"]["parts"][0]["text"].strip()
            if translated:
                return translated
            last_error = "Gemini returned an empty translation."
        except urllib.error.HTTPError as exc:
            try:
                details = json.loads(exc.read().decode("utf-8"))
                message = details.get("error", {}).get("message", "Gemini API request failed.")
            except Exception:
                message = "Gemini API request failed."
            if exc.code == 429:
                last_error = "Gemini quota/rate limit reached."
                # A second model may have a separate quota/availability.
                continue
            if exc.code in (401, 403):
                raise RuntimeError("Gemini API key was rejected. Check GEMINI_API_KEY in Render Environment.")
            if exc.code == 404:
                last_error = f"Gemini model '{model}' is unavailable for this API key."
                continue
            raise RuntimeError(f"Gemini API error ({exc.code}): {message[:350]}")
        except (KeyError, IndexError, TypeError):
            last_error = "Gemini returned an unexpected response."
        except Exception as exc:
            last_error = f"Could not reach Gemini API: {type(exc).__name__}: {str(exc)[:200]}"
    raise RuntimeError(last_error + " Check GEMINI_MODEL / GEMINI_FALLBACK_MODEL and API quota.")


def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        with WHISPER_LOCK:
            if WHISPER_MODEL is None:
                from faster_whisper import WhisperModel
                WHISPER_MODEL = WhisperModel(
                    WHISPER_MODEL_NAME,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=WHISPER_CPU_THREADS,
                    num_workers=1,
                )
    return WHISPER_MODEL


def update(job_id, **changes):
    save_job(job_id, **changes)


async def create_tts_part(text, voice, path):
    import edge_tts
    async with TTS_SEMAPHORE:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(path))


async def process_job(job_id, input_path, target_language, voice_style, make_subtitles, keep_background):
    work = DATA / job_id
    work.mkdir(exist_ok=True)
    try:
        update(job_id, status="processing", progress=3, message="Extracting audio…")
        wav = work / "source.wav"
        await asyncio.to_thread(
            run_cmd,
            ["ffmpeg", "-nostdin", "-y", "-i", str(input_path), "-vn",
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
        )

        update(job_id, progress=15, message="Loading speech recognition model…")
        model = await asyncio.to_thread(get_whisper_model)
        update(job_id, progress=20, message="Transcribing speech…")
        segments, info = await asyncio.to_thread(
            lambda: model.transcribe(
                str(wav), beam_size=1, best_of=1, vad_filter=True,
                condition_on_previous_text=False
            )
        )
        source_segments = await asyncio.to_thread(list, segments)
        transcript = " ".join(s.text.strip() for s in source_segments).strip()
        if not transcript:
            raise RuntimeError("No speech was detected. Try a clearer video with spoken audio.")

        source_code = getattr(info, "language", None) or "unknown"
        source_name = LANG_NAMES.get(source_code, f"detected language ({source_code})")
        target_code = TARGET_CODES.get(target_language)
        if not target_code:
            raise RuntimeError("Unsupported target language.")
        update(job_id, progress=36, message=f"Translating into {target_language}…")

        if source_code == target_code:
            translated = transcript
        else:
            parts = split_for_translation(transcript)
            translated_parts = []
            for i, part in enumerate(parts):
                update(
                    job_id,
                    progress=36 + int((i / max(1, len(parts))) * 16),
                    message=f"Translating part {i + 1}/{len(parts)}…"
                )
                translated_parts.append(await asyncio.to_thread(
                    gemini_translate, part, source_name, target_language
                ))
                # Avoid bursts against free-tier rate limits.
                if i < len(parts) - 1:
                    await asyncio.sleep(0.4)
            translated = " ".join(translated_parts).strip()

        voice = VOICE_MAP.get(target_language, {}).get(voice_style) or EXTRA_VOICES.get(target_language)
        if not voice:
            raise RuntimeError("No voice configured for this language.")

        update(job_id, progress=55, message="Generating dubbed voice…")
        chunks = split_text(translated)
        audio_parts = [work / f"voice_{i:03}.mp3" for i in range(len(chunks))]
        await asyncio.gather(*[
            create_tts_part(chunk, voice, path)
            for chunk, path in zip(chunks, audio_parts)
        ])

        concat_file = work / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in audio_parts),
            encoding="utf-8"
        )
        dubbed_audio = work / "dubbed.mp3"
        await asyncio.to_thread(run_cmd, [
            "ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", str(dubbed_audio)
        ])

        update(job_id, progress=76, message="Creating final video…")
        output = DATA / f"{job_id}_dubbed.mp4"
        if keep_background:
            cmd = [
                "ffmpeg", "-nostdin", "-y", "-i", str(input_path), "-i", str(dubbed_audio),
                "-filter_complex",
                "[0:a]volume=0.16[bg];[1:a]volume=1.0[voice];[bg][voice]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "128k", "-shortest", str(output)
            ]
        else:
            cmd = [
                "ffmpeg", "-nostdin", "-y", "-i", str(input_path), "-i", str(dubbed_audio),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "128k", "-shortest", str(output)
            ]
        await asyncio.to_thread(run_cmd, cmd)

        subtitle_url = None
        if make_subtitles:
            update(job_id, progress=91, message="Writing translated subtitles…")
            srt = DATA / f"{job_id}_translated.srt"
            duration_text = await asyncio.to_thread(
                subprocess.check_output,
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(output)],
                text=True
            )
            duration = max(1.0, float(duration_text.strip()))
            ms = int(duration * 1000)
            h, ms = divmod(ms, 3600000)
            m, ms = divmod(ms, 60000)
            s, ms = divmod(ms, 1000)
            stamp = f"{h:02}:{m:02}:{s:02},{ms:03}"
            srt.write_text(f"1\n00:00:00,000 --> {stamp}\n{translated}\n", encoding="utf-8")
            subtitle_url = f"/files/{srt.name}"

        update(
            job_id, status="completed", progress=100,
            message="Your dubbed video is ready.",
            output_url=f"/files/{output.name}",
            subtitle_url=subtitle_url, transcript=transcript, translation=translated,
            error=None
        )
    except Exception as exc:
        update(job_id, status="failed", progress=100, message=str(exc), error=str(exc))
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="BhashaAI API", version="0.4.0", lifespan=lifespan)

origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=DATA), name="files")


@app.get("/")
def root():
    return {"service": "BhashaAI API", "version": "0.4.0", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {"ok": True, "service": "BhashaAI API", "version": "0.4.0"}


@app.post("/api/dub")
async def create_dub(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    target_language: str = Form("Hindi"),
    voice_style: str = Form("Natural voice (default)"),
    subtitles: bool = Form(True),
    keep_background: bool = Form(False),
):
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a video file.")
    suffix = Path(video.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Supported formats: MP4, MOV, WEBM, M4V, AVI.")

    job_id = uuid.uuid4().hex[:12]
    input_path = DATA / f"{job_id}_input{suffix}"
    total = 0
    try:
        with input_path.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
                out.write(chunk)
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()

    with db() as con:
        con.execute("""
            INSERT INTO jobs (job_id,status,progress,message)
            VALUES (?, 'queued', 0, 'Video uploaded. Job queued.')
        """, (job_id,))
    with JOBS_LOCK:
        JOBS[job_id] = get_job(job_id)

    background_tasks.add_task(
        process_job, job_id, input_path, target_language, voice_style,
        subtitles, keep_background
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Project not found.")
    return job
