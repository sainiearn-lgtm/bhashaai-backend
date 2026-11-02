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
DIARIZATION_PIPELINE = None
DIARIZATION_LOCK = threading.Lock()
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


def get_diarization_pipeline():
    """Load the gated pyannote speaker-diarization model once per server process."""
    global DIARIZATION_PIPELINE
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return None

    if DIARIZATION_PIPELINE is None:
        with DIARIZATION_LOCK:
            if DIARIZATION_PIPELINE is None:
                from pyannote.audio import Pipeline
                DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    token=token
                )
                # Render CPU deployment: keep inference on CPU.
                try:
                    import torch
                    DIARIZATION_PIPELINE.to(torch.device("cpu"))
                except Exception:
                    pass
    return DIARIZATION_PIPELINE


def build_speaker_turns(whisper_segments, diarization=None):
    """
    Convert Whisper word timestamps into consecutive speaker turns.
    If diarization is unavailable, use Whisper segment timestamps as one speaker.
    """
    diarized_regions = []
    if diarization is not None:
        annotation = getattr(diarization, "speaker_diarization", diarization)
        for region, _, speaker in annotation.itertracks(yield_label=True):
            diarized_regions.append((float(region.start), float(region.end), str(speaker)))

    words = []
    for segment in whisper_segments:
        segment_start = float(getattr(segment, "start", 0.0))
        segment_end = float(getattr(segment, "end", segment_start + 1.0))
        segment_words = getattr(segment, "words", None) or []

        if segment_words:
            for word in segment_words:
                word_start = float(getattr(word, "start", segment_start))
                word_end = float(getattr(word, "end", segment_end))
                word_text = str(getattr(word, "word", "")).strip()
                if word_text:
                    words.append((word_start, max(word_start + 0.01, word_end), word_text))
        else:
            text = str(getattr(segment, "text", "")).strip()
            if text:
                words.append((segment_start, max(segment_start + 0.01, segment_end), text))

    turns = []
    for start, end, word_text in words:
        speaker = "SPEAKER_00"
        if diarized_regions:
            midpoint = (start + end) / 2
            overlaps = [
                (max(0.0, min(end, region_end) - max(start, region_start)), label)
                for region_start, region_end, label in diarized_regions
                if region_start <= midpoint <= region_end
                or min(end, region_end) > max(start, region_start)
            ]
            if overlaps:
                speaker = max(overlaps, key=lambda item: item[0])[1]

        if turns and turns[-1]["speaker"] == speaker and start - turns[-1]["end"] <= 1.25:
            turns[-1]["text"] = (turns[-1]["text"] + " " + word_text).strip()
            turns[-1]["end"] = end
        else:
            turns.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": word_text
            })

    return [turn for turn in turns if turn["text"].strip()]


def choose_speaker_voice(target_language, speaker_number, voice_style):
    """
    Assign distinct configured voices automatically by speaker order.
    Current language voice sets contain up to two distinct voices; additional
    speakers cycle through those voices.
    """
    language_voices = VOICE_MAP.get(target_language, {})
    preferred = language_voices.get(voice_style)
    if target_language in ("Hindi", "English", "German"):
        if target_language == "Hindi":
            pool = ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"]
        elif target_language == "English":
            pool = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"]
        else:
            pool = ["de-DE-KatjaNeural", "de-DE-ConradNeural"]
        # Keep the selected style's first-speaker voice when possible.
        if preferred in pool:
            pool = [preferred] + [v for v in pool if v != preferred]
        return pool[speaker_number % len(pool)]
    return EXTRA_VOICES.get(target_language)


def update(job_id, **changes):
    save_job(job_id, **changes)


def get_voice_tuning(voice_style):
    settings = {
        "Natural voice (default)": {"rate": "+0%", "pitch": "+0Hz"},
        "Warm and friendly": {"rate": "-4%", "pitch": "+1Hz"},
        "Professional": {"rate": "-5%", "pitch": "-1Hz"},
        "Energetic": {"rate": "+5%", "pitch": "+1Hz"},
    }
    return settings.get(
        voice_style,
        settings["Natural voice (default)"]
    )


async def create_tts_part(text, voice, path, voice_style="Natural voice (default)"):
    import edge_tts

    tuning = get_voice_tuning(voice_style)
    async with TTS_SEMAPHORE:
        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=tuning["rate"],
            pitch=tuning["pitch"],
            volume="+0%"
        )
        await communicate.save(str(path))


def subtitle_stamp(seconds):
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def make_caption_chunks(text, max_chars=76):
    sentences = [x.strip() for x in re.split(r"(?<=[.!?।؟])\s+", text.strip()) if x.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    captions = []
    for sentence in sentences:
        current = ""
        for word in sentence.split():
            candidate = f"{current} {word}".strip()
            if len(candidate) > max_chars and current:
                captions.append(current)
                current = word
            else:
                current = candidate
        if current:
            captions.append(current)
    return captions


def write_turn_subtitles(turns, audio_parts, srt_path):
    """Time translated captions to each dubbed speaker clip and its audio delay."""
    entries = []
    for turn, (audio_path, delay_ms) in zip(turns, audio_parts):
        duration_text = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            text=True
        )
        clip_duration = max(0.25, float(duration_text.strip()))
        captions = make_caption_chunks(turn.get("translation", ""))
        if not captions:
            continue
        weights = [max(1, len(item)) for item in captions]
        total_weight = sum(weights)
        cursor = delay_ms / 1000.0
        for index, (caption, weight) in enumerate(zip(captions, weights)):
            start = cursor
            end = delay_ms / 1000.0 + clip_duration if index == len(captions) - 1 else cursor + clip_duration * weight / total_weight
            entries.append((start, end, caption))
            cursor = end
    entries.sort(key=lambda item: (item[0], item[1]))
    if not entries:
        return False
    lines = []
    for index, (start, end, caption) in enumerate(entries, start=1):
        lines.extend([str(index), f"{subtitle_stamp(start)} --> {subtitle_stamp(end)}", caption, ""])
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return True


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
                condition_on_previous_text=False, word_timestamps=True
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

        update(job_id, progress=28, message="Detecting speakers…")
        diarization = None
        try:
            pipeline = await asyncio.to_thread(get_diarization_pipeline)
            if pipeline is not None:
                diarization = await asyncio.to_thread(pipeline, str(wav))
        except Exception as exc:
            # Keep the dubbing workflow usable if the optional gated model is
            # not configured or cannot load on the current hosting instance.
            update(
                job_id, progress=30,
                message=f"Speaker detection unavailable; using one voice ({type(exc).__name__})."
            )

        turns = build_speaker_turns(source_segments, diarization)
        if not turns:
            turns = [{
                "speaker": "SPEAKER_00", "start": 0.0,
                "end": 1.0, "text": transcript
            }]

        speaker_order = []
        for turn in turns:
            if turn["speaker"] not in speaker_order:
                speaker_order.append(turn["speaker"])
        speaker_numbers = {name: i for i, name in enumerate(speaker_order)}

        update(job_id, progress=36, message=f"Translating {len(turns)} dialogue turns…")
        for i, turn in enumerate(turns):
            if source_code == target_code:
                turn["translation"] = turn["text"]
            else:
                turn["translation"] = await asyncio.to_thread(
                    gemini_translate, turn["text"], source_name, target_language
                )
            update(
                job_id,
                progress=36 + int(((i + 1) / max(1, len(turns))) * 18),
                message=f"Translating dialogue {i + 1}/{len(turns)}…"
            )
            if i < len(turns) - 1 and source_code != target_code:
                await asyncio.sleep(0.25)

        update(
            job_id, progress=55,
            message=f"Generating voices for {len(speaker_order)} speaker(s)…"
        )
        audio_parts = []
        for i, turn in enumerate(turns):
            voice = choose_speaker_voice(
                target_language,
                speaker_numbers[turn["speaker"]],
                voice_style
            )
            if not voice:
                raise RuntimeError("No voice configured for this language.")
            audio_path = work / f"speaker_turn_{i:04}.mp3"
            await create_tts_part(
                turn["translation"], voice, audio_path, voice_style
            )
            audio_parts.append((audio_path, max(0, int(turn["start"] * 1000))))

        # Place each translated turn at its original speech start time.
        # This preserves conversational order and reduces speaker overlap.
        filter_parts = []
        for i, (_, delay_ms) in enumerate(audio_parts):
            filter_parts.append(
                f"[{i}:a]adelay={delay_ms}|{delay_ms},volume=1.0[a{i}]"
            )
        mix_inputs = "".join(f"[a{i}]" for i in range(len(audio_parts)))
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(audio_parts)}:duration=longest:"
            "dropout_transition=0[dub]"
        )
        dubbed_audio = work / "dubbed.m4a"
        mix_cmd = ["ffmpeg", "-nostdin", "-y"]
        for audio_path, _ in audio_parts:
            mix_cmd.extend(["-i", str(audio_path)])
        mix_cmd.extend([
            "-filter_complex", ";".join(filter_parts),
            "-map", "[dub]", "-c:a", "aac", "-b:a", "128k",
            str(dubbed_audio)
        ])
        await asyncio.to_thread(run_cmd, mix_cmd)

        # Keep a readable combined translation in the job result.
        translated = "\n".join(
            f'{turn["speaker"]}: {turn["translation"]}' for turn in turns
        )

        subtitle_filter = None
        if make_subtitles:
            update(job_id, progress=84, message="Preparing dubbed-language subtitles…")
            srt = work / "dubbed_subtitles.srt"
            if write_turn_subtitles(turns, audio_parts, srt):
                style = (
                    "FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,"
                    "OutlineColour=&H00000000,BorderStyle=1,Outline=2,"
                    "Shadow=1,Alignment=2,MarginV=28"
                )
                subtitle_filter = f"subtitles=filename='{srt.as_posix()}':force_style='{style}'"

        update(job_id, progress=90 if make_subtitles else 84, message="Rendering final video…")
        output = DATA / f"{job_id}_dubbed.mp4"
        if keep_background:
            cmd = [
                "ffmpeg", "-nostdin", "-y", "-i", str(input_path), "-i", str(dubbed_audio),
                "-filter_complex",
                "[0:a]volume=0.16[bg];[1:a]volume=1.0[voice];[bg][voice]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map", "0:v:0", "-map", "[a]"
            ]
        else:
            cmd = [
                "ffmpeg", "-nostdin", "-y", "-i", str(input_path), "-i", str(dubbed_audio),
                "-map", "0:v:0", "-map", "1:a:0"
            ]
        if subtitle_filter:
            cmd += ["-vf", subtitle_filter]
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", str(output)
        ]
        await asyncio.to_thread(run_cmd, cmd)

        update(
            job_id, status="completed", progress=100,
            message="Your dubbed video is ready.",
            output_url=f"/files/{output.name}",
            subtitle_url=None, transcript=transcript, translation=translated,
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


app = FastAPI(title="BhashaAI API", version="0.4.1", lifespan=lifespan)

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
    return {"service": "BhashaAI API", "version": "0.4.1", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {"ok": True, "service": "BhashaAI API", "version": "0.4.1"}


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
