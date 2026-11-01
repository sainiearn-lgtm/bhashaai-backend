import os, uuid, asyncio, subprocess, json, urllib.request, urllib.error
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
JOBS = {}

app = FastAPI(title="BhashaAI API", version="0.3.0")

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


def split_for_translation(text, limit=2800):
    """Split transcript into chunks while keeping sentence boundaries where possible."""
    import re
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
            if len(current) + len(sentence) + 1 > limit and current:
                chunks.append(current)
                current = sentence
            else:
                current = (current + " " + sentence).strip()
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


def gemini_translate(part, source_language, target_language):
    """Translate one text chunk using the Gemini REST API."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in Render > Environment.")

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    prompt = (
        f"Translate the text below from {source_language} into {target_language}. "
        "Preserve the original meaning, names, numbers, and tone. "
        "Return only the translated text. Do not add explanations, notes, or quotation marks.\n\n"
        f"TEXT:\n{part}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Do not expose the API key or request URL in logs/errors.
        try:
            details = json.loads(exc.read().decode("utf-8"))
            message = details.get("error", {}).get("message", "Gemini API request failed.")
        except Exception:
            message = "Gemini API request failed."
        if exc.code == 429:
            raise RuntimeError("Gemini free-tier quota/rate limit reached. Check API limits in Google AI Studio and try again later.")
        if exc.code in (401, 403):
            raise RuntimeError("Gemini API key was rejected. Check GEMINI_API_KEY in Render Environment.")
        if exc.code == 404:
            raise RuntimeError(f"Gemini model '{model}' was not found or is unavailable for this API key. Check GEMINI_MODEL.")
        raise RuntimeError(f"Gemini API error ({exc.code}): {message[:500]}")
    except Exception as exc:
        raise RuntimeError(f"Could not reach Gemini API: {type(exc).__name__}: {str(exc)[:300]}")

    try:
        candidates = result.get("candidates", [])
        translated = candidates[0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        block_reason = result.get("promptFeedback", {}).get("blockReason")
        if block_reason:
            raise RuntimeError(f"Gemini could not process this text (safety block: {block_reason}).")
        raise RuntimeError("Gemini returned an empty or unexpected translation response.")
    if not translated:
        raise RuntimeError("Gemini returned an empty translation.")
    return translated


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
        # Tiny is a safer default for Render's 512 MB free instance.
        model_name = os.getenv("WHISPER_MODEL", "tiny")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(wav), beam_size=5, vad_filter=True)
        source_segments = list(segments)
        transcript = " ".join(s.text.strip() for s in source_segments).strip()
        if not transcript:
            raise RuntimeError("No speech was detected. Try a clearer video with spoken audio.")

        job.update(progress=38, message=f"Translating into {target_language} with Gemini…")
        lang_names = {
            "hi": "Hindi", "en": "English", "de": "German", "es": "Spanish",
            "fr": "French", "ja": "Japanese", "ar": "Arabic"
        }
        target_codes = {
            "Hindi": "hi", "English": "en", "German": "de", "Spanish": "es",
            "French": "fr", "Japanese": "ja", "Arabic": "ar"
        }
        target_code = target_codes.get(target_language)
        if not target_code:
            raise RuntimeError("This target language is not configured.")
        source_code = getattr(info, "language", None) or "unknown"
        source_name = lang_names.get(source_code, f"the detected source language ({source_code})")

        if source_code == target_code:
            translated = transcript
        else:
            parts = split_for_translation(transcript)
            translated_parts = []
            for index, part in enumerate(parts):
                job.update(
                    progress=38 + int((index / max(1, len(parts))) * 14),
                    message=f"Gemini translation: part {index + 1}/{len(parts)}…"
                )
                translated_part = await asyncio.to_thread(
                    gemini_translate, part, source_name, target_language
                )
                translated_parts.append(translated_part)
                # Small pause between chunks to reduce burst rate-limit errors.
                if index < len(parts) - 1:
                    await asyncio.sleep(1.0)
            translated = " ".join(translated_parts).strip()
            if not translated:
                raise RuntimeError("Gemini returned an empty translation.")

        job.update(progress=55, message="Generating dubbed voice…")
        import edge_tts
        voice_map = {
            "Hindi": {"Natural voice (default)": "hi-IN-SwaraNeural", "Warm and friendly": "hi-IN-SwaraNeural",
                      "Professional": "hi-IN-MadhurNeural", "Energetic": "hi-IN-MadhurNeural"},
            "English": {"Natural voice (default)": "en-IN-NeerjaNeural", "Warm and friendly": "en-IN-NeerjaNeural",
                        "Professional": "en-IN-PrabhatNeural", "Energetic": "en-IN-PrabhatNeural"},
            "German": {"Natural voice (default)": "de-DE-KatjaNeural", "Warm and friendly": "de-DE-KatjaNeural",
                       "Professional": "de-DE-ConradNeural", "Energetic": "de-DE-ConradNeural"}
        }
        selected_voice = voice_map.get(target_language, {}).get(voice)
        if not selected_voice:
            selected_voice = {
                "Spanish": "es-ES-ElviraNeural", "French": "fr-FR-DeniseNeural",
                "Japanese": "ja-JP-NanamiNeural", "Arabic": "ar-EG-SalmaNeural"
            }.get(target_language)
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
        concat_file.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in audio_parts),
            encoding="utf-8"
        )
        dubbed_audio = work / "dubbed.mp3"
        run_cmd([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(dubbed_audio)
        ])

        job.update(progress=75, message="Creating dubbed video…")
        output = DATA / f"{job_id}_dubbed.mp4"
        if keep_background:
            run_cmd([
                "ffmpeg", "-y", "-i", str(input_path), "-i", str(dubbed_audio),
                "-filter_complex",
                "[0:a]volume=0.18[bg];[1:a]volume=1.0[voice];[bg][voice]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-shortest", str(output)
            ])
        else:
            run_cmd([
                "ffmpeg", "-y", "-i", str(input_path), "-i", str(dubbed_audio),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                "-shortest", str(output)
            ])

        subtitle_url = None
        if make_subtitles:
            job.update(progress=88, message="Writing translated subtitles…")
            srt = DATA / f"{job_id}_translated.srt"

            def stamp(seconds):
                ms = int(seconds * 1000)
                h, ms = divmod(ms, 3600000)
                m, ms = divmod(ms, 60000)
                s, ms = divmod(ms, 1000)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"

            duration = float(subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
                text=True
            ).strip())
            srt.write_text(
                f"1\n{stamp(0)} --> {stamp(duration)}\n{translated}\n",
                encoding="utf-8"
            )
            subtitle_url = f"/files/{srt.name}"

        job.update(
            status="completed", progress=100, message="Your dubbed video is ready.",
            output_url=f"/files/{output.name}", subtitle_url=subtitle_url,
            transcript=transcript, translation=translated
        )
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
    JOBS[job_id] = {"status": "queued", "progress": 0, "message": "Job queued"}
    background_tasks.add_task(
        process_job, job_id, input_path, target_language, voice_style, subtitles, keep_background
    )
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Project not found.")
    return JOBS[job_id]
