import base64
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from typing import List, Optional

import requests
import yt_dlp
from fastapi import FastAPI, Header, HTTPException
from faster_whisper import WhisperModel
from pydantic import BaseModel

RENDER_API_KEY = os.environ.get("RENDER_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "reels")
YOUTUBE_COOKIES_B64 = os.environ.get("YOUTUBE_COOKIES_B64", "")
POT_PROVIDER_URL = os.environ.get("POT_PROVIDER_URL", "")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
WIDTH, HEIGHT = 1080, 1920
CAPTION_WORDS_PER_LINE = 4
CAPTION_FONT_SIZE = 90
CAPTION_HIGHLIGHT_COLOR = "&H0000FFFF"  # ASS BGR: yellow
CAPTION_BASE_COLOR = "&H00FFFFFF"  # white

app = FastAPI()
_whisper_model: Optional[WhisperModel] = None
_youtube_cookies_path: Optional[str] = None
CLIPS_JOBS: dict = {}


def _get_youtube_cookies_path() -> Optional[str]:
    global _youtube_cookies_path
    if not YOUTUBE_COOKIES_B64:
        return None
    if _youtube_cookies_path is None:
        path = "/tmp/youtube_cookies.txt"
        with open(path, "wb") as f:
            f.write(base64.b64decode(YOUTUBE_COOKIES_B64))
        _youtube_cookies_path = path
    return _youtube_cookies_path


def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "youtube_cookies_configured": bool(YOUTUBE_COOKIES_B64),
        "youtube_cookies_len": len(YOUTUBE_COOKIES_B64),
        "pot_provider_url": POT_PROVIDER_URL or None,
    }


class PrepareClipsRequest(BaseModel):
    video_url: str


def _is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def _run_prepare_job(job_id: str, video_url: str) -> None:
    workdir = tempfile.mkdtemp(prefix="clips_prepare_")
    try:
        if _is_youtube_url(video_url):
            outtmpl = os.path.join(workdir, "source.%(ext)s")
            ydl_opts = {
                "outtmpl": outtmpl,
                "format": "best",
                "merge_output_format": "mp4",
                "quiet": True,
                "no_warnings": True,
            }
            cookies_path = _get_youtube_cookies_path()
            if cookies_path:
                ydl_opts["cookiefile"] = cookies_path
            if POT_PROVIDER_URL:
                ydl_opts["extractor_args"] = {"youtubepot-bgutilhttp": {"base_url": [POT_PROVIDER_URL]}}

            # YouTube's anti-bot format-serving is flaky right now (SABR streaming rollout) --
            # success rate varies attempt to attempt with no change in request. Retry a few
            # times before giving up rather than failing on the first unlucky attempt.
            last_error: Optional[Exception] = None
            for attempt in range(5):
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video_url])
                    last_error = None
                    break
                except yt_dlp.utils.DownloadError as e:
                    last_error = e
                    time.sleep(3)
                    continue
            if last_error is not None:
                raise RuntimeError(f"yt-dlp failed after retries: {last_error}")

            candidates = [f for f in os.listdir(workdir) if f.startswith("source.")]
            if not candidates:
                raise RuntimeError("yt-dlp did not produce an output file")
            source_path = os.path.join(workdir, candidates[0])
        else:
            # Video already hosted somewhere we control (e.g. manually uploaded to
            # Supabase Storage) -- just download it directly, no yt-dlp needed.
            source_path = os.path.join(workdir, "source" + _guess_ext(video_url))
            _download(video_url, source_path)

        duration = _probe_duration(source_path)

        model = _get_whisper_model()
        raw_segments, _ = model.transcribe(source_path, word_timestamps=True, vad_filter=True)

        segments = []
        words = []
        for seg in raw_segments:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            for w in (seg.words or []):
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

        # If the input was already a file we host (e.g. manually uploaded to
        # cortes-inbox/), don't re-upload a second copy -- just reuse that URL.
        if video_url.startswith(SUPABASE_URL):
            source_url = video_url
        else:
            source_url = _upload_to_supabase(source_path, folder="cortes-source/", ext=".mp4", content_type="video/mp4")
        CLIPS_JOBS[job_id] = {
            "status": "done",
            "result": {"source_url": source_url, "duration": duration, "segments": segments, "words": words},
        }
    except Exception as e:
        CLIPS_JOBS[job_id] = {"status": "error", "error": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/clips/prepare")
def clips_prepare(req: PrepareClipsRequest, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    job_id = uuid.uuid4().hex
    CLIPS_JOBS[job_id] = {"status": "processing"}
    threading.Thread(target=_run_prepare_job, args=(job_id, req.video_url), daemon=True).start()
    return {"job_id": job_id}


@app.get("/clips/status/{job_id}")
def clips_status(job_id: str, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    job = CLIPS_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


class RenderClipRequest(BaseModel):
    video_url: str
    start: float
    end: float
    words: List[dict] = []


@app.post("/clips/render")
def clips_render(req: RenderClipRequest, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    if req.end <= req.start:
        raise HTTPException(status_code=422, detail="end must be greater than start")

    workdir = tempfile.mkdtemp(prefix="clips_render_")
    try:
        source_path = os.path.join(workdir, "source" + _guess_ext(req.video_url))
        output_path = os.path.join(workdir, f"{uuid.uuid4().hex}.mp4")
        ass_path = os.path.join(workdir, "captions.ass")
        _download(req.video_url, source_path)

        clip_duration = req.end - req.start
        relative_words = [
            {"word": w["word"], "start": max(0.0, w["start"] - req.start), "end": max(0.0, w["end"] - req.start)}
            for w in req.words
            if w["end"] > req.start and w["start"] < req.end
        ]
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(_build_ass_karaoke(relative_words))

        # Simple v1 reframing: scale to fill the vertical frame height, then center-crop the
        # width. No active-speaker tracking yet -- that's the planned next iteration.
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(req.start), "-i", source_path, "-t", str(clip_duration),
            "-vf",
            f"scale=-2:{HEIGHT},crop={WIDTH}:{HEIGHT},"
            f"eq=contrast=1.1:brightness=-0.03:saturation=0.9,"
            f"subtitles={ass_path}:fontsdir=/app/fonts",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg (clip render) failed: {result.stderr[-2000:]}")

        video_url = _upload_to_supabase(output_path, folder="cortes/", ext=".mp4", content_type="video/mp4")
        shutil.rmtree(workdir, ignore_errors=True)
        return {"video_url": video_url}
    except HTTPException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


def _seconds_to_ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _build_ass_karaoke(words: list[dict]) -> str:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {WIDTH}\n"
        f"PlayResY: {HEIGHT}\n"
        "WrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Karaoke,Poppins,{CAPTION_FONT_SIZE},{CAPTION_HIGHLIGHT_COLOR},{CAPTION_BASE_COLOR},"
        "&H00000000,&H00000000,1,0,1,4,2,2,60,60,300,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for i in range(0, len(words), CAPTION_WORDS_PER_LINE):
        chunk = words[i:i + CAPTION_WORDS_PER_LINE]
        if not chunk:
            continue
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        text = " ".join(
            f"{{\\k{max(1, int(round((w['end'] - w['start']) * 100)))}}}{w['word']}" for w in chunk
        )
        lines.append(f"Dialogue: 0,{_seconds_to_ass_time(start)},{_seconds_to_ass_time(end)},Karaoke,,0,0,0,,{text}")
    return header + "\n".join(lines) + "\n"


def _upload_to_supabase(
    file_path: str, folder: str = "", ext: str = ".mp4", content_type: str = "video/mp4"
) -> str:
    filename = f"{folder}{uuid.uuid4().hex}{ext}"
    with open(file_path, "rb") as f:
        r = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "apikey": SUPABASE_SERVICE_KEY,
                "Content-Type": content_type,
            },
            data=f,
            timeout=60,
        )
    if r.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"supabase upload failed: {r.text}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"


def _guess_ext(url: str) -> str:
    path = url.split("?")[0]
    ext = os.path.splitext(path)[1]
    return ext if ext else ".mp3"


def _download(url: str, dest: str) -> None:
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)


def _probe_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {result.stderr[-500:]}")
    return float(result.stdout.strip())
