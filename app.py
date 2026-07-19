import base64
import math
import os
import random
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
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel

RENDER_API_KEY = os.environ.get("RENDER_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "reels")
KNOWN_HOSTED_PREFIXES = (SUPABASE_URL, "https://pub-66667b109a704a8f8f0a89c4f2ce426d.r2.dev")
YOUTUBE_COOKIES_B64 = os.environ.get("YOUTUBE_COOKIES_B64", "")
POT_PROVIDER_URL = os.environ.get("POT_PROVIDER_URL", "")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
WIDTH, HEIGHT = 1080, 1920
THUMB_FONT_PATH = "/app/fonts/Poppins-Bold.ttf"
THUMB_FONT_SIZE = 110
CAPTION_WORDS_PER_LINE = 2
CAPTION_FONT_SIZE = 130
CAPTION_BASE_COLOR = "&H00FFFFFF"  # white -- the "not yet spoken" color, always white
CAPTION_HIGHLIGHT_COLORS = {
    # ASS BGR format (&H00BBGGRR). Chosen by the highlight-selection LLM per clip mood.
    "branco": "&H00FFFFFF",
    "amarelo": "&H0000FFFF",
    "vermelho": "&H000000FF",
    "ciano": "&H00FFFF00",
    "laranja": "&H00008CFF",
    "rosa": "&H009314FF",
    "verde": "&H0014FF39",
}
DEFAULT_HIGHLIGHT_COLOR = "&H0000FFFF"  # yellow

app = FastAPI()
_whisper_model: Optional[WhisperModel] = None
_youtube_cookies_path: Optional[str] = None
CLIPS_JOBS: dict = {}
CLIPS_RENDER_JOBS: dict = {}
STORY_JOBS: dict = {}


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
        "whisper_model_size": WHISPER_MODEL_SIZE,
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
        if video_url.startswith(KNOWN_HOSTED_PREFIXES):
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
    highlight_color: Optional[str] = None


def _run_render_job(job_id: str, req: "RenderClipRequest") -> None:
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
        highlight_color = CAPTION_HIGHLIGHT_COLORS.get(
            (req.highlight_color or "").strip().lower(), DEFAULT_HIGHLIGHT_COLOR
        )
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(_build_ass_karaoke(relative_words, highlight_color))

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
            raise RuntimeError(f"ffmpeg (clip render) failed: {result.stderr[-2000:]}")

        video_url = _upload_to_supabase(output_path, folder="cortes/", ext=".mp4", content_type="video/mp4")
        CLIPS_RENDER_JOBS[job_id] = {"status": "done", "result": {"video_url": video_url}}
    except Exception as e:
        CLIPS_RENDER_JOBS[job_id] = {"status": "error", "error": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/clips/render")
def clips_render(req: RenderClipRequest, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    if req.end <= req.start:
        raise HTTPException(status_code=422, detail="end must be greater than start")

    job_id = uuid.uuid4().hex
    CLIPS_RENDER_JOBS[job_id] = {"status": "processing"}
    threading.Thread(target=_run_render_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/clips/render-status/{job_id}")
def clips_render_status(job_id: str, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    job = CLIPS_RENDER_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


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


def _build_ass_karaoke(words: list[dict], highlight_color: str = DEFAULT_HIGHLIGHT_COLOR) -> str:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {WIDTH}\n"
        f"PlayResY: {HEIGHT}\n"
        "WrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Karaoke,Poppins,{CAPTION_FONT_SIZE},{highlight_color},{CAPTION_BASE_COLOR},"
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


def _download_image_with_retry(url: str, dest: str, attempts: int = 5) -> None:
    # Pollinations' free tier rate-limits bursts of requests (429) and is occasionally
    # flaky (timeouts, connection resets, transient 5xx) -- back off and retry rather
    # than failing the whole render over what's usually a one-off hiccup.
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            _download(url, dest)
            return
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else None
            if status == 429 or (status is not None and status >= 500):
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            time.sleep(5 * (attempt + 1))
            continue
    raise last_error


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


class StoryRenderRequest(BaseModel):
    audio_url: str
    image_urls: List[str]
    highlight_color: Optional[str] = None
    music_url: Optional[str] = None


def _render_ken_burns_segment(image_path: str, out_path: str, seg_duration: float) -> None:
    zoom_w, zoom_h = WIDTH * 2, HEIGHT * 2
    total_frames = max(1, int(round(seg_duration * 30)))
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-vf",
        f"scale={zoom_w}:{zoom_h}:force_original_aspect_ratio=increase,"
        f"crop={zoom_w}:{zoom_h},"
        f"zoompan=z='min(zoom+0.0015,1.3)':d={total_frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={WIDTH}x{HEIGHT}:fps=30,"
        "setsar=1",
        "-t", str(seg_duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg (ken burns segment) failed: {result.stderr[-1500:]}")


def _run_story_job(job_id: str, req: "StoryRenderRequest") -> None:
    workdir = tempfile.mkdtemp(prefix="story_render_")
    try:
        audio_path = os.path.join(workdir, "narration" + _guess_ext(req.audio_url))
        _download(req.audio_url, audio_path)
        duration = _probe_duration(audio_path)

        model = _get_whisper_model()
        raw_segments, _ = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
        words = []
        for seg in raw_segments:
            for w in (seg.words or []):
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

        n = len(req.image_urls)
        seg_duration = max(0.5, duration / n)
        segment_paths = []
        for i, url in enumerate(req.image_urls):
            img_path = os.path.join(workdir, f"scene_{i}.jpg")
            _download_image_with_retry(url, img_path)
            seg_path = os.path.join(workdir, f"scene_seg_{i}.mp4")
            _render_ken_burns_segment(img_path, seg_path, seg_duration)
            segment_paths.append(seg_path)
            if i < len(req.image_urls) - 1:
                time.sleep(3)  # Pollinations' free tier rate-limits rapid back-to-back requests

        concat_list_path = os.path.join(workdir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for p in segment_paths:
                f.write(f"file '{p}'\n")

        ass_path = os.path.join(workdir, "captions.ass")
        highlight_color = CAPTION_HIGHLIGHT_COLORS.get(
            (req.highlight_color or "").strip().lower(), DEFAULT_HIGHLIGHT_COLOR
        )
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(_build_ass_karaoke(words, highlight_color))

        output_path = os.path.join(workdir, f"{uuid.uuid4().hex}.mp4")
        fade_out_start = max(duration - 1, 0)

        if req.music_url:
            music_path = os.path.join(workdir, "music" + _guess_ext(req.music_url))
            _download(req.music_url, music_path)
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list_path,
                "-i", audio_path,
                "-i", music_path,
                "-filter_complex",
                f"[0:v]subtitles={ass_path}:fontsdir=/app/fonts[vout];"
                f"[2:a]aloop=loop=-1:size=2e9,atrim=0:{duration},volume=0.15,"
                f"afade=t=in:st=0:d=1,afade=t=out:st={fade_out_start}:d=1[music];"
                f"[1:a][music]amix=inputs=2:duration=first:dropout_transition=0[aout]",
                "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                "-c:a", "aac", "-b:a", "128k",
                "-t", str(duration),
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_list_path,
                "-i", audio_path,
                "-vf", f"subtitles={ass_path}:fontsdir=/app/fonts",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest",
                output_path,
            ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg (story final) failed: {result.stderr[-2000:]}")

        video_url = _upload_to_supabase(output_path, folder="story/", ext=".mp4", content_type="video/mp4")
        STORY_JOBS[job_id] = {"status": "done", "result": {"video_url": video_url, "duration": duration}}
    except Exception as e:
        STORY_JOBS[job_id] = {"status": "error", "error": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/story/render")
def story_render(req: StoryRenderRequest, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    if not req.image_urls:
        raise HTTPException(status_code=422, detail="image_urls is required")

    job_id = uuid.uuid4().hex
    STORY_JOBS[job_id] = {"status": "processing"}
    threading.Thread(target=_run_story_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/story/render-status/{job_id}")
def story_render_status(job_id: str, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    job = STORY_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


class ThumbnailRequest(BaseModel):
    image_url: str
    title: str


def _wrap_text(draw: "ImageDraw.ImageDraw", text: str, font: "ImageFont.FreeTypeFont", max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if not current or draw.textlength(trial, font=font) <= max_width:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _build_thumbnail(image_path: str, title: str, out_path: str) -> None:
    img = ImageOps.fit(Image.open(image_path).convert("RGB"), (WIDTH, HEIGHT), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(THUMB_FONT_PATH, THUMB_FONT_SIZE)
    max_width = int(WIDTH * 0.88)
    lines = _wrap_text(draw, title.upper(), font, max_width)

    ref_bbox = font.getbbox("Ág")
    line_height = (ref_bbox[3] - ref_bbox[1]) + 28
    total_height = line_height * len(lines)
    y = HEIGHT - total_height - 160
    band_top = y - 50

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([0, band_top, WIDTH, HEIGHT], fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    for line in lines:
        w = draw.textlength(line, font=font)
        x = (WIDTH - w) / 2
        draw.text((x, y), line, font=font, fill="white", stroke_width=7, stroke_fill="black")
        y += line_height

    img.save(out_path, "JPEG", quality=92)


@app.post("/story/thumbnail")
def story_thumbnail(req: ThumbnailRequest, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    workdir = tempfile.mkdtemp(prefix="thumb_")
    try:
        image_path = os.path.join(workdir, "cover" + _guess_ext(req.image_url))
        _download_image_with_retry(req.image_url, image_path)
        out_path = os.path.join(workdir, f"{uuid.uuid4().hex}.jpg")
        _build_thumbnail(image_path, req.title, out_path)
        thumbnail_url = _upload_to_supabase(out_path, folder="thumbs-biblia/", ext=".jpg", content_type="image/jpeg")
        return {"thumbnail_url": thumbnail_url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


ASSETS_DIR = "/app/assets"
BACKGROUND_SEGMENT_SECONDS = 10.0  # background clips are cycled in chunks this long
NARRATION_SPEED = 1.3
LIKE_DISPLAY_SECONDS = 4.67  # like_greenscreen.mp4's own duration; plays once, no loop/cut
SUBSCRIBE_DISPLAY_SECONDS = 4.95  # subscribe_greenscreen.mp4's own duration; plays once, no loop/cut
LIKE_CHROMA_COLOR = "0x00F90E"
SUBSCRIBE_CHROMA_COLOR = "0x26FF11"

MUSIC_DIR = os.path.join(ASSETS_DIR, "music")
MUSIC_VOLUME = 0.12  # low background bed, stays well under the narration
MUSIC_FADE_SECONDS = 1.0

SATISFYING_JOBS: dict = {}


class SatisfyingRenderRequest(BaseModel):
    narration_url: str
    video_urls: List[str]


def _speed_up_audio(src_path: str, out_path: str, speed: float) -> float:
    cmd = ["ffmpeg", "-y", "-i", src_path, "-filter:a", f"atempo={speed}", "-vn", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg (speed up narration) failed: {result.stderr[-1500:]}")
    return _probe_duration(out_path)


def _loop_trim_mute_clip(src_path: str, out_path: str, target_duration: float) -> None:
    # -stream_loop -1 loops the source indefinitely; -t cuts it to the exact length we
    # need regardless of whether the source is longer or shorter than that -- avoids
    # having to reason about how many whole loops are "enough."
    vf = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1"
    cmd = [
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", src_path, "-t", str(target_duration),
        "-vf", vf, "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg (loop/trim background) failed: {result.stderr[-1500:]}")


def _run_satisfying_job(job_id: str, req: "SatisfyingRenderRequest") -> None:
    workdir = tempfile.mkdtemp(prefix="satisfying_")
    try:
        narration_path = os.path.join(workdir, "narration" + _guess_ext(req.narration_url))
        _download_image_with_retry(req.narration_url, narration_path)
        sped_path = os.path.join(workdir, "narration_sped.m4a")
        total_duration = _speed_up_audio(narration_path, sped_path, NARRATION_SPEED)

        # Cycle through the provided background pool in fixed-length muted chunks
        # until they cover the (sped-up) narration's full duration.
        n_segments = max(1, math.ceil(total_duration / BACKGROUND_SEGMENT_SECONDS))
        seg_paths = []
        elapsed = 0.0
        for i in range(n_segments):
            url = req.video_urls[i % len(req.video_urls)]
            src_path = os.path.join(workdir, f"bgsrc_{i}" + _guess_ext(url))
            _download_image_with_retry(url, src_path)
            seg_len = min(BACKGROUND_SEGMENT_SECONDS, total_duration - elapsed)
            seg_path = os.path.join(workdir, f"bgseg_{i}.mp4")
            _loop_trim_mute_clip(src_path, seg_path, seg_len)
            seg_paths.append(seg_path)
            elapsed += seg_len

        concat_list_path = os.path.join(workdir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for p in seg_paths:
                f.write(f"file '{p}'\n")
        combined_path = os.path.join(workdir, "combined.mp4")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", combined_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg (concat) failed: {result.stderr[-1500:]}")

        like_path = os.path.join(ASSETS_DIR, "like_greenscreen.mp4")
        subscribe_path = os.path.join(ASSETS_DIR, "subscribe_greenscreen.mp4")
        t_like_end = LIKE_DISPLAY_SECONDS
        t_sub_end = t_like_end + SUBSCRIBE_DISPLAY_SECONDS

        music_files = [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith(".mp3")]
        music_path = os.path.join(MUSIC_DIR, random.choice(music_files))
        fade_out_start = max(0.0, total_duration - MUSIC_FADE_SECONDS)

        filter_complex = (
            f"[1:v]scale={WIDTH}:{HEIGHT}[likescaled];"
            f"[likescaled]chromakey={LIKE_CHROMA_COLOR}:0.15:0.05[likekeyed];"
            f"[2:v]scale={WIDTH}:{HEIGHT}[subscaled];"
            f"[subscaled]chromakey={SUBSCRIBE_CHROMA_COLOR}:0.15:0.05[subkeyed];"
            f"[0:v][likekeyed]overlay=x=0:y=0:enable='between(t,0,{t_like_end:.2f})'[v1];"
            f"[v1][subkeyed]overlay=x=0:y=0:enable='between(t,{t_like_end:.2f},{t_sub_end:.2f})'[vout];"
            f"[4:a]atrim=0:{total_duration:.2f},asetpts=PTS-STARTPTS,volume={MUSIC_VOLUME},"
            f"afade=t=in:st=0:d={MUSIC_FADE_SECONDS},afade=t=out:st={fade_out_start:.2f}:d={MUSIC_FADE_SECONDS}[music];"
            f"[3:a][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )

        output_path = os.path.join(workdir, f"{uuid.uuid4().hex}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", combined_path,
            "-i", like_path,
            "-itsoffset", f"{t_like_end:.2f}", "-i", subscribe_path,
            "-i", sped_path,
            "-i", music_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "aac", "-b:a", "128k",
            "-t", str(total_duration),
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg (satisfying final) failed: {result.stderr[-2000:]}")

        video_url = _upload_to_supabase(output_path, folder="satisfatorios/", ext=".mp4", content_type="video/mp4")
        SATISFYING_JOBS[job_id] = {"status": "done", "result": {"video_url": video_url, "duration": total_duration}}
    except Exception as e:
        SATISFYING_JOBS[job_id] = {"status": "error", "error": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/satisfying/render")
def satisfying_render(req: SatisfyingRenderRequest, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    if not req.video_urls:
        raise HTTPException(status_code=422, detail="at least one video_url is required")

    job_id = uuid.uuid4().hex
    SATISFYING_JOBS[job_id] = {"status": "processing"}
    threading.Thread(target=_run_satisfying_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/satisfying/render-status/{job_id}")
def satisfying_render_status(job_id: str, x_api_key: str = Header(default="")):
    if RENDER_API_KEY and x_api_key != RENDER_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    job = SATISFYING_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
