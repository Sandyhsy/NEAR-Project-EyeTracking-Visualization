# Attention-Aware Online Demo
import base64
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# Configuration
# -----------------------------
load_dotenv()
DATA_ROOT = Path("demo_data")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_URL = "https://api.openai.com/v1/responses"

REQUEST_TIMEOUT_SEC = 45
MAX_WORKERS = 2
DEFAULT_RESPONSE_DELAY = 0.5

# -----------------------------
# Utilities
# -----------------------------
def get_api_key() -> str:
    """Read API key from environment each time."""
    load_dotenv(override=True)
    return os.getenv("OPENAI_API_KEY", "").strip()


def is_api_ready() -> bool:
    """Return True if API key is available."""
    return bool(get_api_key())


def clear_all_caches_on_start() -> None:
    """Clear memory cache and disk cache once at app startup."""
    if st.session_state.get("_cache_cleared_once", False):
        return

    # Clear memory cache
    st.session_state.online_cache = {}
    st.session_state.pending_jobs = set()
    st.session_state.recent_results = {}
    st.session_state.display_started_at = {}
    st.session_state.last_rendered_id = None

    # Clear loaded-cache flags for all tasks
    for task_name in ["describe", "compare", "recall"]:
        loaded_key = f"_loaded_cache::{task_name}"
        if loaded_key in st.session_state:
            del st.session_state[loaded_key]

    # Clear disk cache files
    for task_name in ["describe", "compare", "recall"]:
        task_dir = DATA_ROOT / task_name
        cache_path = get_cache_file(task_dir)
        if cache_path.exists():
            try:
                cache_path.unlink()
            except Exception:
                pass

    st.session_state["_cache_cleared_once"] = True
    

def find_heatmap_video(task_dir: Path) -> Optional[Path]:
    """Return heatmap video path if exists."""
    p = task_dir / "heatmap.mp4"
    if p.exists():
        return p
    mp4s = list(task_dir.glob("*.mp4"))
    return mp4s[0] if mp4s else None


def task_title(task: str) -> str:
    return {"describe": "Describe", "compare": "Compare", "recall": "Recall"}.get(task, task.title())


def st_image_compat(img, **kwargs):
    """Compatibility wrapper for st.image across Streamlit versions."""
    try:
        return st.image(img, width='stretch', **kwargs)
    except TypeError:
        return st.image(img, use_column_width=True, **kwargs)


def id_to_frame_filename(resp_id: str) -> str:
    """Map response id like '001' to '001.png'."""
    try:
        n = int(resp_id)
        return f"{n:03d}.png"
    except Exception:
        return f"{resp_id}.png" if "." not in resp_id else resp_id


def sort_response_ids(ids: list[str]) -> list[str]:
    """Sort frame ids numerically when possible."""
    def sort_key(k: str):
        try:
            return (0, int(k))
        except Exception:
            return (1, k)
    return sorted(ids, key=sort_key)


def list_frame_ids(task_dir: Path) -> list[str]:
    """
    Build response ids from local image files.
    Supports:
    - Original_image/001.png
    - aois/001.png
    - aois/001_ref.png + aois/001_current.png
    """
    found = set()

    original_dir = task_dir / "Original_image"
    aoi_dir = task_dir / "aois"

    if original_dir.exists():
        for p in original_dir.iterdir():
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                found.add(p.stem)

    if aoi_dir.exists():
        for p in aoi_dir.iterdir():
            if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
                continue

            stem = p.stem
            if stem.endswith("_ref"):
                found.add(stem[:-4])
            elif stem.endswith("_current"):
                found.add(stem[:-8])
            else:
                found.add(stem)

    return sort_response_ids(list(found))


def get_image_path(task_dir: Path, subfolder: str, resp_id: str) -> Path:
    """Return subfolder image path using frame id."""
    return task_dir / subfolder / id_to_frame_filename(resp_id)


def load_prompt(task_dir: Path) -> str:
    """Load prompt.txt from task folder."""
    p = task_dir / "prompt.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()

    return (
        "You are analyzing a user's visual attention in an experiment.\n\n"
        "Return:\n"
        "Line 1: A short heading (<=10 words).\n"
        "Line 2+: A focused paragraph (3-6 sentences).\n"
        "No bullet points.\n"
        "Be concise and emphasize key differences or important objects."
    )


def get_cache_file(task_dir: Path) -> Path:
    """Return the per-task disk cache file."""
    return task_dir / "llm_cache.json"


def load_disk_cache(task_dir: Path) -> dict:
    """Load per-task disk cache."""
    cache_path = get_cache_file(task_dir)
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_disk_cache(task_dir: Path, cache_data: dict) -> None:
    """Save per-task disk cache."""
    cache_path = get_cache_file(task_dir)
    try:
        cache_path.write_text(
            json.dumps(cache_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def image_to_data_url(image_path: Path) -> str:
    """Convert a local image to a data URL."""
    suffix = image_path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"

    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def get_llm_input_images(task_dir: Path, task_name: str, resp_id: str) -> list[Path]:
    """
    Build image input payload for the model.
    For recall, prefer reference + current.
    """
    aoi_dir = task_dir / "aois"

    if task_name == "recall":
        ref_candidates = [
            aoi_dir / f"{resp_id}_ref.png",
            aoi_dir / f"{resp_id}_ref.jpg",
            aoi_dir / f"{resp_id}_ref.jpeg",
            aoi_dir / f"{resp_id}_ref.webp",
        ]
        cur_candidates = [
            aoi_dir / f"{resp_id}_current.png",
            aoi_dir / f"{resp_id}_current.jpg",
            aoi_dir / f"{resp_id}_current.jpeg",
            aoi_dir / f"{resp_id}_current.webp",
        ]
        single_candidates = [
            aoi_dir / f"{resp_id}.png",
            aoi_dir / f"{resp_id}.jpg",
            aoi_dir / f"{resp_id}.jpeg",
            aoi_dir / f"{resp_id}.webp",
        ]

        image_paths = []

        for p in ref_candidates:
            if p.exists():
                image_paths.append(p)
                break

        for p in cur_candidates:
            if p.exists():
                image_paths.append(p)
                break

        if not image_paths:
            for p in single_candidates:
                if p.exists():
                    image_paths.append(p)
                    break

        return image_paths

    candidates = [
        aoi_dir / f"{resp_id}.png",
        aoi_dir / f"{resp_id}.jpg",
        aoi_dir / f"{resp_id}.jpeg",
        aoi_dir / f"{resp_id}.webp",
    ]

    for p in candidates:
        if p.exists():
            return [p]

    return []


def extract_output_text(response_json: dict) -> str:
    """Extract text from Responses API payload."""
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts = []
    output_items = response_json.get("output", [])

    if isinstance(output_items, list):
        for item in output_items:
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for c in content:
                if c.get("type") == "output_text" and c.get("text"):
                    parts.append(c["text"])

    joined = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    if joined:
        return joined

    return "(No response text returned.)"


def call_vision_llm(image_paths: list[Path], prompt: str) -> str:
    """
    Call the online vision model using the Responses API.
    """
    api_key = get_api_key()

    if not api_key:
        return "[LLM error] OPENAI_API_KEY is not set."

    if not image_paths:
        return "[LLM error] No AOI image found for this frame."

    content = [{"type": "input_text", "text": prompt}]

    for image_path in image_paths:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(image_path),
            }
        )

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            OPENAI_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        data = response.json()
        return extract_output_text(data)

    except requests.Timeout:
        return "[LLM error] Request timed out."
    except requests.HTTPError:
        try:
            detail = response.json()
            return f"[LLM error] HTTP {response.status_code}: {json.dumps(detail, ensure_ascii=False)}"
        except Exception as e:
            return f"[LLM error] HTTP error: {e}"
    except Exception as e:
        return f"[LLM error] {e}"

# -----------------------------
# Background job model
# -----------------------------
@dataclass(frozen=True)
class Job:
    task_name: str
    resp_id: str
    task_dir_str: str


# -----------------------------
# Session state initialization
# -----------------------------
if "task" not in st.session_state:
    st.session_state.task = "describe"
if "playing" not in st.session_state:
    st.session_state.playing = False
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "play_interval" not in st.session_state:
    st.session_state.play_interval = 2.0
if "last_rendered_id" not in st.session_state:
    st.session_state.last_rendered_id = None
if "last_task" not in st.session_state:
    st.session_state.last_task = st.session_state.task
if "response_delay_s" not in st.session_state:
    st.session_state.response_delay_s = DEFAULT_RESPONSE_DELAY

if "online_cache" not in st.session_state:
    st.session_state.online_cache = {}
if "pending_jobs" not in st.session_state:
    st.session_state.pending_jobs = set()
if "job_q" not in st.session_state:
    st.session_state.job_q = queue.Queue()
if "result_q" not in st.session_state:
    st.session_state.result_q = queue.Queue()
if "workers_started" not in st.session_state:
    st.session_state.workers_started = False
if "recent_results" not in st.session_state:
    st.session_state.recent_results = {}
if "display_started_at" not in st.session_state:
    st.session_state.display_started_at = {}

clear_all_caches_on_start()

# -----------------------------
# Cache helpers
# -----------------------------
def get_cache_key(task_name: str, resp_id: str) -> str:
    return f"{task_name}::{resp_id}"


def ensure_task_cache_loaded(task_name: str, task_dir: Path) -> None:
    """Load disk cache into memory once per task."""
    task_key = f"_loaded_cache::{task_name}"
    if st.session_state.get(task_key):
        return

    disk_cache = load_disk_cache(task_dir)
    for resp_id, text in disk_cache.items():
        st.session_state.online_cache[get_cache_key(task_name, resp_id)] = text

    st.session_state[task_key] = True


def get_cached_response(task_name: str, resp_id: str) -> Optional[str]:
    return st.session_state.online_cache.get(get_cache_key(task_name, resp_id))


def set_cached_response(task_name: str, task_dir: Path, resp_id: str, text: str) -> None:
    st.session_state.online_cache[get_cache_key(task_name, resp_id)] = text

    disk_cache = load_disk_cache(task_dir)
    disk_cache[resp_id] = text
    save_disk_cache(task_dir, disk_cache)


# -----------------------------
# Worker pool
# -----------------------------
def worker_loop(job_q: queue.Queue, result_q: queue.Queue) -> None:
    """Background worker loop for API calls."""
    while True:
        job = job_q.get()
        if job is None:
            job_q.task_done()
            break

        try:
            task_dir = Path(job.task_dir_str)
            prompt = load_prompt(task_dir)
            image_paths = get_llm_input_images(task_dir, job.task_name, job.resp_id)
            text = call_vision_llm(image_paths, prompt)
            result_q.put((job.task_name, job.task_dir_str, job.resp_id, text))
        except Exception as e:
            result_q.put((job.task_name, job.task_dir_str, job.resp_id, f"[LLM error] {e}"))
        finally:
            job_q.task_done()


def ensure_workers() -> None:
    """Start background workers once."""
    if st.session_state.workers_started:
        return

    for _ in range(MAX_WORKERS):
        t = threading.Thread(
            target=worker_loop,
            args=(st.session_state.job_q, st.session_state.result_q),
            daemon=True,
        )
        t.start()

    st.session_state.workers_started = True


def drain_results() -> None:
    """Move finished worker results into session cache."""
    drained_any = False

    while True:
        try:
            task_name, task_dir_str, resp_id, text = st.session_state.result_q.get_nowait()
        except queue.Empty:
            break

        task_dir = Path(task_dir_str)
        set_cached_response(task_name, task_dir, resp_id, text)

        pending_key = get_cache_key(task_name, resp_id)
        if pending_key in st.session_state.pending_jobs:
            st.session_state.pending_jobs.remove(pending_key)

        st.session_state.recent_results[pending_key] = text
        drained_any = True

    if drained_any:
        pass


def maybe_enqueue(task_name: str, task_dir: Path, resp_id: str) -> None:
    """Queue API request if not already cached and not pending."""
    cache_key = get_cache_key(task_name, resp_id)

    if get_cached_response(task_name, resp_id) is not None:
        return

    if cache_key in st.session_state.pending_jobs:
        return

    st.session_state.pending_jobs.add(cache_key)
    st.session_state.job_q.put(Job(task_name=task_name, resp_id=resp_id, task_dir_str=str(task_dir)))


# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Attention-Aware", layout="wide")
st.title("Attention-Aware")

st.markdown(
    """
    <style>
      html, body, [class*="css"]  { background-color: white !important; color: black !important; }
      .block-container { padding-top: 2.0rem; padding-bottom: 1rem; padding-left: 1.2rem; padding-right: 1.2rem; }
      div[data-testid="stVerticalBlock"] { gap: 0.6rem; }
      .stButton>button { padding: 0.35rem 0.8rem; background-color: #f7f7f7; color: black; }
      /* Smaller arrow buttons */
      button[kind="secondary"] { padding: 0.25rem 0.45rem; font-size: 16px; height: 38px; }
      /* Reduce top spacing for the slider container */
      div[data-baseweb="slider"] { margin-top: -6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

api_ready = is_api_ready()

if api_ready:
    st.success("API key ready")
    ensure_workers()
    drain_results()
else:
    st.warning("Preparing API key...")

task_dir = DATA_ROOT / st.session_state.task
all_response_ids = list_frame_ids(task_dir)

if not all_response_ids:
    st.error(f"No valid frames found in {task_dir}")
    st.stop()

if st.session_state.selected_id not in all_response_ids:
    st.session_state.selected_id = all_response_ids[0]

if st.session_state.frame_idx >= len(all_response_ids):
    st.session_state.frame_idx = 0

ensure_task_cache_loaded(st.session_state.task, task_dir)


# -----------------------------
# Callbacks
# -----------------------------
def set_task(task_name: str) -> None:
    st.session_state.task = task_name
    st.session_state.playing = False
    st.session_state.frame_idx = 0
    st.session_state.last_rendered_id = None


def start_play() -> None:
    st.session_state.playing = True


def stop_play() -> None:
    st.session_state.playing = False


# -----------------------------
# Controls
# -----------------------------
st.markdown("### Controls")

control_cols = st.columns([1, 1, 1, 2, 2, 1])

with control_cols[0]:
    if st.button("Describe", width='stretch'):
        set_task("describe")

with control_cols[1]:
    if st.button("Compare", width='stretch'):
        set_task("compare")

with control_cols[2]:
    if st.button("Recall", width='stretch'):
        set_task("recall")

with control_cols[3]:
    if st.session_state.task != st.session_state.last_task:
        task_dir = DATA_ROOT / st.session_state.task
        all_response_ids = list_frame_ids(task_dir)
        st.session_state.selected_id = all_response_ids[0]
        st.session_state.last_task = st.session_state.task
        ensure_task_cache_loaded(st.session_state.task, task_dir)

    selected = st.selectbox(
        "Response ID",
        options=all_response_ids,
        index=all_response_ids.index(st.session_state.selected_id) if st.session_state.selected_id in all_response_ids else 0,
        key="response_id_selectbox",
    )
    st.session_state.selected_id = selected

with control_cols[4]:
    play_interval = st.slider(
        "Playback interval (seconds)",
        min_value=0.5,
        max_value=5.0,
        value=float(st.session_state.play_interval),
        step=0.1,
    )
    st.session_state.play_interval = play_interval

with control_cols[5]:
    st.button("Start", on_click=start_play, width="stretch")
    st.button("Stop", on_click=stop_play, width="stretch")


# -----------------------------
# Sync frame index with dropdown
# -----------------------------
current_id = st.session_state.selected_id
if current_id in all_response_ids:
    st.session_state.frame_idx = all_response_ids.index(current_id)
else:
    st.session_state.frame_idx = 0
    current_id = all_response_ids[0]
    st.session_state.selected_id = current_id


# -----------------------------
# Task header
# -----------------------------
st.markdown(f"### {task_title(st.session_state.task)} Visual")


# -----------------------------
# Load images for display
# -----------------------------
original_img_path = get_image_path(task_dir, "Original_image", current_id)

if st.session_state.task == "recall":
    aoi_ref_path = task_dir / "aois" / f"{current_id}_ref.png"
    aoi_current_path = task_dir / "aois" / f"{current_id}_current.png"

    if not aoi_ref_path.exists():
        for ext in [".jpg", ".jpeg", ".webp"]:
            alt = task_dir / "aois" / f"{current_id}_ref{ext}"
            if alt.exists():
                aoi_ref_path = alt
                break

    if not aoi_current_path.exists():
        for ext in [".jpg", ".jpeg", ".webp"]:
            alt = task_dir / "aois" / f"{current_id}_current{ext}"
            if alt.exists():
                aoi_current_path = alt
                break

    if not aoi_ref_path.exists() and not aoi_current_path.exists():
        aoi_img_path = get_image_path(task_dir, "aois", current_id)
    else:
        aoi_img_path = None
else:
    aoi_img_path = get_image_path(task_dir, "aois", current_id)


# -----------------------------
# Online response state
# -----------------------------
cache_key = get_cache_key(st.session_state.task, current_id)
cached_text = get_cached_response(st.session_state.task, current_id)

if api_ready and cached_text is None:
    maybe_enqueue(st.session_state.task, task_dir, current_id)

if st.session_state.last_rendered_id != cache_key:
    st.session_state.display_started_at[cache_key] = time.time()
    st.session_state.last_rendered_id = cache_key

pending = cache_key in st.session_state.pending_jobs
cached_text = get_cached_response(st.session_state.task, current_id)

if cached_text is not None:
    show_text = cached_text
else:
    elapsed = time.time() - st.session_state.display_started_at.get(cache_key, time.time())
    if elapsed < st.session_state.response_delay_s or pending:
        show_text = "Generating response..."
    else:
        show_text = "Generating response..."


# -----------------------------
# Main display layout
# -----------------------------
col1, col2, col3 = st.columns([1.15, 0.65, 1.15])

with col1:
    st.caption("Original image")
    if original_img_path.exists():
        st_image_compat(str(original_img_path))
    else:
        st.info(f"Missing original image: {original_img_path.name}")
        
with col2:
    st.caption("AOI")

    if st.session_state.task == "recall":

        st.caption("Reference Frame")
        if aoi_ref_path.exists():
            st.image(str(aoi_ref_path), width='stretch')
        else:
            st.info("Missing reference image")

        st.caption("Current View")
        if aoi_current_path.exists():
            st.image(str(aoi_current_path), width='stretch')
        else:
            st.info("Missing current image")
    else:
        if aoi_img_path.exists():
            st_image_compat(str(aoi_img_path))
        else:
            st.info(f"Missing AOI image: {aoi_img_path.name}")
            
with col3:
    st.caption("Response")
    st.text_area(
        label="response_box",
        value=show_text,
        height=320,
        label_visibility="collapsed",
        key=f"response_box_{st.session_state.task}_{current_id}_{'ready' if cached_text else 'pending'}",
    )


# -----------------------------
# Heatmap video
# -----------------------------
st.markdown("---")
st.caption("Heatmap video")

heatmap_video = find_heatmap_video(task_dir)
if heatmap_video and heatmap_video.exists():
    st.video(str(heatmap_video))
else:
    st.info("No heatmap video found.")


# -----------------------------
# Playback
# -----------------------------
if st.session_state.playing:
    time.sleep(st.session_state.play_interval)

    next_idx = st.session_state.frame_idx + 1
    if next_idx >= len(all_response_ids):
        next_idx = 0

    st.session_state.frame_idx = next_idx
    st.session_state.selected_id = all_response_ids[next_idx]
    st.rerun()

elif pending:
    time.sleep(0.2)
    st.rerun()