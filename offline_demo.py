# Attention-Aware Offline Demo
# run with: streamlit run offline_demo.py

import json
import time
from pathlib import Path

import streamlit as st


# -----------------------------
# Utilities
# -----------------------------
def find_heatmap_video(task_dir: Path) -> Path | None:
    """Return heatmap video path if exists (case-insensitive)."""
    p = task_dir / "heatmap.mp4"
    if p.exists():
        return p
    mp4s = list(task_dir.glob("*.mp4"))
    return mp4s[0] if mp4s else None


def load_responses(task_dir: Path) -> dict:
    """Load responses.json safely."""
    resp_path = task_dir / "responses.json"
    if not resp_path.exists():
        return {}
    try:
        return json.loads(resp_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def id_to_frame_filename(resp_id: str) -> str:
    """Map response id like '001' -> '001.png'."""
    try:
        n = int(resp_id)
        return f"{n:03d}.png"
    except Exception:
        return resp_id


def get_image_path(task_dir: Path, subfolder: str, resp_id: str) -> Path:
    """Get image path under task_dir/subfolder/ mapped by resp_id."""
    filename = id_to_frame_filename(resp_id)
    return task_dir / subfolder / filename


def list_response_ids(responses: dict) -> list[str]:
    """Return sorted response ids, numeric-first if possible."""
    keys = list(responses.keys())

    def sort_key(k: str):
        try:
            return (0, int(k))
        except Exception:
            return (1, k)

    return sorted(keys, key=sort_key)


def task_title(task: str) -> str:
    return {"describe": "Describe", "compare": "Compare", "recall": "Recall"}.get(task, task.title())


def st_image_compat(img, **kwargs):
    """Compat wrapper for st.image across Streamlit versions."""
    try:
        return st.image(img, use_container_width=True, **kwargs)
    except TypeError:
        return st.image(img, use_column_width=True, **kwargs)


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Attention-Aware", layout="wide")
st.title("Attention-Aware")
st.caption("")

# White background + black text
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

DATA_ROOT = Path("demo_data")


# -----------------------------
# Session state init
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
    st.session_state.play_interval = 1.0


# -----------------------------
# Top controls
# -----------------------------
st.subheader("Controls")

# Load task assets
task_dir = DATA_ROOT / st.session_state.task
if not task_dir.exists():
    st.error(f"Task folder not found: {task_dir}")
    st.stop()

responses = load_responses(task_dir)
ids = list_response_ids(responses)
if not ids:
    st.warning("responses.json is empty or missing.")
    ids = ["001"]

# One-row layout:
# [Describe][Compare][Recall][Dropdown][◀][Slider][▶][Start][Stop]
c_tasks, c_id, c_slider, c_switch = st.columns(
    [3.0, 2.0, 2.0, 1.0],
    gap="small",
)

# Task buttons
with c_tasks:
    c_tasks.caption("Tasks")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Describe", use_container_width=True, key="btn_task_describe"):
            st.session_state.task = "describe"
            st.session_state.playing = False
            st.session_state.frame_idx = 0
    with b2:
        if st.button("Compare", use_container_width=True, key="btn_task_compare"):
            st.session_state.task = "compare"
            st.session_state.playing = False
            st.session_state.frame_idx = 0
    with b3:
        if st.button("Recall", use_container_width=True, key="btn_task_recall"):
            st.session_state.task = "recall"
            st.session_state.playing = False
            st.session_state.frame_idx = 0

# Reload after task changes (so dropdown options match the active task)
task_dir = DATA_ROOT / st.session_state.task
if not task_dir.exists():
    st.error(f"Task folder not found: {task_dir}")
    st.stop()

responses = load_responses(task_dir)
ids = list_response_ids(responses)
if not ids:
    ids = ["001"]

# Response ID dropdown
with c_id:
    c_id.caption("Response ID")
    selected = st.selectbox(
        "Response ID",
        options=ids,
        index=st.session_state.frame_idx if 0 <= st.session_state.frame_idx < len(ids) else 0,
        key="response_select",
        label_visibility="collapsed",
    )

    # When paused, dropdown controls the displayed frame
    if not st.session_state.playing:
        st.session_state.frame_idx = ids.index(selected)

# Interval arrows + slider
min_ivl = 0.0
max_ivl = 3.0
step_ivl = 0.1

with c_slider:
    st.caption("Playback interval (seconds)")
    c_dec, c_slider, c_inc = st.columns([0.45, 2.2, 0.45])
    
    with c_dec:
        if st.button("◀", use_container_width=True, key="interval_dec"):
            st.session_state.play_interval = max(min_ivl, round(st.session_state.play_interval - step_ivl, 1))

    with c_inc:
        if st.button("▶", use_container_width=True, key="interval_inc"):
            st.session_state.play_interval = min(max_ivl, round(st.session_state.play_interval + step_ivl, 1))

    with c_slider:
        new_interval = st.slider(
            "Playback interval (seconds)",
            min_value=min_ivl,
            max_value=max_ivl,
            value=float(st.session_state.play_interval),
            step=step_ivl,
            key="play_interval_slider",
            label_visibility="collapsed",
        )
        st.session_state.play_interval = float(new_interval)

# Start / Stop stacked vertically inside c_switch
with c_switch:
    col_start = st.container()
    col_stop = st.container()

    with col_start:
        if st.button("Start", use_container_width=True, key="btn_start"):
            st.session_state.playing = True
            st.session_state.frame_idx = 0
            st.rerun()

    with col_stop:
        if st.button("Stop", use_container_width=True, key="btn_stop"):
            st.session_state.playing = False
            st.rerun()

# -----------------------------
# Visual + Responses
# -----------------------------
st.divider()
st.subheader(f"{task_title(st.session_state.task)} Visual")

# Clamp frame index
st.session_state.frame_idx = max(0, min(st.session_state.frame_idx, len(ids) - 1))
current_id = ids[st.session_state.frame_idx]

# Row: Original | AOI | Responses
col_img, col_aoi, col_resp = st.columns([1.15, 0.65, 1.15])

with col_img:
    st.caption("Original image")
    img_path = get_image_path(task_dir, "Original_image", current_id)
    if img_path.exists():
        st_image_compat(str(img_path))
    else:
        st.warning(f"Missing: {img_path}")

with col_aoi:
    st.caption("AOI")

    if st.session_state.task == "recall":
        ref_path = task_dir / "aois" / f"{current_id}_ref.png"
        cur_path = task_dir / "aois" / f"{current_id}_current.png"
        single_path = get_image_path(task_dir, "aois", current_id)

        if ref_path.exists() and cur_path.exists():
            st.markdown("**Reference**")
            st_image_compat(str(ref_path))
            st.markdown("**Current**")
            st_image_compat(str(cur_path))
        elif cur_path.exists():
            st_image_compat(str(cur_path))
        elif ref_path.exists():
            st_image_compat(str(ref_path))
        elif single_path.exists():
            st_image_compat(str(single_path))
        else:
            st.warning(f"Missing AOI for ID: {current_id}")

    else:
        aoi_path = get_image_path(task_dir, "aois", current_id)
        if aoi_path.exists():
            st_image_compat(str(aoi_path))
        else:
            st.warning(f"Missing: {aoi_path}")

with col_resp:
    st.caption("Responses")
    resp_text = responses.get(current_id, "")
    st.markdown(f"**ID {current_id}**")
    st.write(resp_text if resp_text else "(No response text)")


# -----------------------------
# Heatmap video
# -----------------------------
st.divider()
st.subheader("Heatmap video")

video_path = find_heatmap_video(task_dir)
if video_path:
    video_col, _ = st.columns([1.25, 2])
    with video_col:
        st.video(str(video_path))
else:
    st.info("No heatmap video found in this task folder.")


# -----------------------------
# Playback logic (MUST be after rendering)
# -----------------------------
# Streamlit updates the UI only after the script finishes a run.
# Therefore, we render the current frame first, then sleep, then advance, then rerun.
if st.session_state.playing:
    time.sleep(float(st.session_state.play_interval))

    if st.session_state.frame_idx < len(ids) - 1:
        st.session_state.frame_idx += 1
        st.rerun()
    else:
        st.session_state.playing = False