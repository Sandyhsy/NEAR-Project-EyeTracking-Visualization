

# NEAR Project – Eye Tracking + LLM Interpretation

## Workflow Overview

```
Raw Eye Tracking Data
        ↓
Window Segmentation
        ↓
Frame Extraction
        ↓
Heatmap & AOI Generation
        ↓
LLM Attention Interpretation
        ↓
Structured Outputs
        ↓
Offline Interactive Demo
```

---

## Modules

### 1. [Eye-tracking Visualization](https://github.com/Sandyhsy/NEAR-Project-EyeTracking-Visualization/blob/main/Eye_Tracking_Data_Process.ipynb)

Generates heatmaps, gaze overlays, AOI crops, and MP4 animations.

This module implements the eye-tracking visualization pipeline for the NEAR Project, transforming raw Pupil Labs exports into interpretable visual representations of participant gaze behavior. It generates gaze point overlays, density-based heatmaps, fixation trajectories, AOI crops, and time-ordered MP4 animations. The outputs provide a structured visual foundation for downstream attention analysis and LLM-based interpretation.

#### Features

* **Heatmaps & Gaze Points:** Overlays gaze density and raw gaze coordinates onto world camera frames.
* **Fixation Trajectories:** Generates static chronological paths (PNG) and animated build-ups (GIF) of fixation points.
* **Pupil & Blink Analysis:** Visualizes 2D pupil diameter for both eyes (Eye 0 and Eye 1) to monitor physiological changes over time.
* **Automated Video Encoding:** Compiles windowed visualization frames into MP4 animations using `imageio` and `libx264`.
* **Batch Processing:** Automated loops to handle multiple subjects (e.g., AT, Ayu, JC, KC, LKH, SYH, YL) and their respective tasks.

#### Setup & Usage
1. Environment: Open the provided notebook in Google Colab.
2. Mount Drive: Ensure your Google Drive is mounted to access the PilotData_V1_10232025 directory.
3. Execution: Run the cells sequentially to process raw gaze data. The script will automatically iterate through the defined TASKS dictionary, generate visualizations, and save them back to your specified Drive output path.


---

### 2. [Attention-Aware Interpretation](https://github.com/Sandyhsy/NEAR-Project-EyeTracking-Visualization/blob/main/Attention_Aware_Output.ipynb)

Uses GPT-4o to generate task-aware summaries of visual attention.

This module implements a complete end-to-end attention analysis pipeline for the NEAR Project, combining gaze visualization and Vision-LLM interpretation. It segments raw eye-tracking recordings into fixed time windows, extracts representative frames, and generates gaze heatmaps and AOI crops from Pupil Labs exports. These structured visual outputs are then analyzed by GPT-4o to produce task-aware summaries (describe, compare, recall) that interpret participant visual attention patterns. The final results are saved in a structured format for downstream analysis and interactive demo visualization.

#### Configuration and API Requirements

* The analysis requires a valid OpenAI API Key with access to the GPT-4o model.
* The API key must be initialized in the environment using `os.environ["OPENAI_API_KEY"]`.
* The module uses the `openai` Python library to send image buffers and text prompts for processing.

#### Analysis Prompts

The following prompt is used to guide the model in interpreting the eye-tracking visualizations:

1. System Prompt:

        - Task 1: `State what the participant is looking at. Describe key visual features of the attended object.`
        - Task 2: `State what the participant is doing (e.g., comparing two objects, evaluating differences, inspecting features). Explain what specific objects or regions are being compared and the key visual similarities and differences that are likely being evaluated.`
        - Task 3: `Compare REFERENCE (Task1) and CURRENT image. Explain what is missing and how gaze suggests the participant is reasoning.`
        - Task 3_1: `Participant is reviewing the photo. Describe multiple important objects and their details.`
        - Task 3_2: `Compare the REFERENCE image and CURRENT image. Describe what is different.`

3. User Prompt:

        Analyze the attached image which shows a subject's gaze during an experiment.
        1. Identify the primary objects or areas the subject is focusing on.
        2. Describe the pattern of attention (e.g., concentrated focus, scanning, or distraction).
        3. Provide a brief summary of the likely intent behind this visual behavior.

---

### 3. [Offline Demo](https://github.com/Sandyhsy/NEAR-Project-EyeTracking-Visualization/blob/main/offline_demo.py)

Streamlit-based local visualization interface.

This component provides a local interactive visualization interface for the NEAR Project, enabling task selection, frame-level inspection, AOI visualization, heatmap playback, and LLM response display. The demo runs without API calls and operates on pre-generated outputs, making it suitable for presentation and offline evaluation.

#### Features

- Select task category (Describe / Compare / Recall)
- Manual frame selection
- Playback interval control (0–3 seconds)
- Manual frame selection
- Play sequential frames with adjustable interval
- View:
  - Original image
  - AOI overlay
  - LLM response text
  - Heatmap video

#### How to Run

```bash
pip install streamlit
streamlit run offline_demo.py
```
