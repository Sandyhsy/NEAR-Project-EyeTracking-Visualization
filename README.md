# NEAR Project Visualization

---
## [Eye-tracking Visualization](https://github.com/Sandyhsy/NEAR-Project-EyeTracking-Visualization/blob/main/Eye_Tracking_Data_Process.ipynb)
This part contains a Python-based data analysis pipeline for the **NEAR Project**, designed to visualize eye-tracking metrics including gaze points, heatmaps, fixation trajectories, and pupil diameter. The system is optimized for **Google Colab** and integrates with **Google Drive** for batch processing of pilot data.

### Features

* **Heatmaps & Gaze Points:** Overlays gaze density and raw gaze coordinates onto world camera frames.
* **Fixation Trajectories:** Generates static chronological paths (PNG) and animated build-ups (GIF) of fixation points.
* **Pupil & Blink Analysis:** Visualizes 2D pupil diameter for both eyes (Eye 0 and Eye 1) to monitor physiological changes over time.
* **Automated Video Encoding:** Compiles windowed visualization frames into MP4 animations using `imageio` and `libx264`.
* **Batch Processing:** Automated loops to handle multiple subjects (e.g., AT, Ayu, JC, KC, LKH, SYH, YL) and their respective tasks.

### Project Resources
Google Drive Link for the outputs: https://drive.google.com/drive/folders/1Faj747rCdxj26TOu5yEzjPpAe-OYxDeq?usp=sharing
<br>
Slides: https://docs.google.com/presentation/d/1O7bTlNXFIGug2euTmfhkF4tqT95M0Ipp1ZLi2jjcgYE/edit?usp=sharing

### Project Structure

The pipeline organizes processed outputs into subdirectories within `1_Data_Analysis`, while reading raw data from subject-specific task folders:

```text
PilotData_V1_10232025/
├── 1_Data_Analysis/
│   ├── blink_pupil/           # Time-series plots of pupil diameter
│   ├── fix/                   # Processed fixation data
│   ├── fix2/                  # Secondary fixation analysis
│   ├── fixation_Trajectory/   # Chronological gaze path visualizations
│   └── heatmap_gazepoint/     # Gaze density and coordinate visualizations
│       ├── animation/         # Compiled MP4 videos
│       ├── gaze_point/        # Raw coordinate PNG frames
│       ├── gaze_point_merge/  # Merged PNG frames
│       ├── heatmap/           # Density map PNG frames
│       └── heatmap_merge/  # Merged PNG frames
└── [Subject_Task]/            # Raw data source folders
    ├── world.mp4              # Scene camera video
    ├── world_timestamps.npy   # Sync timestamps
    └── exports/
        └── 000/
            ├── gaze_positions.csv
            ├── fixations.csv
            └── pupil_positions.csv
```
### Setup & Usage
1. Environment: Open the provided notebook in Google Colab.
2. Mount Drive: Ensure your Google Drive is mounted to access the PilotData_V1_10232025 directory.
3. Execution: Run the cells sequentially to process raw gaze data. The script will automatically iterate through the defined TASKS dictionary, generate visualizations, and save them back to your specified Drive output path.


---

## [Attention-Aware Analysis](https://github.com/Sandyhsy/NEAR-Project-EyeTracking-Visualization/blob/main/Attention_Aware_Output.ipynb)

This module integrates Large Language Models to interpret visual attention data by analyzing gaze-overlayed frames.

### Configuration and API Requirements

* The analysis requires a valid OpenAI API Key with access to the GPT-4o model.
* The API key must be initialized in the environment using `os.environ["OPENAI_API_KEY"]`.
* The module uses the `openai` Python library to send image buffers and text prompts for processing.

### Output Files

* **Analysis Summaries:** Descriptive text or CSV files containing the LLM's interpretation of subject focus.
* **Storage Location:** Results are saved within the `Attention-Aware_Output/` directory on Google Drive.

### Analysis Prompts

The following prompt is used to guide the model in interpreting the eye-tracking visualizations:

**System Prompt:**

1. Task 1:
    > "State what the participant is looking at.
    Describe key visual features of the attended object."

2. Task 2:
    > "State what the participant is doing (e.g., comparing).
    Explain similarities and differences being inspected."

3. Task 3:
    > "Compare REFERENCE (Task1) and CURRENT image.
    Explain what is missing and how gaze suggests the participant is reasoning."

4. Task 3_1:
    > "Participant is reviewing the photo.
    Describe multiple important objects and their details."

5. Task 3_2:
    > "Compare the REFERENCE image and CURRENT image.
    Describe what is different."

**User Prompt:**

> "Analyze the attached image which shows a subject's gaze during an experiment.
> 1. Identify the primary objects or areas the subject is focusing on.
> 2. Describe the pattern of attention (e.g., concentrated focus, scanning, or distraction).
> 3. Provide a brief summary of the likely intent behind this visual behavior."

### Visualization Pipeline

A flow representing the data movement from raw input to LLM-generated insights:

1. **Raw Data Input:** Load Pupil Labs exports (CSV) and world camera videos (MP4).
2. **Preprocessing:** Segment data into three-second windows and extract mid-point frames.
3. **Visualization Generation:** Generate Heatmap and Gaze Point overlays on extracted frames.
4. **LLM Analysis:** Submit visualization frames to GPT-4o via the OpenAI API.
5. **Final Output:** Save descriptive summaries and structured analysis back to Google Drive.

---
