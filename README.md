# NEAR Project Visualization

This repository contains a Python-based data analysis pipeline for the **NEAR Project**, designed to visualize eye-tracking metrics including gaze points, heatmaps, fixation trajectories, and pupil diameter. The system is optimized for **Google Colab** and integrates with **Google Drive** for batch processing of pilot data.

## Features

* **Heatmaps & Gaze Points:** Overlays gaze density and raw gaze coordinates onto world camera frames.
* **Fixation Trajectories:** Generates static chronological paths (PNG) and animated build-ups (GIF) of fixation points.
* **Pupil & Blink Analysis:** Visualizes 2D pupil diameter for both eyes (Eye 0 and Eye 1) to monitor physiological changes over time.
* **Automated Video Encoding:** Compiles windowed visualization frames into MP4 animations using `imageio` and `libx264`.
* **Batch Processing:** Automated loops to handle multiple subjects (e.g., AT, Ayu, JC, KC, LKH, SYH, YL) and their respective tasks.

## Project Resources
Google Drive Link for the outputs: https://drive.google.com/drive/folders/1Faj747rCdxj26TOu5yEzjPpAe-OYxDeq?usp=sharing
<br>
Slides: https://docs.google.com/presentation/d/1O7bTlNXFIGug2euTmfhkF4tqT95M0Ipp1ZLi2jjcgYE/edit?usp=sharing

## Project Structure

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
## Setup & Usage
1. Environment: Open the provided notebook in Google Colab.
2. Mount Drive: Ensure your Google Drive is mounted to access the PilotData_V1_10232025 directory.
3. Execution: Run the cells sequentially to process raw gaze data. The script will automatically iterate through the defined TASKS dictionary, generate visualizations, and save them back to your specified Drive output path.
