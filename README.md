# NEAR Project Visualization

This repository contains a Python-based data analysis pipeline for the **NEAR Project**, designed to visualize eye-tracking metrics including gaze points, heatmaps, fixation trajectories, and pupil diameter. The system is optimized for **Google Colab** and integrates with **Google Drive** for batch processing of pilot data.

## Features

* **Heatmaps & Gaze Points:** Overlays gaze density and raw gaze coordinates onto world camera frames.
* **Fixation Trajectories:** Generates static chronological paths (PNG) and animated build-ups (GIF) of fixation points.
* **Pupil & Blink Analysis:** Visualizes 2D pupil diameter for both eyes (Eye 0 and Eye 1) to monitor physiological changes over time.
* **Automated Video Encoding:** Compiles windowed visualization frames into MP4 animations using `imageio` and `libx264`.
* **Batch Processing:** Automated loops to handle multiple subjects (e.g., AT, Ayu, JC, KC, LKH, SYH, YL) and their respective tasks.


## Project Structure

Google Drive Link for the outputs: https://drive.google.com/drive/folders/1Faj747rCdxj26TOu5yEzjPpAe-OYxDeq?usp=sharing
The pipeline organizes processed outputs into subdirectories within `1_Data_Analysis`, while reading raw data from subject-specific task folders:

```text
PilotData_V1_10232025/
├── 1_Data_Analysis/
│   ├── blink_pupil/           # Time-series plots of pupil diameter
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
