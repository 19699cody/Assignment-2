# Setup and Run Instructions

## Requirements
- Python 3.10 or compatible version
- Gymnasium
- NumPy
- Matplotlib

## Installation
Install the required packages:
python -m pip install "gymnasium[toy-text]" numpy matplotlib
python -m pip install "gymnasium[toy-text]" "imageio[ffmpeg]"

## Run
Train and evaluate the Q-learning agent:
python main.py


Generate the Question F comparison plot (requires the three saved reward files):
python compare_f.py

To record a gameplay episode using the trained Q-table:
python record_video.py