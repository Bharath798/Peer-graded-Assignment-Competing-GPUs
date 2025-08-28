# Peer-graded-Assignment-Competing-GPUs
 This project implements a Tic-Tac-Toe game with a GPU twist. You play against an AI opponent powered by the minimax algorithm, while the program actively forces GPU activity (via CUDA kernels) and tracks real GPU utilization in real time. 

Features

✅ Classic Tic-Tac-Toe game with a Tkinter GUI

✅ Minimax algorithm ensures the GPU opponent plays optimally

✅ CUDA GPU activity triggered during moves (via Numba)

✅ GPU utilization monitoring (using nvidia-smi)

✅ Automatic frame capture of the GUI (with Pillow)

✅ Option to save the game session as a video (.mp4)

✅ Robust fallback: if CUDA is unavailable, the game still works in CPU-only mode


Installation

Clone the repository
```
git clone https://github.com/yourusername/tic-tac-toe-gpu.git

cd tic-tac-toe-gpu
```

Set up Python environment

```
pip install numpy

pip install tk

pip install pillow

pip install imageio

pip install matplotlib

pip install numba
```

Usage

Run the game:

```
python CUDA_Competing_GPUs.py
```

Gameplay

1) You are X

2) The GPU opponent is O

3) Click on the grid to make your move

4) The GPU opponent will respond optimally


Technical Highlights

1) CUDA Integration

* Uses Numba @cuda.jit kernels for GPU activity

* Warm-up kernel to stabilize CUDA context

2) AI Opponent

* Implements minimax algorithm with depth-based scoring

* Guarantees optimal play (impossible to beat if you play randomly)

3) GUI & Multimedia

* Tkinter for game board

* PIL.ImageGrab for frame capturing

* ImageIO for video compilation
