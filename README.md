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
git clone https://github.com/yourusername/tic-tac-toe-gpu.git
cd tic-tac-toe-gpu
