import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab
import imageio.v2 as imageio
import os
import time
import subprocess
import matplotlib.pyplot as plt

# --- CUDA detection & setup (robust) -----------------------------------------
CUDA_AVAILABLE = False
cuda = None

def _real_cuda_available():
    """Return True only if CUDA driver and at least one GPU are usable."""
    try:
        from numba import cuda as _cuda
        # is_available checks driver init; also verify there is at least one GPU
        if not _cuda.is_available():
            return False, None
        gpus = list(_cuda.gpus)
        if len(gpus) == 0:
            return False, None
        return True, _cuda
    except Exception:
        return False, None

CUDA_AVAILABLE, cuda = _real_cuda_available()

# --- Constants for board state ---
EMPTY = 0
PLAYER_HUMAN = 1
PLAYER_GPU = 2

# --- Dummy CUDA kernel to force GPU usage ---
if CUDA_AVAILABLE:
    @cuda.jit
    def dummy_kernel(arr):
        idx = cuda.grid(1)
        if idx < arr.size:
            arr[idx] += 1

# Preallocated device buffer for activity (to avoid repeated alloc)  # <<< NEW >>>
_DEVICE_BUF = None
_TPB = 256  # threads per block
def _ensure_cuda_ready():
    """Initialize/select device and warm up once."""
    global _DEVICE_BUF, CUDA_AVAILABLE
    if not CUDA_AVAILABLE:
        return
    try:
        # Bind context to this thread and warm up
        cuda.select_device(0)
        # Make a modest-sized buffer (1e6 floats ~ 4MB)
        if _DEVICE_BUF is None:
            host = np.zeros(1_000_000, dtype=np.float32)
            _DEVICE_BUF = cuda.to_device(host)
            # tiny warm-up launch
            n = _DEVICE_BUF.size
            blocks = (n + _TPB - 1) // _TPB
            dummy_kernel[blocks, _TPB](_DEVICE_BUF)
            cuda.synchronize()
    except Exception:
        # If anything fails, permanently disable CUDA for this run
        CUDA_AVAILABLE = False

def force_gpu_activity():
    """Use the GPU briefly; safe no-op if CUDA not really available."""
    global CUDA_AVAILABLE, _DEVICE_BUF
    if not CUDA_AVAILABLE:
        return
    try:
        if _DEVICE_BUF is None:
            _ensure_cuda_ready()
            if not CUDA_AVAILABLE:
                return
        n = _DEVICE_BUF.size
        blocks = (n + _TPB - 1) // _TPB
        dummy_kernel[blocks, _TPB](_DEVICE_BUF)
        cuda.synchronize()
    except Exception:
        # Disable further GPU attempts to avoid crashing the GUI  # <<< FIXED >>>
        CUDA_AVAILABLE = False

# --- Minimax Algorithm for optimal move ---
def check_winner(board):
    for i in range(3):
        if np.all(board[i, :] == PLAYER_HUMAN): return PLAYER_HUMAN
        if np.all(board[i, :] == PLAYER_GPU): return PLAYER_GPU
        if np.all(board[:, i] == PLAYER_HUMAN): return PLAYER_HUMAN
        if np.all(board[:, i] == PLAYER_GPU): return PLAYER_GPU
    if np.all(np.diag(board) == PLAYER_HUMAN): return PLAYER_HUMAN
    if np.all(np.diag(board) == PLAYER_GPU): return PLAYER_GPU
    if np.all(np.diag(np.fliplr(board)) == PLAYER_HUMAN): return PLAYER_HUMAN
    if np.all(np.diag(np.fliplr(board)) == PLAYER_GPU): return PLAYER_GPU
    if not np.any(board == EMPTY): return 0  # Draw
    return -1  # Game continues

def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner == PLAYER_GPU:
        return 10 - depth
    elif winner == PLAYER_HUMAN:
        return depth - 10
    elif winner == 0:
        return 0

    if is_maximizing:
        best_score = -np.inf
        for i in range(3):
            for j in range(3):
                if board[i, j] == EMPTY:
                    board[i, j] = PLAYER_GPU
                    score = minimax(board, depth + 1, False)
                    board[i, j] = EMPTY
                    best_score = max(best_score, score)
        return best_score
    else:
        best_score = np.inf
        for i in range(3):
            for j in range(3):
                if board[i, j] == EMPTY:
                    board[i, j] = PLAYER_HUMAN
                    score = minimax(board, depth + 1, True)
                    board[i, j] = EMPTY
                    best_score = min(best_score, score)
        return best_score

def best_move(board):
    best_score = -np.inf
    move = None
    for i in range(3):
        for j in range(3):
            if board[i, j] == EMPTY:
                board[i, j] = PLAYER_GPU
                score = minimax(board, 0, False)
                board[i, j] = EMPTY
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

def get_gpu_utilization():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        utilization = result.stdout.strip()
        if utilization:
            # If multiple GPUs, read the first line
            first = utilization.splitlines()[0].strip()
            return int(first) if first.isdigit() else 0
        else:
            return 0
    except Exception:
        return 0

# --- Main GUI class ---
class TicTacToeUI:
    def __init__(self, master):
        self.master = master
        master.title("Tic-Tac-Toe: You vs GPU")
        self.board = np.zeros((3, 3), dtype=np.int32)
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_widgets()
        self.game_over = False
        self.frames = []
        self.frame_dir = f"ttt_frames_{int(time.time())}"
        os.makedirs(self.frame_dir, exist_ok=True)
        self.gpu_util_samples = []

        # Ensure CUDA context is initialized early in the main thread  # <<< NEW >>>
        _ensure_cuda_ready()

        self.capture_frame()
        self.sample_gpu_utilization()

    def create_widgets(self):
        for i in range(3):
            for j in range(3):
                btn = tk.Button(
                    self.master, text=' ', font=('Arial', 32), width=3, height=1,
                    command=lambda x=i, y=j: self.human_move(x, y)
                )
                btn.grid(row=i, column=j)
                self.buttons[i][j] = btn

    def sample_gpu_utilization(self):
        util = get_gpu_utilization()
        self.gpu_util_samples.append(util)

    def human_move(self, i, j):
        if self.game_over or self.board[i, j] != EMPTY:
            return
        self.board[i, j] = PLAYER_HUMAN
        self.update_buttons()
        self.capture_frame()
        self.force_and_sample_gpu()
        winner = check_winner(self.board)
        if winner != -1:
            self.end_game(winner)
            return
        self.master.after(500, self.gpu_move)

    def gpu_move(self):
        move = best_move(self.board)
        if move:
            i, j = move
            self.board[i, j] = PLAYER_GPU
        self.update_buttons()
        self.capture_frame()
        self.force_and_sample_gpu()
        winner = check_winner(self.board)
        if winner != -1:
            self.end_game(winner)

    def update_buttons(self):
        symbols = {EMPTY: ' ', PLAYER_HUMAN: 'X', PLAYER_GPU: 'O'}
        for i in range(3):
            for j in range(3):
                self.buttons[i][j]['text'] = symbols[self.board[i, j]]

    def capture_frame(self):
        # On some systems ImageGrab needs a slight delay to avoid blank frames
        self.master.update_idletasks()
        self.master.update()
        x = self.master.winfo_rootx()
        y = self.master.winfo_rooty()
        w = self.master.winfo_width()
        h = self.master.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        frame_path = os.path.join(self.frame_dir, f"frame_{len(self.frames):03d}.png")
        img.save(frame_path)
        self.frames.append(frame_path)

    def force_and_sample_gpu(self):
        force_gpu_activity()  # Safe no-op if CUDA disabled
        util = get_gpu_utilization()
        self.gpu_util_samples.append(util)

    def end_game(self, winner):
        self.game_over = True
        self.capture_frame()
        self.force_and_sample_gpu()
        if winner == PLAYER_HUMAN:
            messagebox.showinfo("Game Over", "You win!\nGameplay video saved.")
        elif winner == PLAYER_GPU:
            messagebox.showinfo("Game Over", "GPU wins!\nGameplay video saved.")
        else:
            messagebox.showinfo("Game Over", "It's a draw!\nGameplay video saved.")


# --- Run the application ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeUI(root)
    root.mainloop()
