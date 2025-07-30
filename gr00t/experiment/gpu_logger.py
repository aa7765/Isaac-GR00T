import time
import torch
from torch.utils.tensorboard import SummaryWriter


class GpuSystemLogger:
    def __init__(self, log_dir="runs/system_metrics"):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step_start_time = None
        self.train_start_time = None
        self.train_end_time = None

    def start_training(self):
        """Call once at the beginning of training"""
        self.train_start_time = time.time()

    def end_training(self):
        """Call once at the end of training"""
        self.train_end_time = time.time()
        if self.train_start_time is not None:
            total_time = self.train_end_time - self.train_start_time
            self.writer.add_scalar("System/Total_Training_Time_sec", total_time, 0)
            print(f"✅ [GpuSystemLogger] Total training time: {total_time:.2f} seconds")

    def start_step(self):
        """Call at the beginning of each training step"""
        self.step_start_time = time.time()

    def end_step(self, step: int, batch_size: int = None):
        """Call at the end of each training step"""
        if self.step_start_time is None:
            return

        step_time = time.time() - self.step_start_time
        if batch_size:
            self.writer.add_scalar("System/Samples_per_sec", batch_size / step_time, step)
        self.writer.add_scalar("System/Step_Time_sec", step_time, step)

        # ✅ Live training time tracking
        if self.train_start_time is not None:
            elapsed = time.time() - self.train_start_time
            self.writer.add_scalar("System/Elapsed_Training_Time_sec", elapsed, step)

        # ✅ GPU stats (only if CUDA available)
        if torch.cuda.is_available():
            self.writer.add_scalar("GPU/Allocated_Memory_GB", torch.cuda.memory_allocated() / 1e9, step)
            self.writer.add_scalar("GPU/Reserved_Memory_GB", torch.cuda.memory_reserved() / 1e9, step)
            self.writer.add_scalar("GPU/Max_Memory_Allocated_GB", torch.cuda.max_memory_allocated() / 1e9, step)

    def close(self):
        """Call after training ends"""
        self.end_training()
        self.writer.close()
