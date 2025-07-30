from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from gr00t.experiment.gpu_logger import GpuSystemLogger


class GpuLoggingCallback(TrainerCallback):
    def __init__(self, log_dir, batch_size):
        self.logger = GpuSystemLogger(log_dir)
        self.batch_size = batch_size

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.logger.start_training()  # ✅ Start tracking training time

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.logger.start_step()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.logger.end_step(step=state.global_step, batch_size=self.batch_size)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.logger.close()  # ✅ Logs total training time and closes writer
