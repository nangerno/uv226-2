from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
from typing import Callable, Optional, Dict, Any
import time
import shutil
import json
import stat
from transformers.trainer_utils import is_main_process
import wandb
import torch
from state_manager import get_state, set_state

# Constants
MAX_TRIES = 9

# Checkpoint selection thresholds
LOSS_WORSE_THRESHOLD = 1.15  # 15% worse than best
LOSS_SIGNIFICANTLY_WORSE = 1.2  # 20% worse - early stop
OVERFITTING_RATIO_THRESHOLD = 2.0  # Eval loss 2x train loss
OVERFITTING_PENALTY_FACTOR = 0.3
SKIP_FULL_EVAL_THRESHOLD = 1.1

# Checkpoint interpolation
INTERPOLATION_WEIGHT_BEST = 0.6
INTERPOLATION_WEIGHT_SECOND = 0.4
INTERPOLATION_SIMILARITY_THRESHOLD = 1.05  # Within 5%

# Time estimation constants
EVAL_SAVE_TIME_BUFFER_MINUTES = 12
TIME_VARIANCE_MINUTES = 3

# Evaluation frequency thresholds
EVAL_FREQUENT_STEPS_TIGHT = 30  # Very frequent in tight time mode
EVAL_FREQUENT_STEPS_NORMAL = 50  # Frequent near end of training
EVAL_MODERATE_STEPS = 100  # Moderately frequent

# Progress thresholds for adaptive evaluation
PROGRESS_THRESHOLD_TIGHT_70 = 0.7  # Last 30% of training
PROGRESS_THRESHOLD_TIGHT_50 = 0.5  # Last 50% of training
PROGRESS_THRESHOLD_FREQUENT = 0.8  # Last 20% of training
PROGRESS_THRESHOLD_MODERATE = 0.6  # Last 40% of training

# Time-aware mode thresholds (minutes)
TIME_TIGHT_THRESHOLD = 30
TIME_AMPLE_THRESHOLD = 120


MIS_MATCH_VOCAB_SIZE_MODELS = [
    'NousResearch/Nous-Capybara-7B-V1',
    'berkeley-nest/Starling-LM-7B-alpha',
    'NousResearch/Hermes-2-Theta-Llama-3-8B',
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4'
]

ERROR_GENERATION_CONFIG_MODELS = [
    "lmsys/vicuna-7b-v1.5", 
    "lmsys/vicuna-13b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b", 
    "defog/llama-3-sqlcoder-8b"
]

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

print(f"LOCAL_RANK: {LOCAL_RANK} in customized_trainer.py", flush=True)


def safe_remove_directory(directory: str) -> None:
    """
    Safely remove directory with error handling for readonly files.
    
    Args:
        directory: Path to directory to remove
    """
    if not os.path.exists(directory):
        return
    
    try:
        # First try: simple removal with ignore_errors
        shutil.rmtree(directory, ignore_errors=True)
    except Exception as e:
        # Second try: handle readonly files by changing permissions
        print(f"Warning: Error removing directory (ignoring): {e}", flush=True)
        try:
            def handle_remove_readonly(func, path, exc):
                """Handle readonly files by changing permissions before removal."""
                if os.path.exists(path):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
            
            shutil.rmtree(directory, onerror=handle_remove_readonly)
        except Exception:
            # If all else fails, just continue - the copytree will handle it
            pass


class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        max_steps: int = -1,
        checking_step: int = 100,
        total_steps_all_epochs: int = -1,
        end_time: str = "",
        checking_mode: str = "none"
    ):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        self.checking_step = checking_step
        self.total_steps_all_epochs = total_steps_all_epochs
        self.checking_mode = checking_mode
        self.end_time = end_time
        self._capture_eval_loss_at_checking = False
        self._checking_step_eval_loss = None
        # DECISIVE: Track top 2 checkpoints for interpolation
        self.top_checkpoints = []  # List of (step, eval_loss, train_loss, generalization_score)
        self.max_top_checkpoints = 2
        # FRESH: Smart checkpoint pruning - track predicted quality
        self.checkpoint_predictions = {}  # step -> predicted_loss
        self.last_eval_loss = None
        self.eval_loss_trend = []  # Track trend for prediction
        # FRESH: Incremental evaluation - track subset evaluations
        self.subset_eval_results = {}  # step -> subset_eval_loss
        self.skip_full_eval_threshold = SKIP_FULL_EVAL_THRESHOLD
        
    def compute_loss(self, state: TrainerState, metrics: Dict[str, Any]) -> Optional[float]:
        """Extract eval_loss from metrics."""
        return metrics.get("eval_loss", None)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Custom logic to decide whether to save or evaluate
        # print(f"************* on_step_end: {state.global_step}, check eval", flush=True)
        # TODO: implement the logic to save the model without evaluating if there is no check points --> avoid evaluating takes too much time
        # Check if the checking_step is reached
        # print(f"Checking the model at step: {state.global_step}, checking_step: {self.checking_step}, checking_mode: {self.checking_mode}", flush=True)
        if state.global_step == self.checking_step and self.checking_mode == "first_time":
            # print(f"Checking the model at step: {state.global_step}", flush=True)
            # check the time so far to estimate the training time in total 
            my_state = get_state()
            if "train" not in my_state or "start_time" not in my_state.get("train", {}):
                print(f"Warning: Missing 'train' or 'start_time' in state, skipping time estimation", flush=True)
                return control
            start_time_obj = datetime.datetime.strptime(my_state["train"]["start_time"], "%Y-%m-%d %H:%M:%S")
            # start_train_time may not exist in all cases, use start_time as fallback
            start_train_time_str = my_state["train"].get("start_train_time", my_state["train"]["start_time"])
            start_train_time_obj = datetime.datetime.strptime(start_train_time_str, "%Y-%m-%d %H:%M:%S")
            
            log_content = f"Checking the model at step: {state.global_step}"
            now = datetime.datetime.now()
            preparation_time = (start_train_time_obj - start_time_obj).total_seconds()
            log_content += f"\nPreparation time: {preparation_time}"
            time_so_far = (now - start_time_obj).total_seconds()
            log_content += f"\nTime so far: {time_so_far}"
            time_for_one_step = (now - start_train_time_obj).total_seconds() / self.checking_step
            log_content += f"\nTime for one step: {time_for_one_step}"
            
            # Improved time estimation: account for potential slowdown as training progresses
            # Early steps are often faster (warmup, cache effects), so we apply a slowdown factor
            # This is a conservative estimate to avoid over-committing
            progress_ratio = state.global_step / max(self.total_steps_all_epochs, 1)
            # Apply 1.1x slowdown factor for remaining steps (10% slower on average)
            # This accounts for: longer sequences, more complex gradients, memory fragmentation
            slowdown_factor = 1.1 if progress_ratio > 0.1 else 1.0  # Only apply after 10% progress
            
            # Now estimate the total training time for this training
            log_content += f"\nTotal steps all epochs: {self.total_steps_all_epochs}"
            remaining_steps = self.total_steps_all_epochs - state.global_step
            total_remaining_training_time = time_for_one_step * remaining_steps * slowdown_factor
            if slowdown_factor > 1.0:
                log_content += f"\nApplied slowdown factor: {slowdown_factor:.2f}x (conservative estimate)"
            log_content += f"\nTotal remaining training time: {total_remaining_training_time}"
            # n * time_so_far + total_remaining_training_time = total_remaining_time
            end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
            total_remaining_time = (end_time_obj - now).total_seconds()
            log_content += f"\nTotal remaining time: {total_remaining_time}"
            
            # Improved time estimation accounting for evaluation overhead
            # Each run requires: training time + evaluation time + checkpoint saving time
            # Estimate evaluation time: typically 2-5 minutes per evaluation
            # With adaptive evaluation frequency, estimate ~10-20 evaluations per run
            # Store for use in evaluation skipping logic
            self.avg_eval_time_minutes = 3  # Conservative estimate
            evaluations_per_run = 15  # Estimate based on adaptive frequency
            eval_overhead_per_run = self.avg_eval_time_minutes * evaluations_per_run * 60  # in seconds
            
            # Estimate checkpoint saving time (including potential interpolation)
            checkpoint_save_time = 5 * 60  # 5 minutes base, could be more with interpolation
            
            # Total overhead per run: eval + save + variance
            max_var_time_sofar = TIME_VARIANCE_MINUTES * 60
            overhead_per_run = eval_overhead_per_run + checkpoint_save_time + max_var_time_sofar
            
            # Calculate how many additional runs we can fit
            # Formula: n * (time_per_run + overhead) + current_run_remaining_time <= total_remaining_time
            time_per_run = time_so_far + max_var_time_sofar
            total_time_per_run = time_per_run + overhead_per_run
            
            # Reserve time for final checkpoint saving and evaluation
            final_save_buffer = max(EVAL_SAVE_TIME_BUFFER_MINUTES * 60, checkpoint_save_time * 2)  # At least 2x for safety
            
            available_time = total_remaining_time - total_remaining_training_time - final_save_buffer
            n = available_time / total_time_per_run if total_time_per_run > 0 else 0
            n = int(n)
            
            log_content += f"\nTime estimation details:"
            log_content += f"\n  Evaluation overhead per run: {eval_overhead_per_run/60:.1f} min ({evaluations_per_run} evals × {self.avg_eval_time_minutes} min)"
            log_content += f"\n  Checkpoint save time: {checkpoint_save_time/60:.1f} min"
            log_content += f"\n  Total overhead per run: {overhead_per_run/60:.1f} min"
            log_content += f"\n  Final save buffer: {final_save_buffer/60:.1f} min"
            my_state["check_details"] = {
                "now": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                "start_time": str(start_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "start_train_time": str(start_train_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "checking_step": self.checking_step,
                "checking_mode": self.checking_mode,
                "estimation_of_steps": n,
                "preparation_time": preparation_time,
                "time_so_far": time_so_far,
                "time_for_one_step": time_for_one_step,
                "total_remaining_training_time": total_remaining_training_time,
                "total_remaining_time": total_remaining_time,
                "end_time": self.end_time,
            }
            if n > 0: # we should try more 
                log_content += f"\nEstimated number of steps to complete the training: {n}"
                # Don't save at first_time checking to save time and space (match top miner)
                control.should_save = False
                control.should_training_stop = True
                args.save_strategy = "no"
                # save the current loss of this step to the state;
                last_log = state.log_history[-1]
                my_state["train"]["current_loss"] = last_log["loss"]
                my_state["mode"] = "continue"
                if n > MAX_TRIES:
                    n = MAX_TRIES
                log_content += f"\nFinal number: {n + 1}"
                my_state["next_runs"] = n + 1 # including the current run
            else:
                print(f"Time is not enough so we will finish the training", flush=True)
                my_state["mode"] = "finish"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)            
            return control
    
        elif state.global_step == self.checking_step and self.checking_mode == "second_time": # at second time, we don't estimate the training time again, just save the current_loss
            log_content = f"Checking the model at step: {state.global_step} where check_mode=second_time"            
            my_state = get_state()
            if "train" not in my_state:
                my_state["train"] = {}
            current_loss = state.log_history[-1]["loss"]
            my_state["train"]["current_loss"] = current_loss
            
            # CRITICAL: Trigger evaluation to get eval_loss for better checkpoint selection
            # This is the key improvement - use eval_loss instead of train_loss for selection
            print(f"Triggering evaluation at checking_step {state.global_step} to get eval_loss for multi-run selection", flush=True)
            control.should_evaluate = True
            control.should_save = True  # Need to save to get eval metrics
            # Store flag to capture eval_loss after evaluation
            self._capture_eval_loss_at_checking = True
            self._checking_step_eval_loss = None
                
            # Stop training temporarily to wait for evaluation to complete
            # We'll decide in on_evaluate whether to continue or stop based on eval_loss
            control.should_training_stop = True
            
            # OPTIMIZATION: If this is clearly the best run so far and we have limited time,
            # consider committing early to save time for final training
            if "runs" in my_state and len(my_state["runs"]) > 0:
                # Check if current run is significantly better than previous runs
                previous_best_loss = min([run.get("current_eval_loss", run["current_loss"]) for run in my_state["runs"]])
                # If current loss is >10% better, this is a strong signal
                if current_loss < previous_best_loss * 0.9:
                    remaining_runs = my_state.get("next_runs", 1) - len(my_state["runs"]) - 1
                    if remaining_runs > 2:  # If we have many runs left
                        print(f"OPTIMIZATION: Current run loss {current_loss:.6f} is >10% better than best {previous_best_loss:.6f}. Consider early commitment after eval.", flush=True)
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)
        
            
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            # FRESH: Smart checkpoint pruning - predict quality before evaluating
            # If predicted to be poor, skip evaluation to save time
            if state.global_step in self.checkpoint_predictions:
                predicted_loss = self.checkpoint_predictions[state.global_step]
                best_loss = self.best_checkpoint_info["loss"] if self.best_checkpoint_info else float('inf')
                if predicted_loss > best_loss * LOSS_WORSE_THRESHOLD:
                    avg_eval_time = getattr(self, 'avg_eval_time_minutes', 3)
                    print(f"FRESH: Skipping evaluation at step {state.global_step} - predicted loss {predicted_loss:.6f} is poor (best: {best_loss:.6f}), saving ~{avg_eval_time} min", flush=True)
                    control.should_evaluate = False
                    control.should_save = False  # Don't save poor checkpoints
                    return control
            
            # OPTIMIZATION: Skip evaluation if loss trend is clearly improving
            # If we've seen consistent improvement, we can skip some evaluations
            if len(self.eval_loss_trend) >= 3:
                recent_trends = self.eval_loss_trend[-3:]
                if all(t < 0 for t in recent_trends):  # All recent trends are negative (improving)
                    avg_improvement = sum(recent_trends) / len(recent_trends)
                    # If improving at >0.5% per evaluation, skip this one (will catch up later)
                    if abs(avg_improvement) > self.last_eval_loss * 0.005 if self.last_eval_loss else False:
                        # Only skip if we're not near the end (don't risk missing best checkpoint)
                        if self.max_steps == -1 or state.global_step < self.max_steps * 0.9:
                            print(f"OPTIMIZATION: Skipping evaluation at step {state.global_step} - loss improving consistently ({avg_improvement:.6f}/eval), will evaluate next time", flush=True)
                            control.should_evaluate = False
                            return control
            
            # do not allow the pod to be stopped by any reason 
                # first check if there is at least one checkpoint or not 
            print(f"Evaluating the model at step: {state.global_step} the reason: {when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                if not self.has_checkpoint: # if there is no checkpoint, we just save the model, do not evaluate
                    print(f"No checkpoint found, just save the model at step: {state.global_step}", flush=True)
                    control.should_evaluate = False
                    self.save_only = True
        return control

    def _handle_subset_evaluation(self, state: TrainerState, eval_loss: float, control: TrainerControl) -> Optional[TrainerControl]:
        """
        Handle subset evaluation - skip full evaluation if subset loss is poor.
        
        Returns:
            TrainerControl if should skip full evaluation, None otherwise
        """
        best_loss = self.best_checkpoint_info["loss"] if self.best_checkpoint_info else float('inf')
        if eval_loss > best_loss * self.skip_full_eval_threshold:
            print(f"FRESH: Skipping full evaluation - subset loss {eval_loss:.6f} > best {best_loss:.6f} * {self.skip_full_eval_threshold}", flush=True)
            return control
        else:
            print(f"FRESH: Subset loss {eval_loss:.6f} is promising, proceeding with full evaluation", flush=True)
            return None

    def _track_and_predict_loss_trend(self, eval_loss: float, state: TrainerState) -> None:
        """
        Track eval loss trend and predict next checkpoint quality.
        
        Args:
            eval_loss: Current evaluation loss
            state: Trainer state
        """
        if self.last_eval_loss is not None:
            trend = eval_loss - self.last_eval_loss
            self.eval_loss_trend.append(trend)
            # Keep only last 5 trends
            if len(self.eval_loss_trend) > 5:
                self.eval_loss_trend = self.eval_loss_trend[-5:]
            
            # Predict next checkpoint quality based on trend
            if len(self.eval_loss_trend) >= 2:
                avg_trend = sum(self.eval_loss_trend[-3:]) / min(3, len(self.eval_loss_trend))
                predicted_next_loss = eval_loss + avg_trend
                self.checkpoint_predictions[state.global_step + 100] = predicted_next_loss
                
                # If predicted loss is very poor, suggest early stopping
                best_loss = self.best_checkpoint_info["loss"] if self.best_checkpoint_info else eval_loss
                if predicted_next_loss > best_loss * LOSS_WORSE_THRESHOLD:
                    print(f"FRESH: Predicted next checkpoint will be poor (predicted={predicted_next_loss:.6f} vs best={best_loss:.6f}), consider early stopping", flush=True)
        
        self.last_eval_loss = eval_loss

    def _handle_checking_step_evaluation(self, state: TrainerState, eval_loss: float, control: TrainerControl) -> TrainerControl:
        """
        Handle evaluation at checking_step for multi-run selection.
        
        Args:
            state: Trainer state
            eval_loss: Current evaluation loss
            control: Trainer control
            
        Returns:
            Modified TrainerControl
        """
        my_state = get_state()
        if "train" not in my_state:
            my_state["train"] = {}
        my_state["train"]["current_eval_loss"] = eval_loss
        print(f"CRITICAL: Stored eval_loss {eval_loss:.6f} at checking_step {state.global_step} for multi-run selection", flush=True)
        
        # Now make decision based on eval_loss instead of train_loss
        current_is_the_best = False
        control.should_training_stop = True
        
        if "runs" in my_state and len(my_state["runs"]) > 0:
            # Use eval_loss for comparison (CRITICAL IMPROVEMENT)
            if all("current_eval_loss" in run for run in my_state["runs"]):
                current_min_eval_loss = min([run.get("current_eval_loss", run["current_loss"]) for run in my_state["runs"]])
                comparison_loss = eval_loss
                print(f"Comparing eval_loss: {eval_loss:.6f} vs best eval_loss: {current_min_eval_loss:.6f}", flush=True)
            else:
                # Fallback: compare with train_loss if eval_loss not available in previous runs
                current_min_loss = min([run["current_loss"] for run in my_state["runs"]])
                comparison_loss = eval_loss  # Still use eval_loss for this run
                current_min_eval_loss = current_min_loss
                print(f"Comparing eval_loss: {eval_loss:.6f} vs best train_loss: {current_min_eval_loss:.6f} (fallback)", flush=True)
            
            # Improved comparison: use small epsilon for floating point comparison
            # and consider it "equal" if within 0.1% (accounts for numerical precision)
            epsilon = current_min_eval_loss * 0.001
            if eval_loss <= current_min_eval_loss + epsilon:
                # If this is the last run and loss is best/equal, continue to completion
                if len(my_state["runs"]) + 1 == my_state["next_runs"]:
                    if eval_loss < current_min_eval_loss - epsilon:
                        print(f"CRITICAL: Eval loss {eval_loss:.6f} is better than best {current_min_eval_loss:.6f}, continuing training", flush=True)
                    else:
                        print(f"CRITICAL: Eval loss {eval_loss:.6f} is equal to best {current_min_eval_loss:.6f} (within 0.1%), continuing training", flush=True)
                    current_is_the_best = True
                else:
                    # OPTIMIZATION: Early commitment strategy
                    # If this run is significantly better (>5%) and we have many runs left,
                    # consider committing early to save time for final training
                    relative_improvement = (current_min_eval_loss - eval_loss) / current_min_eval_loss if current_min_eval_loss > 0 else 0
                    remaining_runs = my_state.get("next_runs", 1) - len(my_state["runs"]) - 1
                    
                    if relative_improvement > 0.05 and remaining_runs > 2:
                        print(f"OPTIMIZATION: Eval loss {eval_loss:.6f} is {relative_improvement*100:.1f}% better than best {current_min_eval_loss:.6f}", flush=True)
                        print(f"  Consider early commitment: {remaining_runs} runs remaining. Current strategy: continue exploration.", flush=True)
                    
                    print(f"Eval loss {eval_loss:.6f} is best/equal, but not last run. Will continue with next LR.", flush=True)
            else:
                # Adaptive early termination with improved logic
                # Calculate relative difference
                relative_diff = (eval_loss - current_min_eval_loss) / current_min_eval_loss if current_min_eval_loss > 0 else float('inf')
                
                if relative_diff > (LOSS_SIGNIFICANTLY_WORSE - 1.0):
                    # Significantly worse: stop immediately
                    print(f"ADAPTIVE TERMINATION: Eval loss {eval_loss:.6f} is {relative_diff*100:.1f}% worse than best {current_min_eval_loss:.6f}, stopping early", flush=True)
                    control.should_training_stop = True
                elif relative_diff > (LOSS_WORSE_THRESHOLD - 1.0):
                    # Moderately worse: still stop but log the difference
                    print(f"Eval loss {eval_loss:.6f} is {relative_diff*100:.1f}% worse than best {current_min_eval_loss:.6f}, stopping this run", flush=True)
                    control.should_training_stop = True
                else:
                    # Slightly worse: still stop but note it's close
                    print(f"Eval loss {eval_loss:.6f} is {relative_diff*100:.1f}% worse than best {current_min_eval_loss:.6f}, stopping this run (close but not best)", flush=True)
                    control.should_training_stop = True
        else:
            # First run in the series
            if my_state.get("next_runs", 1) == 1:
                current_is_the_best = True
                print(f"Only one run, continuing training", flush=True)
        
        if current_is_the_best:
            control.should_training_stop = False
            my_state["mode"] = "finish"
        else:
            # Don't save if not best to save time and space
            control.should_save = False
        
        if is_main_process(LOCAL_RANK):
            set_state(my_state)
        self._capture_eval_loss_at_checking = False
        self._checking_step_eval_loss = eval_loss
        
        return control

    def _compute_generalization_score(self, eval_loss: float, train_loss: Optional[float]) -> tuple[float, float]:
        """
        Compute generalization score with overfitting penalty.
        
        Args:
            eval_loss: Evaluation loss
            train_loss: Training loss (optional)
            
        Returns:
            Tuple of (generalization_score, overfitting_penalty)
        """
        generalization_score = eval_loss
        overfitting_penalty = 0.0
        
        if train_loss is not None and train_loss > 0:
            overfitting_gap = max(0, eval_loss - train_loss)
            overfitting_penalty = OVERFITTING_PENALTY_FACTOR * overfitting_gap
            generalization_score = eval_loss - overfitting_penalty
            print(f"Step {self.best_checkpoint_info['step'] if self.best_checkpoint_info else 'unknown'}: eval_loss={eval_loss:.6f}, train_loss={train_loss:.6f}, overfitting_gap={overfitting_gap:.6f}, generalization_score={generalization_score:.6f}", flush=True)
        else:
            print(f"Step {self.best_checkpoint_info['step'] if self.best_checkpoint_info else 'unknown'}: eval_loss={eval_loss:.6f}, train_loss=None, generalization_score={generalization_score:.6f}", flush=True)
        
        return generalization_score, overfitting_penalty

    def _should_update_best_checkpoint(self, generalization_score: float, eval_loss: float, state: TrainerState) -> bool:
        """
        Determine if best checkpoint should be updated.
        Uses generalization_score as primary metric, with tie-breaking logic.
        
        Args:
            generalization_score: Computed generalization score
            eval_loss: Evaluation loss
            state: Trainer state for additional context
            
        Returns:
            True if should update, False otherwise
        """
        if self.best_checkpoint_info is None:
            return True
        
        current_best_score = self.best_checkpoint_info.get("generalization_score", self.best_checkpoint_info["loss"])
        current_best_eval_loss = self.best_checkpoint_info["loss"]
        
        # Primary comparison: generalization_score
        if generalization_score < current_best_score:
            return True
        elif generalization_score > current_best_score:
            return False
        else:
            # Tie-breaking: if generalization scores are equal, prefer:
            # 1. Lower eval_loss (better raw performance)
            # 2. Earlier step (less risk of overfitting, more training time remaining)
            if eval_loss < current_best_eval_loss:
                return True
            elif eval_loss > current_best_eval_loss:
                return False
            else:
                # If both scores and losses are equal, prefer earlier checkpoint
                # (gives more time for further training)
                return state.global_step < self.best_checkpoint_info["step"]

    def _should_use_interpolation(self) -> bool:
        """
        Determine if checkpoint interpolation should be used.
        Only interpolate if checkpoints are similar in quality (within threshold).
        
        Returns:
            True if should interpolate, False otherwise
        """
        if len(self.top_checkpoints) < 2:
            return False
        
        best_score = self.top_checkpoints[0]["generalization_score"]
        second_score = self.top_checkpoints[1]["generalization_score"]
        
        # Only interpolate if second best is within threshold of best
        # This ensures we're not mixing good and bad checkpoints
        return second_score < best_score * INTERPOLATION_SIMILARITY_THRESHOLD
    
    def _compute_adaptive_interpolation_weights(self, ckpt1: dict, ckpt2: dict) -> tuple[float, float]:
        """
        Compute adaptive interpolation weights based on performance gap.
        If checkpoints are very similar, use more balanced weights.
        If one is clearly better, give it more weight.
        
        Args:
            ckpt1: First checkpoint info (best)
            ckpt2: Second checkpoint info (second best)
            
        Returns:
            Tuple of (weight1, weight2) for interpolation
        """
        score1 = ckpt1["generalization_score"]
        score2 = ckpt2["generalization_score"]
        
        # Calculate performance gap ratio
        gap_ratio = score2 / score1 if score1 > 0 else 1.0
        
        # If scores are very similar (gap < 2%), use balanced weights (0.55, 0.45)
        # If gap is larger, use more weight on best checkpoint
        if gap_ratio < 1.02:  # Within 2%
            weight1, weight2 = 0.55, 0.45
        elif gap_ratio < 1.05:  # Within 5% (similarity threshold)
            # Linear interpolation between balanced and default weights
            # Closer scores -> more balanced, further scores -> more weight on best
            balance_factor = (1.05 - gap_ratio) / 0.03  # 0 to 1 as gap goes from 5% to 2%
            weight1 = 0.55 + balance_factor * 0.05  # 0.55 to 0.6
            weight2 = 1.0 - weight1
        else:
            # Use default weights for larger gaps
            weight1, weight2 = INTERPOLATION_WEIGHT_BEST, INTERPOLATION_WEIGHT_SECOND
        
        return weight1, weight2

    def _load_checkpoint_state_dicts(self, ckpt1_path: str, ckpt2_path: str, filename: str) -> Optional[tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], bool]]:
        """
        Load state dicts from two checkpoints.
        
        Args:
            ckpt1_path: Path to first checkpoint directory
            ckpt2_path: Path to second checkpoint directory
            filename: Name of the model file to load
            
        Returns:
            Tuple of (state1, state2, has_safetensors) or None if loading fails
        """
        file1_path = os.path.join(ckpt1_path, filename)
        file2_path = os.path.join(ckpt2_path, filename)
        
        if not os.path.exists(file2_path):
            return None
        
        try:
            from safetensors.torch import load_file
            has_safetensors = True
        except ImportError:
            has_safetensors = False
        
        try:
            if filename.endswith(".safetensors") and has_safetensors:
                state1 = load_file(file1_path)
                state2 = load_file(file2_path)
            else:
                state1 = torch.load(file1_path, map_location="cpu")
                state2 = torch.load(file2_path, map_location="cpu")
            
            return state1, state2, has_safetensors
        except Exception as e:
            print(f"  Warning: Failed to load state dicts from {filename}: {e}", flush=True)
            return None

    def _interpolate_state_dicts(self, state1: Dict[str, torch.Tensor], state2: Dict[str, torch.Tensor], weight1: float, weight2: float) -> Dict[str, torch.Tensor]:
        """
        Interpolate two state dicts with given weights.
        
        Args:
            state1: First state dict
            state2: Second state dict
            weight1: Weight for first state dict
            weight2: Weight for second state dict
            
        Returns:
            Interpolated state dict
        """
        interpolated = {}
        for key in state1.keys():
            if key in state2 and state1[key].shape == state2[key].shape:
                interpolated[key] = weight1 * state1[key] + weight2 * state2[key]
            else:
                interpolated[key] = state1[key]  # Use ckpt1 if key missing in ckpt2
        return interpolated

    def _interpolate_checkpoints(self, ckpt1: dict, ckpt2: dict) -> str:
        """
        Interpolate two checkpoints and save to submission directory.
        Uses adaptive weights based on performance gap.
        
        Args:
            ckpt1: First checkpoint info dict (best)
            ckpt2: Second checkpoint info dict (second best)
            
        Returns:
            Interpolation info string
        """
        print(f"DECISIVE: Interpolating top 2 checkpoints for better generalization", flush=True)
        print(f"  Checkpoint 1: step={ckpt1['step']}, gen_score={ckpt1['generalization_score']:.6f}, eval_loss={ckpt1['eval_loss']:.6f}", flush=True)
        print(f"  Checkpoint 2: step={ckpt2['step']}, gen_score={ckpt2['generalization_score']:.6f}, eval_loss={ckpt2['eval_loss']:.6f}", flush=True)
        
        # Use adaptive weights based on performance gap
        weight1, weight2 = self._compute_adaptive_interpolation_weights(ckpt1, ckpt2)
        print(f"  Using adaptive weights: {weight1:.3f} * ckpt1 + {weight2:.3f} * ckpt2 (based on performance gap)", flush=True)
        ckpt1_path = os.path.join(self.output_dir, f"checkpoint-{ckpt1['step']}")
        ckpt2_path = os.path.join(self.output_dir, f"checkpoint-{ckpt2['step']}")
        
        # Check if both checkpoints exist
        if not (os.path.exists(ckpt1_path) and os.path.exists(ckpt2_path)):
            print(f"  Warning: Checkpoints not found, using best checkpoint only", flush=True)
            self._copy_best_checkpoint()
            return "single_best"
        
        # Find model files (could be .bin or .safetensors)
        import glob
        ckpt1_files = glob.glob(os.path.join(ckpt1_path, "*.safetensors"))
        if not ckpt1_files:
            ckpt1_files = glob.glob(os.path.join(ckpt1_path, "*.bin"))
        
        if not ckpt1_files:
            print(f"  Warning: Could not find model files, using best checkpoint only", flush=True)
            self._copy_best_checkpoint()
            return "single_best"
        
        # Copy first checkpoint as base
        shutil.copytree(ckpt1_path, self.submission_dir)
        
        # Interpolate weights (already computed adaptively above)
        
        try:
            from safetensors.torch import save_file
            has_safetensors_save = True
        except ImportError:
            has_safetensors_save = False
        
        for file_path in ckpt1_files:
            filename = os.path.basename(file_path)
            output_path = os.path.join(self.submission_dir, filename)
            
            result = self._load_checkpoint_state_dicts(ckpt1_path, ckpt2_path, filename)
            if result is None:
                continue
            
            state1, state2, has_safetensors = result
            interpolated = self._interpolate_state_dicts(state1, state2, weight1, weight2)
            
            # Save interpolated weights
            try:
                if filename.endswith(".safetensors") and has_safetensors_save:
                    save_file(interpolated, output_path)
                else:
                    torch.save(interpolated, output_path)
            except Exception as e:
                print(f"  Warning: Failed to save interpolated weights for {filename}: {e}", flush=True)
        
        print(f"  Successfully interpolated checkpoints", flush=True)
        return f"interpolated:step1={ckpt1['step']},step2={ckpt2['step']},w1={weight1},w2={weight2}"

    def _copy_best_checkpoint(self) -> None:
        """Copy the best checkpoint to submission directory."""
        best_checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{self.best_checkpoint_info['step']}")
        shutil.copytree(best_checkpoint_path, self.submission_dir)

    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        self.save_only = False
        # Use eval_loss (which is cross-entropy) for checkpoint selection
        # This directly optimizes for test_loss which is also cross-entropy
        eval_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return control 
        
        # FRESH: Incremental evaluation - check if this is a subset evaluation
        is_subset_eval = metrics.get("_is_subset_eval", False)
        if is_subset_eval:
            subset_loss = eval_loss
            self.subset_eval_results[state.global_step] = subset_loss
            print(f"FRESH: Subset evaluation at step {state.global_step}: loss={subset_loss:.6f}", flush=True)
            
            # Skip full evaluation if subset loss is poor
            skip_result = self._handle_subset_evaluation(state, subset_loss, control)
            if skip_result is not None:
                return skip_result
        
        print(f"GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}", flush=True)
        
        # FRESH: Smart checkpoint pruning - predict quality early
        if eval_loss is not None:
            self._track_and_predict_loss_trend(eval_loss, state)
        
        # CRITICAL: Store eval_loss at checking_step for multi-run selection
        if hasattr(self, '_capture_eval_loss_at_checking') and self._capture_eval_loss_at_checking:
            if state.global_step == self.checking_step and self.checking_mode == "second_time":
                if eval_loss is not None:
                    return self._handle_checking_step_evaluation(state, eval_loss, control)
        
        # Monitor train/eval gap for overfitting detection
        if state.log_history:
            last_train_loss = state.log_history[-1].get("loss", None)
            if last_train_loss and eval_loss:
                gap_ratio = eval_loss / last_train_loss if last_train_loss > 0 else float('inf')
                if gap_ratio > OVERFITTING_RATIO_THRESHOLD:
                    print(f"WARNING: Overfitting detected! Train loss: {last_train_loss:.6f}, Eval loss: {eval_loss:.6f}, Ratio: {gap_ratio:.2f}", flush=True)
        
        # DECISIVE: Overfitting-aware checkpoint selection
        # Track train_loss to compute generalization_score
        train_loss = None
        if state.log_history:
            last_log = state.log_history[-1]
            train_loss = last_log.get("loss", None)
        
        # Cross-entropy loss optimization: track best cross-entropy loss
        # eval_loss is already cross-entropy, so we use it directly
        if eval_loss is not None:
            # DECISIVE: Compute generalization_score to penalize overfitting
            generalization_score, overfitting_penalty = self._compute_generalization_score(eval_loss, train_loss)
            
            # Update best checkpoint using generalization_score (not just eval_loss)
            should_update = self._should_update_best_checkpoint(generalization_score)
            
            if should_update:
                print(f"DECISIVE: Updating best checkpoint at step {state.global_step} with generalization_score: {generalization_score:.6f} (eval_loss: {eval_loss:.6f})", flush=True)
                self.best_checkpoint_info = {
                    "loss": eval_loss,
                    "step": state.global_step,
                    "train_loss": train_loss,
                    "generalization_score": generalization_score,
                    "overfitting_penalty": overfitting_penalty
                }
                self.update_best_checkpoint = True
            
            # DECISIVE: Track top 2 checkpoints for interpolation
            checkpoint_entry = {
                "step": state.global_step,
                "eval_loss": eval_loss,
                "train_loss": train_loss,
                "generalization_score": generalization_score
            }
            
            # Add to top checkpoints list
            self.top_checkpoints.append(checkpoint_entry)
            
            # Keep only top 2 by generalization_score
            self.top_checkpoints.sort(key=lambda x: x["generalization_score"])
            if len(self.top_checkpoints) > self.max_top_checkpoints:
                self.top_checkpoints = self.top_checkpoints[:self.max_top_checkpoints]
            
            top_ckpt_info = [(c['step'], f"gen_score={c['generalization_score']:.6f}") for c in self.top_checkpoints]
            print(f"Top {len(self.top_checkpoints)} checkpoints: {top_ckpt_info}", flush=True)
            
            if not should_update and self.best_checkpoint_info is not None:
                current_best_score = self.best_checkpoint_info.get("generalization_score", self.best_checkpoint_info["loss"])
                print(f" At step: {state.global_step} The generalization_score: {generalization_score:.6f} is not better than current best: {current_best_score:.6f}, update_best_checkpoint={self.update_best_checkpoint}", flush=True)
            

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        
        if state.global_step == self.max_steps and self.max_steps != -1:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            control.should_training_stop = True
        
        self.has_checkpoint = True
        
        if not is_main_process(LOCAL_RANK): # if not main process, skip this
            return 
            
        if self.save_only: # if only save, do not evaluate 
            print(f"Only save the model at step: {state.global_step}, no evaluation", flush=True)
            current_step = state.global_step
            # Remove existing directory if it exists
            safe_remove_directory(self.submission_dir)
                
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{current_step}"),
                self.submission_dir
            )
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{current_step},no_eval")
            
            # release the flag
            self.save_only = False
            return 
            
        # Custom logic after model is saved
        # You can trigger external services, logs, or backups here
        if (
            self.update_best_checkpoint
            and is_main_process(LOCAL_RANK)
        ):
            print(f"Copy the best checkpoint to the submission directory at step: {state.global_step}", flush=True)
            # Remove existing directory if it exists
            safe_remove_directory(self.submission_dir)
            
            best_eval_loss = self.best_checkpoint_info["loss"]
            best_gen_score = self.best_checkpoint_info.get("generalization_score", best_eval_loss)
            
            # OPTIMIZATION: Check remaining time before interpolation
            # Interpolation takes extra time, so only do it if we have enough time
            should_interpolate = False
            if self._should_use_interpolation():
                # Check if we have enough time for interpolation (takes ~2-3x longer than simple copy)
                if self.end_time:
                    try:
                        end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
                        end_time_obj = end_time_obj.replace(tzinfo=timezone.utc)
                        now = datetime.datetime.now(timezone.utc)
                        remaining_minutes = (end_time_obj - now).total_seconds() / 60
                        # Only interpolate if we have at least 15 minutes remaining
                        if remaining_minutes >= 15:
                            should_interpolate = True
                        else:
                            print(f"  Skipping interpolation - only {remaining_minutes:.1f} min remaining (need 15+ min)", flush=True)
                    except:
                        should_interpolate = True  # If time check fails, try interpolation anyway
                else:
                    should_interpolate = True
            
            # DECISIVE: Checkpoint interpolation - use top 2 checkpoints if available
            if should_interpolate:
                try:
                    interpolation_info = self._interpolate_checkpoints(
                        self.top_checkpoints[0],
                        self.top_checkpoints[1]
                    )
                except Exception as e:
                    # Fallback: just use best checkpoint if interpolation fails
                    print(f"  Warning: Interpolation failed ({e}), using best checkpoint only", flush=True)
                    safe_remove_directory(self.submission_dir)
                    self._copy_best_checkpoint()
                    interpolation_info = "single_best"
            else:
                # Use single best checkpoint (with generalization_score)
                print(f"Using best checkpoint (generalization_score={best_gen_score:.6f}, eval_loss={best_eval_loss:.6f})", flush=True)
                self._copy_best_checkpoint()
                interpolation_info = "single_best"
            
            self.update_best_checkpoint = False
            # add a loss.txt file to the submission directory
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{self.best_checkpoint_info['step']},{best_eval_loss},{best_gen_score},{interpolation_info}")


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics: Dict[str, Any]) -> Optional[float]:
        eval_loss = None
        if state.log_history:
            last_log_entry = state.log_history[-1]
            eval_loss = last_log_entry.get("eval_reward", None)
            print(f"choose eval_loss ({eval_loss}) as eval_reward from: last_log_entry: {last_log_entry}; \n metrics: {metrics}", flush=True)
        else:
            print(f"state.log_history is empty", flush=True)
            
        if eval_loss is not None:
            eval_loss = - eval_loss
            
        return eval_loss
    
    def penalize_eval_loss(self, eval_loss: float) -> float:
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool: 
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)  # Make end_time timezone-aware in UTC
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"*** current time: {now} end_time: {end_time} time_diff: {time_diff}", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps
        # FRESH: Time-aware adaptive training
        self.time_aware_mode = "normal"  # normal, tight, ample

    def __call__(self, global_step: int) -> dict:
        # FRESH: Time-aware adaptive training - adjust strategy based on remaining time
        remaining_minutes = None
        if self.end_time:
            try:
                end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
                end_time_obj = end_time_obj.replace(tzinfo=timezone.utc)
                now = datetime.datetime.now(timezone.utc)
                remaining_minutes = (end_time_obj - now).total_seconds() / 60
                
                # Adjust mode based on remaining time
                if remaining_minutes < TIME_TIGHT_THRESHOLD:
                    self.time_aware_mode = "tight"
                elif remaining_minutes > TIME_AMPLE_THRESHOLD:
                    self.time_aware_mode = "ample"
                else:
                    self.time_aware_mode = "normal"
            except:
                pass
        
        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}
        
        # DECISIVE: Adaptive evaluation frequency - more frequent near end
        # FRESH: Time-aware - adjust frequency based on remaining time
        if self.max_steps != -1 and self.max_steps > 0:
            progress = global_step / self.max_steps
            
            # In tight time mode, evaluate more frequently to catch best checkpoint quickly
            if self.time_aware_mode == "tight":
                if progress > PROGRESS_THRESHOLD_TIGHT_70:  # Last 30% of training
                    if global_step % EVAL_FREQUENT_STEPS_TIGHT == 0 and global_step > 1:
                        return {"eval": True, "reason": "periodic_frequent_tight"}
                elif progress > PROGRESS_THRESHOLD_TIGHT_50:  # Last 50% of training
                    if global_step % EVAL_FREQUENT_STEPS_NORMAL == 0 and global_step > 1:
                        return {"eval": True, "reason": "periodic_frequent_tight"}
            elif progress > PROGRESS_THRESHOLD_FREQUENT:  # Last 20% of training
                # Evaluate every N steps (very frequent to catch best checkpoint)
                if global_step % EVAL_FREQUENT_STEPS_NORMAL == 0 and global_step > 1:
                    return {"eval": True, "reason": "periodic_frequent"}
            elif progress > PROGRESS_THRESHOLD_MODERATE:  # Last 40% of training
                # Evaluate every N steps (moderately frequent)
                if global_step % EVAL_MODERATE_STEPS == 0 and global_step > 1:
                    return {"eval": True, "reason": "periodic_frequent"}
        
        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}
        
        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f"***ALERT: The time is about to run out need to eval & save the model", flush=True)
                # the eval time might be higher than the end_time, so we need to let the pod not stop by setting a flag for this
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}
        
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            return {"eval": True, "reason": "max_step"}

        return {"eval": False, "reason": "none"}


def set_generation_config(model_name, model):
    try:
        if model_name in ERROR_GENERATION_CONFIG_MODELS:
            model.generation_config = GenerationConfig(temperature=None, top_p=None)
    except:
        print(f"Error setting generation config for model {model_name}")
        pass


def resize_if_needed(model_name, model, token_nums):
    try:
        if model_name in MIS_MATCH_VOCAB_SIZE_MODELS:
            model.resize_token_embeddings(token_nums)
    except:
        print(f"Error resizing token embeddings for model {model_name}")
        pass


def init_wandb(train_request: Dict):
    # set wandb_mode=offline; do not upload the data to wandb export WANDB_MODE=offline
    return True
    task_id = train_request["task_id"]
    expected_repo_name = train_request["expected_repo_name"]
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = train_request["wandb_log_dir"]
    os.environ["WANDB_RUN_ID"] = f"{task_id}_{expected_repo_name}"
    os.environ["WANDB_NAME"] = f"{task_id}_{expected_repo_name}"
    if is_main_process(LOCAL_RANK):
        os.makedirs(train_request["wandb_log_dir"], exist_ok=True)
    return True


class ProgressiveBatchSizeCallback(TrainerCallback):
    def __init__(
        self,
        initial_batch_size: int = 16,
        max_batch_size: int = 128,
        stability_steps: int = 50,
        min_speed_improvement: float = 0.05,  # 5% minimum speed improvement
        eval_accuracy_drop_threshold: float = 0.01,  # 1% accuracy drop threshold
        batch_size_multiplier: float = 1.5,  # Increase by 50% each time
        memory_safety_margin: float = 0.9,  # Use 90% of available memory
    ):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.stability_steps = stability_steps
        self.min_speed_improvement = min_speed_improvement
        self.eval_accuracy_drop_threshold = eval_accuracy_drop_threshold
        self.batch_size_multiplier = batch_size_multiplier
        self.memory_safety_margin = memory_safety_margin
        
        # State tracking
        self.current_batch_size = None
        self.base_learning_rate = None
        self.stability_check_start_step = None
        self.previous_losses = []
        self.previous_speeds = []  # Steps per second
        self.previous_eval_losses = []
        self.best_eval_loss = None
        self.last_step_time = None
        self.stable_training_confirmed = False
        self.scaling_active = False
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize batch size scaling monitoring at training start"""
        if not is_main_process(LOCAL_RANK):
            return control
        
        # Note: Batch size should be set before trainer creation (done in training scripts)
        # This callback monitors training and logs recommendations for batch size scaling
        self.current_batch_size = args.per_device_train_batch_size
        self.base_learning_rate = args.learning_rate
        self.stability_check_start_step = state.global_step
        self.scaling_active = True
        
        print(f"ProgressiveBatchSize: Monitoring initialized with batch_size={self.current_batch_size}, lr={self.base_learning_rate}", flush=True)
        print(f"ProgressiveBatchSize: Will monitor training stability and recommend batch size increases", flush=True)
        return control
    
    def _check_training_stability(self, state) -> bool:
        """Check if training is stable (loss is decreasing smoothly)"""
        if len(self.previous_losses) < self.stability_steps:
            return False
        
        # Check if loss is decreasing (last 20% of stability window)
        recent_losses = self.previous_losses[-int(self.stability_steps * 0.2):]
        if len(recent_losses) < 2:
            return False
        
        # Loss should be decreasing or stable (not increasing significantly)
        loss_trend = recent_losses[-1] - recent_losses[0]
        if loss_trend > 0.1:  # Loss increased by more than 0.1
            return False
        
        # Check for NaN or Inf
        if any(not isinstance(l, float) or not (l == l) for l in recent_losses):
            return False
        
        return True
    
    def _check_memory_usage(self) -> bool:
        """Check if we're near memory limit"""
        if not torch.cuda.is_available():
            return False
        
        try:
            memory_allocated = torch.cuda.memory_allocated(LOCAL_RANK) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(LOCAL_RANK) / (1024**3)  # GB
            memory_total = torch.cuda.get_device_properties(LOCAL_RANK).total_memory / (1024**3)  # GB
            
            memory_usage_ratio = memory_reserved / memory_total
            safe_memory_limit = self.memory_safety_margin
            
            if memory_usage_ratio >= safe_memory_limit:
                print(f"ProgressiveBatchSize: Memory usage {memory_usage_ratio:.2%} >= {safe_memory_limit:.2%}, near limit", flush=True)
                return True
        except Exception as e:
            print(f"ProgressiveBatchSize: Error checking memory: {e}", flush=True)
        
        return False
    
    def _calculate_speed_improvement(self) -> Optional[float]:
        """Calculate speed improvement from batch size increase"""
        if len(self.previous_speeds) < 2:
            return None
        
        # Compare average speed of last batch size vs previous
        # This is a simplified check - in practice, we'd track speeds per batch size
        recent_speeds = self.previous_speeds[-10:] if len(self.previous_speeds) >= 10 else self.previous_speeds
        if len(recent_speeds) < 2:
            return None
        
        avg_speed = sum(recent_speeds) / len(recent_speeds)
        # For simplicity, assume speed scales linearly with batch size (in practice it's sublinear)
        # We'll use a heuristic: if we doubled batch size, we expect >50% speedup
        return avg_speed
    
    def _check_eval_accuracy_drop(self) -> bool:
        """Check if validation accuracy has dropped"""
        if len(self.previous_eval_losses) < 2:
            return False
        
        if self.best_eval_loss is None:
            return False
        
        current_eval_loss = self.previous_eval_losses[-1]
        # If eval loss increased significantly, accuracy dropped
        if current_eval_loss > self.best_eval_loss * (1 + self.eval_accuracy_drop_threshold):
            print(f"ProgressiveBatchSize: Eval loss increased from {self.best_eval_loss:.6f} to {current_eval_loss:.6f}", flush=True)
            return True
        
        return False
    
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor training and adjust batch size"""
        if not is_main_process(LOCAL_RANK) or not self.scaling_active:
            return control
        
        # Track training loss
        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                self.previous_losses.append(last_log["loss"])
                # Keep only recent losses for stability check
                if len(self.previous_losses) > self.stability_steps * 2:
                    self.previous_losses = self.previous_losses[-self.stability_steps:]
        
        # Track training speed (steps per second)
        if self.last_step_time is not None:
            time_diff = datetime.datetime.now() - self.last_step_time
            if time_diff.total_seconds() > 0:
                speed = 1.0 / time_diff.total_seconds()
                self.previous_speeds.append(speed)
                if len(self.previous_speeds) > 100:
                    self.previous_speeds = self.previous_speeds[-100:]
        self.last_step_time = datetime.datetime.now()
        
        # Check if we should stop scaling
        if self._check_memory_usage():
            print(f"ProgressiveBatchSize: Stopping scaling - near memory limit", flush=True)
            self.scaling_active = False
            return control
        
        if self._check_eval_accuracy_drop():
            print(f"ProgressiveBatchSize: Stopping scaling - validation accuracy dropped", flush=True)
            self.scaling_active = False
            return control
        
        # Check training stability before first increase
        if not self.stable_training_confirmed:
            if state.global_step >= self.stability_check_start_step + self.stability_steps:
                if self._check_training_stability(state):
                    self.stable_training_confirmed = True
                    print(f"ProgressiveBatchSize: Training stability confirmed at step {state.global_step}", flush=True)
                else:
                    print(f"ProgressiveBatchSize: Training not stable yet, waiting...", flush=True)
            return control
        
        # Monitor and log recommendations for batch size increases
        # Note: Actual batch size changes require restarting training with new parameters
        steps_since_last_check = state.global_step - self.stability_check_start_step
        if steps_since_last_check >= self.stability_steps and self.current_batch_size < self.max_batch_size:
            # Calculate recommended new batch size
            new_batch_size = int(self.current_batch_size * self.batch_size_multiplier)
            new_batch_size = min(new_batch_size, self.max_batch_size)
            
            if new_batch_size > self.current_batch_size:
                # Scale learning rate proportionally (linear scaling rule)
                lr_scale = new_batch_size / self.current_batch_size
                new_learning_rate = self.base_learning_rate * lr_scale
                
                print(f"ProgressiveBatchSize: RECOMMENDATION - Training is stable, could increase batch size", flush=True)
                print(f"ProgressiveBatchSize:   Current: batch_size={self.current_batch_size}, lr={args.learning_rate:.8f}", flush=True)
                print(f"ProgressiveBatchSize:   Recommended: batch_size={new_batch_size}, lr={new_learning_rate:.8f} (scale={lr_scale:.2f})", flush=True)
                print(f"ProgressiveBatchSize:   Note: To apply, restart training with these new parameters", flush=True)
                
                # Track that we've made a recommendation (for future auto-restart implementation)
                self.stability_check_start_step = state.global_step
                self.stable_training_confirmed = False  # Need to confirm stability again
                self.previous_losses = []  # Reset for new stability check
                self.previous_speeds = []  # Reset speed tracking
        
        return control
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Track evaluation metrics"""
        if not is_main_process(LOCAL_RANK):
            return control
        
        eval_loss = metrics.get("eval_loss", None)
        if eval_loss is None and state.log_history:
            last_log_entry = state.log_history[-1]
            eval_reward = last_log_entry.get("eval_reward", None)
            if eval_reward is not None:
                # For GRPO: negate reward to convert to loss
                eval_loss = -eval_reward
        
        if eval_loss is not None:
            self.previous_eval_losses.append(eval_loss)
            if len(self.previous_eval_losses) > 20:
                self.previous_eval_losses = self.previous_eval_losses[-20:]
            
            if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
        
        return control


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback to prevent overfitting.
    Stops training when eval_loss doesn't improve for 'patience' evaluations.
    Works with both standard eval_loss and GRPO's eval_reward (negated).
    
    Patience is adaptive based on remaining time and training progress.
    """
    def __init__(self, patience: int = 300, min_delta: float = 0.0001, end_time: str = "", max_steps: int = -1):
        self.base_patience = patience
        self.patience = patience  # Will be adjusted adaptively
        self.min_delta = min_delta
        self.best_loss = None
        self.wait = 0
        self.stopped_epoch = 0
        self.end_time = end_time
        self.max_steps = max_steps
        self.eval_times = []  # Track evaluation times to estimate overhead
    
    def _calculate_adaptive_patience(self, state) -> int:
        """
        Calculate adaptive patience based on remaining time and training progress.
        In tight time situations, use lower patience to save time.
        In early training, use lower patience to avoid wasting time on poor runs.
        """
        base_patience = self.base_patience
        
        # Adjust based on remaining time
        if self.end_time:
            try:
                end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
                end_time_obj = end_time_obj.replace(tzinfo=timezone.utc)
                now = datetime.datetime.now(timezone.utc)
                remaining_minutes = (end_time_obj - now).total_seconds() / 60
                
                # In tight time (< 30 min), reduce patience significantly
                if remaining_minutes < 30:
                    base_patience = max(10, int(base_patience * 0.2))  # 20% of base, min 10
                elif remaining_minutes < 60:
                    base_patience = max(20, int(base_patience * 0.4))  # 40% of base, min 20
                elif remaining_minutes < 120:
                    base_patience = int(base_patience * 0.6)  # 60% of base
            except:
                pass
        
        # Adjust based on training progress
        if self.max_steps > 0:
            progress = state.global_step / self.max_steps
            # Early in training: lower patience (don't waste time on bad runs)
            if progress < 0.2:
                base_patience = max(15, int(base_patience * 0.5))
            # Late in training: higher patience (might be near convergence)
            elif progress > 0.8:
                base_patience = int(base_patience * 1.2)
        
        return base_patience
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Update adaptive patience
        self.patience = self._calculate_adaptive_patience(state)
        
        # Try to get eval_loss first, then eval_reward (for GRPO)
        eval_loss = metrics.get("eval_loss", None)
        if eval_loss is None and state.log_history:
            last_log_entry = state.log_history[-1]
            eval_reward = last_log_entry.get("eval_reward", None)
            if eval_reward is not None:
                # For GRPO: negate reward to convert to loss (lower is better)
                eval_loss = -eval_reward
        
        if eval_loss is None:
            return control
        
        if self.best_loss is None:
            self.best_loss = eval_loss
            self.wait = 0
        elif eval_loss < self.best_loss - self.min_delta:
            # Significant improvement
            self.best_loss = eval_loss
            self.wait = 0
            print(f"Early stopping: Improved eval_loss to {eval_loss:.6f} (best: {self.best_loss:.6f}), resetting patience counter (patience={self.patience})", flush=True)
        else:
            # No improvement
            self.wait += 1
            print(f"Early stopping: No improvement for {self.wait}/{self.patience} evaluations. Best loss: {self.best_loss:.6f}, Current: {eval_loss:.6f}", flush=True)
            if self.wait >= self.patience:
                print(f"Early stopping triggered at step {state.global_step}. Best loss: {self.best_loss:.6f}. Stopping training to prevent overfitting.", flush=True)
                control.should_training_stop = True
                self.stopped_epoch = state.epoch
        
        return control