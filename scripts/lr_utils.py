import math
import numpy as np
from typing import List, Optional, Tuple

def _suggest_learning_rates(
    best_lr: float,
    n: int,
    log_range: float = 0.4
) -> list[float]:
    if n < 0:
        raise ValueError("Number of tries (n) cannot be negative.")
    if n == 0:
        return []
    if n == 1:
        return [best_lr]

    # print("best_lr: ", best_lr)
    # Calculate the lower and upper bounds for the learning rate search
    # on a logarithmic scale.
    lower_bound = best_lr / (10 ** log_range)
    upper_bound = best_lr * (10 ** log_range)

    # Convert bounds to log scale
    log_lower = math.log10(lower_bound)
    log_upper = math.log10(upper_bound)

    # Generate n logarithmically spaced values
    log_spaced_values = [
        log_lower + i * (log_upper - log_lower) / (n - 1)
        for i in range(n)
    ]

    # Convert the log-spaced values back to the original scale
    learning_rates = [10 ** val for val in log_spaced_values]

    return sorted(learning_rates)


def suggest_learning_rates(
    best_lr: float,
    n: int,
    log_range: float = 0.2
) -> list[float]:
    lrs = _suggest_learning_rates(best_lr, n, log_range)
    if n % 2 == 1:
        return lrs
    else: # exclude one and add best_lr to the middle
        lrs = lrs[1:] + [best_lr]
        lrs = sorted(lrs)
        return lrs


def extend_learning_rates(
    lr: float,
    n: int,
    log_range: float = 0.2
) -> list[float]:
    lrs = _suggest_learning_rates(lr, n, log_range)
    # loop over lrs to find the item that is the closest to lr (should be the same) and replace it with lr and move that item to the left (index = 0)
    # Find the index of the learning rate in lrs that is closest to lr
    closest_idx = min(range(len(lrs)), key=lambda i: abs(lrs[i] - lr))
    # Replace that value with the actual lr to ensure precision
    lrs[closest_idx] = lr
    # Move that lr to the first position (index 0)
    if closest_idx != 0:
        lrs.insert(0, lrs.pop(closest_idx))
    return lrs


def smart_explore_learning_rates(
    current_lr: float,
    previous_runs: List[dict],
    n: int,
    log_range: float = 0.2
) -> List[float]:
    """
    Smart learning rate exploration using previous run results.
    
    Uses a simple model to predict loss from LR and explores promising regions.
    Falls back to standard log-scale exploration if insufficient data.
    
    Args:
        current_lr: Current learning rate (best so far)
        previous_runs: List of previous run results with 'lr' and 'current_loss' keys
        n: Number of learning rates to generate
        log_range: Log range for exploration (default 0.2 = ~1.58x range)
    
    Returns:
        List of learning rates to explore, with current_lr first
    """
    if n <= 1:
        return [current_lr] if n == 1 else []
    
    # Need at least 2 previous runs to build a model
    if len(previous_runs) < 2:
        # Fall back to standard exploration
        return extend_learning_rates(current_lr, n, log_range)
    
    # Extract LR and loss pairs
    lrs = []
    losses = []
    for run in previous_runs:
        if "lr" in run and "current_loss" in run:
            try:
                lr_val = float(run["lr"])
                loss_val = float(run["current_loss"])
                if loss_val > 0 and lr_val > 0:  # Valid values
                    lrs.append(lr_val)
                    losses.append(loss_val)
            except (ValueError, TypeError):
                continue
    
    if len(lrs) < 2:
        return extend_learning_rates(current_lr, n, log_range)
    
    # Fit a simple quadratic model: loss = a * lr^2 + b * lr + c
    # Use log space for better numerical stability
    # Filter out invalid values (must be positive for log10)
    valid_pairs = [(lr, loss) for lr, loss in zip(lrs, losses) if lr > 0 and loss > 0]
    if len(valid_pairs) < 2:
        return extend_learning_rates(current_lr, n, log_range)
    
    log_lrs = [math.log10(lr) for lr, _ in valid_pairs]
    log_losses = [math.log10(loss) for _, loss in valid_pairs]
    
    try:
        # Fit polynomial (degree 2) in log space
        coeffs = np.polyfit(log_lrs, log_losses, deg=min(2, len(lrs) - 1))
        
        # Find optimal LR (minimum of the fitted curve)
        # For quadratic: loss = a*x^2 + b*x + c, minimum at x = -b/(2*a)
        if len(coeffs) >= 2 and abs(coeffs[0]) > 1e-10:
            optimal_log_lr = -coeffs[1] / (2 * coeffs[0])
            optimal_lr = 10 ** optimal_log_lr
        else:
            # Linear fit or degenerate, use current_lr
            optimal_lr = current_lr
        
        # Clamp optimal_lr to reasonable range around current_lr
        min_lr = current_lr / (10 ** log_range)
        max_lr = current_lr * (10 ** log_range)
        optimal_lr = max(min_lr, min(max_lr, optimal_lr))
        
        # Generate exploration around optimal LR
        # Use smaller range for smart exploration (more focused)
        smart_range = log_range * 0.6  # 60% of standard range
        lower_bound = optimal_lr / (10 ** smart_range)
        upper_bound = optimal_lr * (10 ** smart_range)
        
        # Ensure current_lr is in the range
        lower_bound = min(lower_bound, current_lr / 2)
        upper_bound = max(upper_bound, current_lr * 2)
        
        log_lower = math.log10(lower_bound)
        log_upper = math.log10(upper_bound)
        
        # Generate n-1 values (we'll add current_lr)
        if n > 1:
            if n == 2:
                # Special case: just generate one value at midpoint
                log_spaced = [(log_lower + log_upper) / 2]
            else:
                # Generate n-1 values evenly spaced
                log_spaced = [
                    log_lower + i * (log_upper - log_lower) / (n - 2)
                    for i in range(n - 1)
                ]
            candidate_lrs = [10 ** val for val in log_spaced]
        else:
            candidate_lrs = []
        
        # Add current_lr and ensure it's first
        candidate_lrs.append(current_lr)
        candidate_lrs = sorted(set(candidate_lrs))  # Remove duplicates
        
        # If we have fewer than n values, fill with standard exploration
        if len(candidate_lrs) < n:
            standard_lrs = extend_learning_rates(current_lr, n, log_range)
            # Merge and deduplicate
            all_lrs = sorted(set(candidate_lrs + standard_lrs))
            # Take n closest to optimal_lr
            all_lrs.sort(key=lambda x: abs(math.log10(x) - math.log10(optimal_lr)))
            candidate_lrs = all_lrs[:n]
        
        # Ensure current_lr is first
        if current_lr in candidate_lrs:
            candidate_lrs.remove(current_lr)
        candidate_lrs.insert(0, current_lr)
        
        # Limit to n values
        return candidate_lrs[:n]
        
    except Exception as e:
        # Fall back to standard exploration on any error
        print(f"Warning: Smart LR exploration failed ({e}), using standard exploration", flush=True)
        return extend_learning_rates(current_lr, n, log_range)


def test():
    lr = 0.00014523947500000002
    for n in [3,4,5, 6]:
        lrs = extend_learning_rates(lr, n)
        print(lrs)
        assert lrs[0] == lr
    
    # Test smart exploration
    print("\nTesting smart exploration:")
    previous_runs = [
        {"lr": "0.0001", "current_loss": 0.5},
        {"lr": "0.0002", "current_loss": 0.4},
        {"lr": "0.0003", "current_loss": 0.45},
    ]
    smart_lrs = smart_explore_learning_rates(lr, previous_runs, 5)
    print(f"Smart LRs: {smart_lrs}")

if __name__ == "__main__":
    test()