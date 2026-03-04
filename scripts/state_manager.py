import json 
import os
import redis
import time
from typing import Optional, Tuple
from contextlib import contextmanager

STATE_KEY = "state"
STATE_LOCK_KEY = "state_lock"
LOCK_TIMEOUT = 30  # seconds
LOCK_ACQUIRE_TIMEOUT = 10  # seconds

def _get_redis_client() -> redis.Redis:
    """Get a Redis client connection with configuration from environment variables."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    password = os.getenv("REDIS_PASSWORD", None)
    db = int(os.getenv("REDIS_DB", 0))
    
    return redis.Redis(
        host=host,
        port=port,
        password=password,
        db=db,
        decode_responses=True
    )


@contextmanager
def state_lock():
    """Context manager for acquiring and releasing state lock."""
    client = _get_redis_client()
    lock = client.lock(STATE_LOCK_KEY, timeout=LOCK_TIMEOUT)
    
    acquired = False
    try:
        acquired = lock.acquire(blocking=True, timeout=LOCK_ACQUIRE_TIMEOUT)
        if not acquired:
            raise RuntimeError(f"Could not acquire state lock within {LOCK_ACQUIRE_TIMEOUT} seconds")
        yield lock
    finally:
        if acquired:
            try:
                lock.release()
            except Exception as e:
                # Log but don't fail if release fails (lock may have expired)
                print(f"Warning: Failed to release state lock: {e}", flush=True)


def get_state() -> dict:
    """Get state from Redis with automatic retry on failure."""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            client = _get_redis_client()
            value = client.get(STATE_KEY)
            
            if value is None:
                return {}
            
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse state JSON (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return {}
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"Warning: Redis connection error (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            # Return empty state on final failure
            return {}
        except Exception as e:
            print(f"Error: Unexpected error getting state: {e}", flush=True)
            return {}
    
    return {}


def set_state(state: dict) -> None:
    """Set state in Redis with atomic update using lock."""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            with state_lock():
                client = _get_redis_client()
                json_value = json.dumps(state, ensure_ascii=False)
                client.set(STATE_KEY, json_value)
                return
        except RuntimeError as e:
            print(f"Warning: Could not acquire lock (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"Warning: Redis connection error (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise
        except Exception as e:
            print(f"Error: Unexpected error setting state: {e}", flush=True)
            raise


def get_state_atomic() -> Tuple[dict, int]:
    """Get state with version number for optimistic locking.
    
    Returns:
        Tuple of (state_dict, version_number)
    """
    client = _get_redis_client()
    with state_lock():
        value = client.get(STATE_KEY)
        version = client.incr(f"{STATE_KEY}:version") if value else 1
        
        if value is None:
            return {}, version
        
        try:
            return json.loads(value), version
        except json.JSONDecodeError:
            return {}, version


def set_state_atomic(state: dict, expected_version: Optional[int] = None) -> bool:
    """Set state only if version matches (optimistic locking).
    
    Args:
        state: State dictionary to set
        expected_version: Expected version number. If None, always succeeds.
    
    Returns:
        True if update succeeded, False if version mismatch
    """
    client = _get_redis_client()
    with state_lock():
        if expected_version is not None:
            current_version = int(client.get(f"{STATE_KEY}:version") or 0)
            if current_version != expected_version:
                return False
        
        json_value = json.dumps(state, ensure_ascii=False)
        client.set(STATE_KEY, json_value)
        client.incr(f"{STATE_KEY}:version")
        return True


def test():
    state = get_state()
    print(json.dumps(state, indent=4, ensure_ascii=False))
    
if __name__ == "__main__":
    test()