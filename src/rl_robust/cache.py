from pathlib import Path
from joblib import Memory

def setup_cache(cache_dir: str | Path = "cache") -> Memory:
    """Setup joblib memory cache"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Memory(str(cache_dir), verbose=0)

# Global cache instance
memory = setup_cache()
