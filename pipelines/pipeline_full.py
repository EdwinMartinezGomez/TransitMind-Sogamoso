"""Pipeline: Full end-to-end (Fases 0-4) with state tracking."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.logger import setup_logger
from src.shared.utils import ensure_dir, resolve_path

logger = setup_logger("pipeline_full")

STATE_FILE = resolve_path("experiments/pipeline_state.json")

def _load_state():
    ensure_dir(STATE_FILE.parent)
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_completed": None, "steps": {}}

def _save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def main():
    state = _load_state()
    steps = [
        ("generate_data", "pipelines.pipeline_generate_data"),
        ("train_timegan", "pipelines.pipeline_train_timegan"),
        ("evaluate_tstr", "pipelines.pipeline_evaluate_tstr"),
    ]

    last = state.get("last_completed")
    skip = last is not None
    
    for name, module_name in steps:
        if skip:
            if name == last:
                skip = False
            continue

        logger.info("step_start", step=name)
        start = time.time()
        try:
            import importlib
            mod = importlib.import_module(module_name)
            mod.main()
            elapsed = time.time() - start
            state["last_completed"] = name
            state["steps"][name] = {"status": "success", "duration_s": round(elapsed, 2)}
            _save_state(state)
            logger.info("step_complete", step=name, duration_s=round(elapsed, 2))
        except Exception as e:
            state["steps"][name] = {"status": "failed", "error": str(e)}
            _save_state(state)
            logger.error("step_failed", step=name, error=str(e))
            raise

    logger.info("full_pipeline_complete")

if __name__ == "__main__":
    main()
