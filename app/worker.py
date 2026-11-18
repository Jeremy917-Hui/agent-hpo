"""One persistent local process pulls Runs and calls Study.optimize sequentially.

SQLite app metadata arbitrates one local slot. This is not distributed leasing;
PostgreSQL quotas/heartbeat/recovery are deferred to M7.
"""

import os
import subprocess
import sys
import time
from uuid import uuid4

from app.config import SOURCE_ROOT, Settings
from app.runs import RunStore


def ensure_worker(settings):
    store = RunStore(settings)
    existing = store.enable_worker()
    if existing:
        return {"pid": existing["pid"], "worker_id": existing["worker_id"], "started_new_process": False}
    token = uuid4().hex
    log = settings.log_dir / ("worker_launcher_" + token + ".log")
    environment = {**os.environ, "PROJECT_ROOT": str(settings.project_root), "DATA_DIR": str(settings.data_dir),
                   "LOG_DIR": str(settings.log_dir), "TEMP": str(settings.tmp_dir), "TMP": str(settings.tmp_dir),
                   "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(SOURCE_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   "MIN_FREE_DISK_GIB": str(settings.min_free_disk_gib)}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    options = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with log.open("xb") as output:
        process = subprocess.Popen([sys.executable, "-m", "app.worker"], cwd=settings.project_root,
            env=environment, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, **options)
    return {"launcher_pid": process.pid, "log_path": str(log), "started_new_process": True}


def worker_loop(settings):
    settings.prepare()
    store = RunStore(settings)
    store.initialize()
    worker_id = store.register_worker()
    if worker_id is None:
        return 0  # A concurrent launcher lost the single-slot race. It never trains.
    try:
        # Import native numerical/Optuna libraries only after owning the local slot.
        import optuna
        from app.hpo import execute_run
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        while True:
            run_id, stopped = store.claim_next(worker_id)
            if stopped:
                return 0
            if run_id:
                execute_run(settings, run_id, worker_id)
            else:
                time.sleep(0.5)
    finally:
        store.release_worker(worker_id)


if __name__ == "__main__":
    raise SystemExit(worker_loop(Settings.load()))
