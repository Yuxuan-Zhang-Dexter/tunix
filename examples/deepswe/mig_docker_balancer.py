"""α: round-robin docker.from_env() across N2D MIG worker IPs.

Loaded by `python -c "import mig_docker_balancer; ..."` BEFORE tunix imports r2egym.

Reads `MIG_WORKER_IPS` env var (comma-separated). Each call to `docker.from_env()`
returns a fresh DockerClient pointed at the next IP via ssh:// (round-robin,
thread-safe). r2egym caches the result in RepoEnv.__init__ (line 146 of
docker.py), so a single episode = a single worker = sticky bash state.

If MIG_WORKER_IPS is empty or unset, falls back to the real docker.from_env()
(local docker socket) — useful for A.1a/A.1b regression checks.
"""

from __future__ import annotations

import os
import threading

import docker

WORKER_IPS = [ip.strip() for ip in os.environ.get("MIG_WORKER_IPS", "").split(",") if ip.strip()]
SSH_USER = os.environ.get("MIG_SSH_USER", "yuxuan")

_lock = threading.Lock()
_idx = 0

_real_from_env = docker.from_env


def _patched_from_env(*args, **kwargs):
    global _idx
    if not WORKER_IPS:
        return _real_from_env(*args, **kwargs)
    with _lock:
        ip = WORKER_IPS[_idx % len(WORKER_IPS)]
        _idx += 1
    kwargs["base_url"] = f"ssh://{SSH_USER}@{ip}"
    return docker.DockerClient(*args, **kwargs)


docker.from_env = _patched_from_env

print(f"[mig_docker_balancer] patched docker.from_env -> round-robin over {len(WORKER_IPS)} workers: {WORKER_IPS}", flush=True)
