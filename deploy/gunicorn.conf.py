"""Gunicorn config for MergeSE.

Run with:
    gunicorn -c deploy/gunicorn.conf.py server.app:app
"""
import multiprocessing
import os

bind = os.environ.get("MERGESE_BIND", "127.0.0.1:8765")
# Keep at 1: the job registry is in-process memory, so a job started in worker
# A is a 404 from worker B. Scale with threads instead of workers until the
# job state moves to shared storage.
workers = int(os.environ.get("MERGESE_WORKERS", "1"))
threads = int(os.environ.get("MERGESE_THREADS", "16"))
worker_class = "gthread"
timeout = 0            # don't kill long-running uploads
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("MERGESE_LOGLEVEL", "info")
