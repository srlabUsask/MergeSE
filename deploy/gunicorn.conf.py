"""Gunicorn config for MergeSE.

Run with:
    gunicorn -c deploy/gunicorn.conf.py server.app:app
"""
import multiprocessing
import os

bind = os.environ.get("MERGESE_BIND", "127.0.0.1:8765")
workers = int(os.environ.get("MERGESE_WORKERS", "2"))
threads = int(os.environ.get("MERGESE_THREADS", "8"))
worker_class = "gthread"
timeout = 0            # don't kill long-running uploads
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("MERGESE_LOGLEVEL", "info")
