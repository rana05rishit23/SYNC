#!/usr/bin/env bash
set -e
pip install -r requirements.txt
pytest backend/tests -q
uvicorn backend.app.main:app --reload --port 8000
