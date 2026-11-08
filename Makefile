.PHONY: install test run eval

install:
	pip install -r requirements.txt

test:
	pytest backend/tests -v

run:
	uvicorn backend.app.main:app --reload --port 8000

eval:
	python scripts/lang_eval.py
