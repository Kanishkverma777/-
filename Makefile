.PHONY: install train test lint format run-app docker-build docker-run clean download-data

install:
	pip install -e ".[dev]"

download-data:
	python scripts/download_data.py

train:
	python -m heart_disease.train

test:
	pytest tests/ -v --cov=src/heart_disease --cov-report=term-missing

lint:
	ruff check src/ tests/ app/
	mypy src/

format:
	ruff format src/ tests/ app/
	ruff check --fix src/ tests/ app/

run-app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t heart-disease-app .

docker-run:
	docker run -p 8501:8501 heart-disease-app

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/*.egg-info dist build
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

