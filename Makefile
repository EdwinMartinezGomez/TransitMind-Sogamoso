.PHONY: generate-data train evaluate full-pipeline mlflow-ui test clean install api run-all run-setup run-services layer2-ingest layer2-api layer2-test

# === Ejecucion unificada (PowerShell) ===
run-all:
	powershell -ExecutionPolicy Bypass -File ./run.ps1

run-setup:
	powershell -ExecutionPolicy Bypass -File ./run.ps1 -Setup

run-services:
	powershell -ExecutionPolicy Bypass -File ./run.ps1 -ServicesOnly

install:
	pip install -r requirements.txt

generate-data:
	python pipelines/pipeline_generate_data.py

train:
	python pipelines/pipeline_train_timegan.py

evaluate:
	python pipelines/pipeline_evaluate_tstr.py

full-pipeline:
	python pipelines/pipeline_full.py

mlflow-ui:
	mlflow ui --backend-store-uri file:./experiments/mlruns --port 5000

api:
	uvicorn src.layer1_timegan.api:app --host 0.0.0.0 --port 8000 --reload

layer2-ingest:
	python -c "from src.layer2_llm.rag.document_ingestion import DocumentIngestion; from src.shared.utils import load_yaml_config; d = DocumentIngestion(load_yaml_config('llm_config.yaml')); print(d.ingest_all(force_reingest=True))"

layer2-api:
	uvicorn src.layer2_llm.api:app --host 0.0.0.0 --port 8001 --reload

layer2-test:
	python pipelines/pipeline_layer2.py

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --cov=src --cov-report=html

clean:
	rm -rf data/processed/*.csv data/synthetic/generated_flows/*.csv
	rm -rf models/timegan/checkpoints/*.pt
	rm -rf experiments/mlruns experiments/pipeline_state.json
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
