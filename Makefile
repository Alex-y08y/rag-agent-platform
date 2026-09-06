.PHONY: help install backend frontend dev docker-up docker-down test test-cov lint clean seed eval

PYTHON := python
PIP := pip
BACKEND_DIR := backend
FRONTEND_DIR := frontend

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt
	cd $(FRONTEND_DIR) && npm install

backend: ## Run backend dev server
	cd $(BACKEND_DIR) && $(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend: ## Run frontend dev server
	cd $(FRONTEND_DIR) && npm run dev

dev: ## Run both backend and frontend
	@echo "Starting backend on :8000 and frontend on :5173..."
	@cd $(BACKEND_DIR) && $(PYTHON) -m uvicorn app.main:app --reload --port 8000 &
	@cd $(FRONTEND_DIR) && npm run dev

docker-up: ## Start full stack with Docker Compose
	docker-compose up -d --build

docker-down: ## Stop full stack
	docker-compose down

docker-lite: ## Start lightweight stack
	docker-compose -f docker-compose.lite.yml up -d --build

test: ## Run pytest
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run pytest with coverage
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest tests/ -v --cov=app --cov-report=term-missing

lint: ## Run code quality checks
	cd $(BACKEND_DIR) && $(PYTHON) -m py_compile app/main.py

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(FRONTEND_DIR)/dist $(FRONTEND_DIR)/node_modules/.vite

seed: ## Seed demo data
	cd $(BACKEND_DIR) && $(PYTHON) ../scripts/seed_demo_data.py

eval: ## Run RAG evaluation
	cd $(BACKEND_DIR) && $(PYTHON) ../scripts/run_evaluation.py

init-db: ## Initialize database tables
	cd $(BACKEND_DIR) && $(PYTHON) -c "from app.core.database import init_db; init_db(); print('Database initialized')"
