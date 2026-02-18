SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python
PIP ?= pip
NPM ?= npm
WEB_DIR := web

.PHONY: help install verify-deps setup-postgres ingest ingest-quick langgraph-dev langgraph-up chat chat-stack gui-stack gui-stack-up web-install web-dev web-check users-db users-init users-setup users-list users-add users-reset-pw users-deactivate users-activate users-remove

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install Python package in editable mode
	$(PIP) install -e .

verify-deps: ## Verify Python dependencies for ingestion/chat
	biomedagent-db verify-deps

setup-postgres: ## Run PostgreSQL setup helper
	biomedagent-db setup_postgres

ingest: ## Run full ingestion pipeline
	biomedagent-db ingest

ingest-quick: ## Run lightweight prototype ingest profile
	biomedagent-db ingest --quick-prototype

langgraph-dev: ## Start LangGraph dev server
	langgraph dev

langgraph-up: ## Start LangGraph server in persistent mode
	langgraph up

chat: ## Start CLI chat (assumes LangGraph server is already running)
	biomedagent-db chat --server-url $${LANGGRAPH_API_URL:-http://localhost:2024} --assistant-id $${BIOAGENT_ASSISTANT_ID:-co_scientist}

chat-stack: ## Start LangGraph dev + CLI chat in one command
	./scripts/run_langgraph_and_chat.sh

gui-stack: ## Start LangGraph dev + Next.js GUI in one command
	./scripts/run_langgraph_and_web.sh

gui-stack-up: ## Start LangGraph up (persistent) + Next.js GUI in one command
	./scripts/run_langgraph_up_and_web.sh

web-install: ## Install web dependencies
	cd $(WEB_DIR) && $(NPM) install

web-dev: ## Run web app in dev mode
	cd $(WEB_DIR) && $(NPM) run dev

web-check: ## Run web typecheck and lint
	cd $(WEB_DIR) && $(NPM) run typecheck && $(NPM) run lint

users-db: ## Create app database if missing (APP_DATABASE_URL)
	$(PYTHON) scripts/manage_users.py init-db

users-init: ## Create app_users table in APP_DATABASE_URL
	$(PYTHON) scripts/manage_users.py init-schema

users-setup: ## Create users table + seed demo users
	$(PYTHON) scripts/manage_users.py init-db
	$(PYTHON) scripts/manage_users.py init-schema
	$(PYTHON) scripts/manage_users.py seed

users-list: ## List all users and status
	$(PYTHON) scripts/manage_users.py list

users-add: ## Add user: make users-add EMAIL=x@y.com NAME="Dr. X" ROLE=user
	$(PYTHON) scripts/manage_users.py add --email $(EMAIL) --name "$(NAME)" --role $(or $(ROLE),user)

users-reset-pw: ## Reset password: make users-reset-pw EMAIL=x@y.com
	$(PYTHON) scripts/manage_users.py reset-password --email $(EMAIL)

users-deactivate: ## Deactivate: make users-deactivate EMAIL=x@y.com
	$(PYTHON) scripts/manage_users.py deactivate --email $(EMAIL)

users-activate: ## Activate: make users-activate EMAIL=x@y.com
	$(PYTHON) scripts/manage_users.py activate --email $(EMAIL)

users-remove: ## Remove: make users-remove EMAIL=x@y.com
	$(PYTHON) scripts/manage_users.py remove --email $(EMAIL)
