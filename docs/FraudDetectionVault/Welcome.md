# Welcome to MyProject

**MyProject** is a Dockerized Python project template designed for reliability, reproducibility, and ease of use. This vault serves as the central knowledge base for the project.

## 📂 Navigation

- [[architecture|Architecture Overview]]: Understanding the system design and technology stack.
- [[KANBAN|Project Kanban Board]]: Track tasks, bugs, and feature requests.

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)

### Quick Start (Docker)
Run the application using Docker Compose:
```bash
docker-compose up --build
```

### Local Development
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
2. **Setup Pre-commit Hooks:**
   ```bash
   pre-commit install
   ```

## 🛠️ Development Tools

- **Linting:** We use `ruff` for fast Python linting.
  ```bash
  docker-compose run app ruff check .
  ```
- **Testing:** We use `pytest` for unit testing.
  ```bash
  docker-compose run app pytest
  ```
- **CLI Management:** The `manage.py` script provides project-specific commands.

## 📝 Notes
- This documentation is built to be viewed in **Obsidian**.
- The `docs/` folder is mapped directly to this vault.
