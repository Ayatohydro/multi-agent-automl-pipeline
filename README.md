 Kaggle Competition Copilot: Multi-Agent AutoML for Tabular Data

1. Problem

When you join a new Kaggle competition or dataset, the first few hours are usually spent on:
- understanding the dataset,
- doing basic EDA,
- training a first baseline model,
- tracking experiments,
- and writing a summary.

This project automates that **early workflow** for **tabular datasets** using a multi-agent system.

2. Solution Overview

The system takes:
- a CSV file path
- a target column name

Then it runs:
1. **Intake** – load dataset, detect task type.
2. **EDA** – basic statistics, missing values, target distribution.
3. **Baseline Model** – train a RandomForest classifier/regressor, compute a score.
4. **Planner** – suggest and optionally run new experiments (tuned parameters).
5. **Report** – generate a Markdown report combining EDA + experiments + best result.

All state is stored in a simple session object so agents can share information.

3. Architecture

- `core/session_service.py`  
  In-memory session store:
  - dataset_path, target, task_type
  - EDA summary
  - experiments list
  - best_score
  - training_status

- `agents/`
  - `intake_agent.py` – sets up session using the dataset.
  - `eda_agent.py` – EDA, saves summary to session.
  - `model_agent.py` – trains RandomForest, logs experiments, updates best_score, sets training_status (RUNNING → COMPLETED).
  - `planner_agent.py` – reads best experiment and suggests new model parameters.
  - `report_agent.py` – creates a Markdown report from session data.

- `tools/`
  - `data_tools.py` – custom tools for:
    - loading CSVs,
    - splitting train/validation with safe stratification,
    - one-hot encoding categorical features.
  - `logging_tools.py` – shared logging utilities (observability).

- `core/orchestrator.py`  
  Orchestrates the full pipeline:
  **Intake → EDA → Baseline Model → Planner Loop → Report**.

4. Features Demonstrated (for the course)

- **Multi-agent system** – separate agents with clear roles.
- **Tools** – custom data tools + logging tools.
- **Sessions & Memory** – `SessionService` shared across agents.
- **Long-running operations** – `training_status` around model fitting.
- **Observability** – structured logs from every agent and from the orchestrator.

5. How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
