# report_agent.py

from tools.logging_tools import setup_logger, log_event, log_error
from core.session_service import SessionService


class ReportAgent:
    """
    Report Agent:
    - Reads session info, EDA summary, and experiments
    - Produces a markdown-style text report
    """

    def __init__(self, session_service: SessionService):
        self.logger = setup_logger("ReportAgent")
        self.session_service = session_service

    def run(self, session_id: str):
        try:
            session = self.session_service.get_session(session_id)
            if session is None:
                raise ValueError(f"Session '{session_id}' not found")

            dataset_path = session.get("dataset_path")
            target = session.get("target")
            task_type = session.get("task_type")
            eda_summary = session.get("eda_summary", {})
            experiments = session.get("experiments", [])
            best_score = session.get("best_score")

            log_event(self.logger, "ReportAgent", "Generating final report")

            shape = eda_summary.get("shape", ("?", "?"))
            dtypes = eda_summary.get("dtypes", {})
            missing = eda_summary.get("missing_values", {})
            target_dist = eda_summary.get("target_distribution", "N/A")

            lines = []

            lines.append(f"# Kaggle Competition Copilot Report\n")
            lines.append(f"**Dataset:** `{dataset_path}`")
            lines.append(f"**Target Column:** `{target}`")
            lines.append(f"**Task Type:** `{task_type}`")
            lines.append(f"**Shape:** {shape[0]} rows × {shape[1]} columns\n")

            lines.append("## 1. EDA Summary")
            lines.append("**Column Types:**")
            for col, dt in dtypes.items():
                lines.append(f"- `{col}`: {dt}")

            lines.append("\n**Missing Values per Column:**")
            for col, mv in missing.items():
                lines.append(f"- `{col}`: {mv}")

            lines.append("\n**Target Distribution:**")
            lines.append(str(target_dist))

            lines.append("\n## 2. Experiments Run")
            if not experiments:
                lines.append("No experiments were run.")
            else:
                for i, exp in enumerate(experiments, start=1):
                    lines.append(f"### Experiment {i}")
                    lines.append(f"- Model: `{exp.get('model_name')}`")
                    lines.append(f"- Score: `{exp.get('score')}`")
                    lines.append(f"- Key Params: `n_estimators={exp['params'].get('n_estimators')}`, "
                                 f"max_depth={exp['params'].get('max_depth', None)}, "
                                 f"max_features={exp['params'].get('max_features', None)}")

            lines.append("\n## 3. Best Result")
            lines.append(f"**Best Score:** {best_score}")

            if best_score is None or best_score < 0.7:
                lines.append("\nOverall performance is modest. Consider more feature engineering, "
                             "trying different models (e.g. XGBoost), or tuning hyperparameters further.")
            else:
                lines.append("\nThe model achieves a reasonably strong baseline. "
                             "Next steps could include more advanced tuning and cross-validation.")

            report_text = "\n".join(lines)

            log_event(self.logger, "ReportAgent", "Report generation completed")

            return {"status": "success", "report": report_text}

        except Exception as e:
            log_error(self.logger, "ReportAgent", str(e))
            return {"status": "error", "message": str(e)}
