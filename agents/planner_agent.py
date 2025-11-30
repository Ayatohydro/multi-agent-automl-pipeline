# planner_agent.py

from tools.logging_tools import setup_logger, log_event, log_error
from core.session_service import SessionService


class PlannerAgent:
    """
    Planner Agent:
    - Reads past experiments & best_score from session
    - Suggests next model parameter experiments
    """

    def __init__(self, session_service: SessionService):
        self.logger = setup_logger("PlannerAgent")
        self.session_service = session_service

    def run(self, session_id: str, n_suggestions: int = 3):
        try:
            session = self.session_service.get_session(session_id)
            if session is None:
                raise ValueError(f"Session '{session_id}' not found")

            experiments = session.get("experiments", [])
            best_score = session.get("best_score", None)
            task_type = session.get("task_type", "classification")

            log_event(self.logger, "PlannerAgent",
                      f"Planning next experiments. best_score={best_score}")

            suggestions = []

            # If no experiments yet, suggest a simple baseline
            if not experiments:
                suggestions.append({
                    "description": "Baseline RandomForest with default params",
                    "model_params": {}
                })
            else:
                # Simple heuristic based on last experiment
                last_exp = experiments[-1]
                last_params = last_exp.get("params", {})
                last_n_estimators = last_params.get("n_estimators", 100)

                # Suggest increasing trees
                suggestions.append({
                    "description": f"Increase n_estimators from {last_n_estimators} to {last_n_estimators * 2}",
                    "model_params": {
                        "n_estimators": last_n_estimators * 2
                    }
                })

                # Suggest limiting depth
                suggestions.append({
                    "description": "Try limiting max_depth to 5",
                    "model_params": {
                        "max_depth": 5
                    }
                })

                # Suggest using fewer features via max_features
                suggestions.append({
                    "description": "Try max_features='sqrt'",
                    "model_params": {
                        "max_features": "sqrt"
                    }
                })

            # Trim to n_suggestions
            suggestions = suggestions[:n_suggestions]

            log_event(self.logger, "PlannerAgent",
                      f"Generated {len(suggestions)} suggestions")

            return {
                "status": "success",
                "best_score": best_score,
                "suggestions": suggestions
            }

        except Exception as e:
            log_error(self.logger, "PlannerAgent", str(e))
            return {"status": "error", "message": str(e)}
