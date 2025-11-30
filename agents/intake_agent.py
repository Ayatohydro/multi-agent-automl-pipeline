# intake_agent.py

from tools.logging_tools import setup_logger, log_event, log_error
from tools.data_tools import load_dataset, detect_task_type
from core.session_service import SessionService


class IntakeAgent:
    """
    Intake Agent:
    - Receives dataset path & target column
    - Loads dataset
    - Detects task type
    - Updates session memory
    """

    def __init__(self, session_service: SessionService):
        self.logger = setup_logger("IntakeAgent")
        self.session_service = session_service

    def run(self, session_id: str, dataset_path: str, target_col: str):
        try:
            log_event(self.logger, "IntakeAgent", f"Starting intake for dataset: {dataset_path}")

            # Load dataset using custom tool
            df = load_dataset(dataset_path)

            # Detect task type (classification / regression)
            task_type = detect_task_type(df, target_col)

            # Store in session memory
            self.session_service.update_session(session_id, "dataset_path", dataset_path)
            self.session_service.update_session(session_id, "target", target_col)
            self.session_service.update_session(session_id, "task_type", task_type)

            log_event(self.logger, "IntakeAgent",
                      f"Session Updated: target={target_col}, task_type={task_type}")

            return {
                "status": "success",
                "task_type": task_type,
                "rows": len(df),
                "columns": list(df.columns)
            }

        except Exception as e:
            log_error(self.logger, "IntakeAgent", str(e))
            return {"status": "error", "message": str(e)}
