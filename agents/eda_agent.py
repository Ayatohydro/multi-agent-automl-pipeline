# eda_agent.py

import pandas as pd
from tools.logging_tools import setup_logger, log_event, log_error
from tools.data_tools import load_dataset
from core.session_service import SessionService


class EDAAgent:
    """
    EDA Agent:
    - Loads dataset
    - Computes basic statistics
    - Returns summary dictionary
    """

    def __init__(self, session_service: SessionService):
        self.logger = setup_logger("EDAAgent")
        self.session_service = session_service

    def run(self, session_id: str):
        try:
            session = self.session_service.get_session(session_id)

            dataset_path = session["dataset_path"]
            target = session["target"]

            log_event(self.logger, "EDAAgent", "Performing EDA...")

            df = load_dataset(dataset_path)

            summary = {
                "shape": df.shape,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "target_distribution": df[target].value_counts().to_dict()
                if df[target].nunique() < 30 else "Too many unique values",
                "description": df.describe(include="all").fillna("").to_dict()
            }
             #  save in session memory so ReportAgent can use it later
            self.session_service.update_session(session_id, "eda_summary", summary)

            log_event(self.logger, "EDAAgent",
                      f"EDA completed: rows={df.shape[0]}, cols={df.shape[1]}")

            return {"status": "success", "summary": summary}

        except Exception as e:
            log_error(self.logger, "EDAAgent", str(e))
            return {"status": "error", "message": str(e)}
