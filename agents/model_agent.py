# model_agent.py

from tools.logging_tools import setup_logger, log_event, log_error
from tools.data_tools import load_dataset, basic_train_val_split
from core.session_service import SessionService

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score


class ModelAgent:
    """
    Model Agent:
    - Loads dataset
    - Splits into train/validation
    - Trains a baseline model (RandomForest)
    - Computes score
    - Logs experiment into session memory
    - Updates best_score
    - Simulates a 'long-running' training step using status flags
    """

    def __init__(self, session_service: SessionService):
        self.logger = setup_logger("ModelAgent")
        self.session_service = session_service

    def run(self, session_id: str, model_params: dict | None = None):
        """
        Train a baseline model for the current session.

        model_params: optional dict to override default RandomForest params.
        """
        if model_params is None:
            model_params = {}

        try:
            session = self.session_service.get_session(session_id)
            if session is None:
                raise ValueError(f"Session '{session_id}' not found")

            dataset_path = session["dataset_path"]
            target_col = session["target"]
            task_type = session["task_type"]

            log_event(self.logger, "ModelAgent",
                      f"Starting model training for task_type={task_type}")

            # Load data
            df = load_dataset(dataset_path)

            # Basic split
            X_train, X_val, y_train, y_val = basic_train_val_split(df, target_col)

            # Merge defaults with user params safely
            default_params = {"n_estimators": 100, "random_state": 42}

            # User parameters override defaults
            combined_params = {**default_params, **model_params}

            if task_type == "classification":
                model = RandomForestClassifier(**combined_params)
            else:
                model = RandomForestRegressor(**combined_params)


            # --- Long-running operation simulation: mark as RUNNING ---
            self.session_service.update_session(session_id, "training_status", "RUNNING")
            log_event(self.logger, "ModelAgent", "Training status: RUNNING")

            # Fit model (this could be long-running for big datasets)
            model.fit(X_train, y_train)

            # Predictions & score
            y_pred = model.predict(X_val)

            if task_type == "classification":
                score = accuracy_score(y_val, y_pred)
            else:
                score = r2_score(y_val, y_pred)

            # --- Training done ---
            self.session_service.update_session(session_id, "training_status", "COMPLETED")
            log_event(self.logger, "ModelAgent",
                      f"Training completed with score={score:.4f}")

            # Log experiment in session memory
            experiment = {
                "model_name": model.__class__.__name__,
                "task_type": task_type,
                "params": model.get_params(),
                "score": float(score)
            }
            self.session_service.add_experiment(session_id, experiment)

            best_score = self.session_service.get_session(session_id)["best_score"]

            return {
                "status": "success",
                "task_type": task_type,
                "model_name": model.__class__.__name__,
                "score": float(score),
                "best_score": float(best_score) if best_score is not None else None,
                "training_status": "COMPLETED"
            }

        except Exception as e:
            log_error(self.logger, "ModelAgent", str(e))
            self.session_service.update_session(session_id, "training_status", "ERROR")
            return {"status": "error", "message": str(e)}
