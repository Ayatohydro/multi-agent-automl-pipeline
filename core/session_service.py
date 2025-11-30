
class SessionService:
    """
    Simple in-memory session store.
    Demonstrates 'Sessions & Memory' required for Kaggle Agents submission.
    """

    def __init__(self):
        self.sessions = {}

    def create_session(self, session_id: str):
        """Create a new session with default fields."""
        self.sessions[session_id] = {
            "dataset_path": None,
            "target": None,
            "task_type": None,
            "experiments": [],
            "best_score": None
        }
        return self.sessions[session_id]

    def get_session(self, session_id: str):
        """Retrieve an existing session."""
        return self.sessions.get(session_id, None)

    def update_session(self, session_id: str, key: str, value):
        """Update any field in the session."""
        if session_id in self.sessions:
            self.sessions[session_id][key] = value

    def add_experiment(self, session_id: str, experiment: dict):
        """Add model experiment details."""
        if session_id in self.sessions:
            self.sessions[session_id]["experiments"].append(experiment)

            # update best score automatically
            score = experiment.get("score")
            best = self.sessions[session_id]["best_score"]

            if best is None or score > best:
                self.sessions[session_id]["best_score"] = score
