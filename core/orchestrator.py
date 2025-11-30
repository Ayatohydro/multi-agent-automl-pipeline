# orchestrator.py

import os
import sys

# Allow imports when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.session_service import SessionService
from tools.logging_tools import setup_logger, log_event
from agents.intake_agent import IntakeAgent
from agents.eda_agent import EDAAgent
from agents.model_agent import ModelAgent
from agents.planner_agent import PlannerAgent
from agents.report_agent import ReportAgent


def run_pipeline(
    dataset_path: str,
    target_col: str,
    session_id: str = "run1",
    n_planned_runs: int = 1
):
    """
    Full pipeline:
    Intake -> EDA -> Baseline Model -> Planner -> Extra Models -> Report
    """

    logger = setup_logger("Orchestrator")
    log_event(logger, "Orchestrator", "Starting pipeline")

    # 1. Create session + agents
    session_service = SessionService()
    session_service.create_session(session_id)

    intake = IntakeAgent(session_service)
    eda = EDAAgent(session_service)
    model_agent = ModelAgent(session_service)
    planner = PlannerAgent(session_service)
    report_agent = ReportAgent(session_service)

    # 2. Intake
    intake_result = intake.run(session_id, dataset_path, target_col)
    if intake_result["status"] != "success":
        log_event(logger, "Orchestrator", f"Intake failed: {intake_result}", "ERROR")
        return

    # 3. EDA
    eda_result = eda.run(session_id)
    if eda_result["status"] != "success":
        log_event(logger, "Orchestrator", f"EDA failed: {eda_result}", "ERROR")
        return

    # 4. Baseline model
    baseline_result = model_agent.run(session_id)
    if baseline_result["status"] != "success":
        log_event(logger, "Orchestrator", f"Model training failed: {baseline_result}", "ERROR")
        return

    # 5. Planner – suggest next experiments
    plan_result = planner.run(session_id)
    if plan_result["status"] == "success":
        suggestions = plan_result["suggestions"]
        log_event(
            logger,
            "Orchestrator",
            f"Planner suggested {len(suggestions)} experiments"
        )

        # Run a few extra experiments (loop agent behavior)
        for i, suggestion in enumerate(suggestions[:n_planned_runs], start=1):
            log_event(
                logger,
                "Orchestrator",
                f"Running planned experiment {i}: {suggestion['description']}"
            )
            model_agent.run(session_id, model_params=suggestion["model_params"])

    # 6. Final report
    report_result = report_agent.run(session_id)
    if report_result["status"] != "success":
        log_event(logger, "Orchestrator", f"Report failed: {report_result}", "ERROR")
        return

    report_text = report_result["report"]

    # Save report to file
    output_path = os.path.join(os.path.dirname(__file__), "..", "report.md")
    output_path = os.path.abspath(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    log_event(logger, "Orchestrator", f"Pipeline completed. Report saved to {output_path}")
    print("\n=== PIPELINE COMPLETED ===")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    # Default demo run using Bank_customer_churn.csv
    run_pipeline(dataset_path="Bank_Customer_Churn.csv", target_col="churn", session_id="bank_run",n_planned_runs=2)
