# test_agents.py - quick developer test script
from core.session_service import SessionService
from agents.intake_agent import IntakeAgent
from agents.eda_agent import EDAAgent
from agents.model_agent import ModelAgent

DATA_PATH = "Bank_Customer_Churn.csv"    
TARGET_COL = "churn"       

def run_demo(session_id="demo_test"):
    session = SessionService()
    session.create_session(session_id)

    intake = IntakeAgent(session)
    eda = EDAAgent(session)
    model_agent = ModelAgent(session)

    try:
        print("\n--- Running Intake Agent ---")
        result1 = intake.run(session_id, DATA_PATH, TARGET_COL)
        print(result1)

        if result1.get("status") != "success":
            print("Intake failed, aborting demo.")
            return

        print("\n--- Running EDA Agent ---")
        result2 = eda.run(session_id)
        print(result2)

        print("\n--- Running Model Agent ---")
        result3 = model_agent.run(session_id)
        print(result3)

        print("\n--- Session After Training ---")
        print(session.get_session(session_id))

    except Exception as e:
        print("Error during demo:", e)

if __name__ == "__main__":
    run_demo()
