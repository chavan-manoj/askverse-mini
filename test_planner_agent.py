import logging
from dotenv import load_dotenv
from askverse_mini.ai.planner_agent import PlannerAgent

def test_planner_agent():
    planner = PlannerAgent()

    while True:
        user_query = input("\nEnter your query (q|quit to quit): ").strip()
        if user_query.lower() in ("q", "quit"):
            print("Exiting the planner agent test.")
            break

        try:
            plan = planner.plan(user_query)
            print("\nGenerated Plan:")
            print(plan)
        except Exception as e:
            logging.error(f"Error generating plan: {e}")

def main():
    logging.basicConfig(level=logging.ERROR)
    load_dotenv()
    test_planner_agent()

if __name__ == "__main__":
    main()
