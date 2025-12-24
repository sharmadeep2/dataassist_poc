import json
import csv
from pathlib import Path
import logging

# Import your log_event function and constants from app.py
try:
    from app import log_event, JSONL_PATH, CSV_PATH, LOG_DIR, logger
except ImportError:
    # Standalone fallback for logger and paths
    LOG_DIR = Path("./logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    JSONL_PATH = LOG_DIR / "events.jsonl"
    CSV_PATH = LOG_DIR / "events.csv"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "app_runtime.log", encoding="utf-8"),
            logging.StreamHandler()
        ],
    )
    logger = logging.getLogger("poc-logging")

    # Dummy log_event for standalone test
    def log_event(event_type, **kwargs):
        event = {"event_type": event_type, **kwargs}
        with JSONL_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        with CSV_PATH.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(event.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(event)
        logger.info(f"Logged event: {event_type}")
        return event

def test_logging():
    logger.info("Starting logging verification test.")

    # Simulate logging three events
    log_event(
        "answer_generated",
        query="Show me the project details for Q3 FY24",
        table_confidence=0.92,
        llm_score=95,
        explanation="The answer is fully supported by context.",
        user_feedback="thumbs_up",
        composite_reward=0.88,
        faithfulness=0.85,
        answer_relevancy=0.93,
        context_count=5,
        doc_ids=["abc123", "def456"],
        extra={"phase": "test_generation"}
    )
    log_event(
        "user_feedback",
        query="Show me the project details for Q3 FY24",
        llm_score=95,
        explanation="User confirmed answer was helpful.",
        user_feedback="thumbs_up",
        composite_reward=0.88,
        context_count=5,
        doc_ids=["abc123", "def456"],
        extra={"phase": "test_feedback"}
    )
    log_event(
        "ragas_evaluated",
        query="Show me the project details for Q3 FY24",
        llm_score=95,
        explanation="Ragas metrics are high.",
        user_feedback="thumbs_up",
        composite_reward=0.88,
        faithfulness=0.85,
        answer_relevancy=0.93,
        context_count=5,
        doc_ids=["abc123", "def456"],
        extra={"phase": "test_ragas"}
    )

    # Check JSONL file
    print("\n--- JSONL Log Tail ---")
    if JSONL_PATH.exists():
        with JSONL_PATH.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-3:]
        for line in lines:
            print(json.dumps(json.loads(line), indent=2))
    else:
        print("JSONL log file not found.")

    # Check CSV file
    print("\n--- CSV Log Tail ---")
    if CSV_PATH.exists():
        with CSV_PATH.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            for row in reader[-3:]:
                print(row)
    else:
        print("CSV log file not found.")

    # Check runtime log
    runtime_log = LOG_DIR / "app_runtime.log"
    print("\n--- Runtime Log Tail ---")
    if runtime_log.exists():
        with runtime_log.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-10:]
        print("".join(lines))
    else:
        print("Runtime log file not found.")

    logger.info("Completed logging verification test.")

if __name__ == "__main__":
    test_logging()