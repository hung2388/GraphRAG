import os
import glob
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def clear_db():
    print("Clearing Neo4j database...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    print("Database cleared.")

def clear_files():
    print("Clearing debug logs...")
    debug_files = glob.glob(os.path.join("data", "debug", "*.txt"))
    for f in debug_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error removing {f}: {e}")
            
    report_file = os.path.join("reports", "evaluation_report.json")
    if os.path.exists(report_file):
        print("Clearing evaluation_report.json...")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("[]")
            
    print("Files cleared.")

if __name__ == "__main__":
    clear_db()
    clear_files()
