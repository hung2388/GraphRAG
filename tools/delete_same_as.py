import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def delete_same_as():
    print("Deleting all SAME_AS relationships...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("MATCH ()-[r:SAME_AS]->() DELETE r")
    driver.close()
    print("Deleted.")

if __name__ == "__main__":
    delete_same_as()
