from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_neighbors(tx, entity_name, hops=1):
    query = f"""
    MATCH (a:Entity {{name: $name}})-[*1..{hops}]-(b:Entity)
    RETURN DISTINCT b.name AS name, b.type AS type
    """
    result = tx.run(query, name=entity_name)
    return [dict(record) for record in result]

with driver.session() as session:
    neighbors = session.execute_read(get_neighbors, "Đại Việt Sử Ký Toàn Thư", hops=2)
    for n in neighbors:
        print(n)

driver.close()