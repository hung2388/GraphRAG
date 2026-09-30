from google import genai
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# ============================================================
# LOAD CONFIG
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ============================================================
# CLIENT
# ============================================================

client = genai.Client(api_key=GEMINI_API_KEY)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


# ============================================================
# LOCAL SEARCH
# ============================================================

def local_search(entity_name, hops=2, limit_chunks=5):

    def _query(tx):

        query = f"""
        MATCH (start:Entity {{name: $name}})

        OPTIONAL MATCH
            (start)-[*1..{hops}]-(neighbor:Entity)

        WITH
            start,
            collect(DISTINCT neighbor) AS neighbors

        WITH
            [start] + neighbors AS all_nodes

        UNWIND all_nodes AS n

        MATCH (n)-[:MENTIONED_IN]->(c:Chunk)

        RETURN DISTINCT
            n.name AS entity_name,
            n.type AS entity_type,
            c.text AS chunk_text

        LIMIT $limit
        """

        result = tx.run(
            query,
            name=entity_name,
            limit=limit_chunks
        )

        return [dict(record) for record in result]


    with driver.session() as session:

        return session.execute_read(
            _query
        )


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(entity_name, hops=2):

    rows = local_search(
        entity_name,
        hops=hops
    )

    if not rows:
        return None

    # Các entity tìm được
    entities_found = sorted(
        set(
            r["entity_name"]
            for r in rows
        )
    )

    # Các chunk liên quan
    chunks = sorted(
        set(
            r["chunk_text"]
            for r in rows
        )
    )

    context = (
        "Các entity liên quan:\n"
        + ", ".join(entities_found)
        + "\n\n"
    )

    context += (
        "Đoạn văn gốc liên quan:\n"
        + "\n---\n".join(chunks)
    )

    return context


# ============================================================
# EXTRACT ENTITY FROM QUESTION
# ============================================================

def extract_query_entity(question):

    prompt = f"""
Câu hỏi: "{question}"

Xác định tên thực thể chính mà câu hỏi đang hỏi về.

Chỉ trả về ĐÚNG tên thực thể.
Không giải thích.
Không thêm chữ nào khác.

Nếu không xác định được, trả về:
UNKNOWN
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context):

    prompt = f"""
Bạn là chatbot hỏi đáp dựa trên Knowledge Graph.

Hãy trả lời câu hỏi bằng tiếng Việt dựa ONLY trên thông tin
được cung cấp trong context.

Nếu context không đủ thông tin để trả lời,
hãy nói rõ rằng không tìm thấy đủ thông tin.

Context:
{context}

Câu hỏi:
{question}

Trả lời:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# CHATBOT
# ============================================================

def chatbot(question, hops=2):

    # --------------------------------------------------------
    # Bước 1:
    # Gemini xác định entity trong câu hỏi
    # --------------------------------------------------------

    entity = extract_query_entity(question)

    print("\n[Query Entity]:", entity)

    if entity == "UNKNOWN":

        return (
            "Xin lỗi, tôi không xác định được "
            "thực thể bạn đang hỏi."
        )


    # --------------------------------------------------------
    # Bước 2:
    # Neo4j tìm entity + neighbor + chunk
    # --------------------------------------------------------

    context = build_context(
        entity,
        hops=hops
    )

    if not context:

        return (
            f"Không tìm thấy thông tin liên quan "
            f"đến '{entity}' trong Knowledge Graph."
        )


    # --------------------------------------------------------
    # Debug: xem context mà LLM nhận được
    # --------------------------------------------------------

    print("\n========== RETRIEVED CONTEXT ==========")
    print(context)
    print("========================================")


    # --------------------------------------------------------
    # Bước 3:
    # Gemini generate answer
    # --------------------------------------------------------

    return generate_answer(
        question,
        context
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("       GRAPH RAG CHATBOT")
    print("========================================")

    print("Neo4j:", NEO4J_URI)

    print("\nKG đã được load sẵn.")
    print("Không chạy bước chunk extraction.")
    print("Gõ 'exit' để thoát.")

    try:

        while True:

            question = input("\nBạn hỏi: ")

            if question.lower() == "exit":
                break

            try:

                answer = chatbot(
                    question,
                    hops=2
                )

                print("\nBot trả lời:")
                print(answer)

            except Exception as e:

                print("\n!!! ERROR !!!")
                print(e)

    finally:

        driver.close()