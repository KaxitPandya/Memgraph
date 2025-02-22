import os
from dotenv import load_dotenv
from langchain_community.chains.graph_qa.memgraph import MemgraphQAChain
from langchain_community.graphs import MemgraphGraph
from langchain_openai import ChatOpenAI

# 1. Environment Setup and Connection
load_dotenv()  # Loads environment variables from .env file

# Get your OpenAI API key and set your Memgraph connection details
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
url = "bolt://localhost:7687"  # Default Memgraph connection URL
username = ""  # Provide username if authentication is enabled
password = ""  # Provide password if authentication is enabled

# Establish connection to Memgraph and refresh the schema
graph = MemgraphGraph(url=url, username=username, password=password, refresh_schema=True)


# 2. (Optional) Clearing Existing Data
# graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
# graph.query("DROP GRAPH")
# graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")


# 3. Seeding the Graph with Data
# Here we create nodes for two people and a company, and then define relationships.
seed_query = """
MERGE (john:Person {id: "John", name: "John", title: "Director of Digital Marketing", age: 35})
MERGE (jane:Person {id: "Jane", name: "Jane", title: "Chief Marketing Officer", age: 42})
MERGE (acme:Company {id: "AcmeCorp", name: "AcmeCorp", industry: "Technology"})
MERGE (john)-[:COLLABORATES_WITH]->(jane)
MERGE (john)-[:WORKS_FOR]->(acme)
MERGE (jane)-[:WORKS_FOR]->(acme)
"""

graph.query(seed_query)
graph.refresh_schema()  # Refresh the graph schema to pick up new node/relationship types

# 4. Integrating an LLM for Graph-Based Q&A
# Initialize the language model (LLM) with your API key.
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4-turbo")

# Create a MemgraphQAChain instance, which allows you to ask natural language questions
chain = MemgraphQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)

# Example natural language questions
questions = [
    "What is John's title?",
    "Who collaborates with Jane?",
    "Which company does John work for?",
    "What industry is AcmeCorp in?"
]

for question in questions:
    response = chain.invoke(question)
    print(f"Question: {question}\nAnswer: {response['result']}\n")

# 5. Running Advanced Direct Queries
# (a) Query: Find all persons older than 40
query_age = """
MATCH (p:Person)
WHERE p.age > 40
RETURN p.id AS Name, p.title AS Title, p.age AS Age
"""
result_age = graph.query(query_age)
print("Persons older than 40:", result_age)

# (b) Query: Find paths between John and Jane (exploring relationships)
query_path = """
MATCH path = (john:Person {id: "John"})-[*]-(jane:Person {id: "Jane"})
RETURN path
"""
result_path = graph.query(query_path)
print("Paths between John and Jane:", result_path)

# 6. Demonstrating Graph Updates
# (a) Update a node's property (e.g., update John's age)
update_query = """
MATCH (john:Person {id: "John"})
SET john.age = 36
RETURN john
"""
result_update = graph.query(update_query)
print("Updated John's information:", result_update)

# (b) Deleting a node (if needed)
# For example, if you want to delete a temporary node:
delete_query = """
MATCH (temp:Person {id: "Temp"})
DETACH DELETE temp
"""
# Uncomment the line below if you have a node to delete.
# graph.query(delete_query)
