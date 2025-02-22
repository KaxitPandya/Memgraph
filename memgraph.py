import os
from dotenv import load_dotenv
from langchain_community.chains.graph_qa.memgraph import MemgraphQAChain
from langchain_community.graphs import MemgraphGraph
from langchain_openai import ChatOpenAI
load_dotenv()  

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
url = "bolt://localhost:7687"
username = ""  
password = ""  

graph = MemgraphGraph(url=url, username=username, password=password, refresh_schema=True)

# Clear existing data
# graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
# graph.query("DROP GRAPH")
# graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")

# Seed the graph
seed_query = """
MERGE (john:Person {id: "John", title: "Director of Digital Marketing"})
MERGE (jane:Person {id: "Jane", title: "Chief Marketing Officer"})
MERGE (john)-[:COLLABORATES_WITH]->(jane)
"""
graph.query(seed_query)
graph.refresh_schema()

llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4-turbo")

chain = MemgraphQAChain.from_llm(llm, graph=graph, allow_dangerous_requests=True)
response = chain.invoke("What is John's title?")
print("Generated answer:", response["result"])