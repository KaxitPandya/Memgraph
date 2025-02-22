# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from llama_index.extractors.entity import EntityExtractor
from pydantic import ConfigDict
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from llama_index.core import KnowledgeGraphIndex

model_config = ConfigDict(protected_namespaces=())

username = ""  # Default is empty
password = ""  # Default is empty
url = "bolt://localhost:7687"

graph_store = MemgraphPropertyGraphStore(
    username=username,
    password=password,
    url=url
)

from llama_index.core import SimpleDirectoryReader
import os

file_path = "charles.txt"
if os.path.exists(file_path):
    print(f"The file {file_path} exists.")
else:
    print(f"The file {file_path} does not exist.")

# Load the data
documents = SimpleDirectoryReader(input_files=["charles.txt"]).load_data()

# Configure settings
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.graph_store = graph_store

# Create an entity extractor
entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_to_exclude=["QUANTITY", "CARDINAL", "MONEY", "ORDINAL", "TIME", "DATE", "PERCENT"],
    model_config=model_config,
)

# Create the knowledge graph index
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    extractor=entity_extractor,
    include_embeddings=True,
    max_triplets_per_chunk=10,
)


query_engine = kg_index.as_query_engine()
response = query_engine.query("What is Charles Darwin known for?")
print(response)
