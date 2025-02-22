import os
from dotenv import load_dotenv
load_dotenv()  # Ensure your .env file contains OPENAI_API_KEY
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 1. Initialize the Language Model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4-turbo")

# 2. Load and Split the PDF Document
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "sample.pdf"  # Update this path to your PDF file
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"Loaded {len(documents)} pages from the PDF.")

full_text = "\n".join([doc.page_content for doc in documents])
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(full_text)
print(f"Split text into {len(chunks)} chunks.")

# 3. Build the Knowledge Graph (GraphRAG)
combined_text = "\n".join(chunks)

from langchain_community.cache import InMemoryCache
try:
    from langchain_community.graphs.index_creator import GraphIndexCreator
    GraphIndexCreator.cache = InMemoryCache()
    # Rebuild model schema; suppress errors for now.
    GraphIndexCreator.model_rebuild(raise_errors=False)
except Exception as e:
    print("Warning during model_rebuild:", e)

index_creator = GraphIndexCreator(llm=llm)
graph = index_creator.from_text(combined_text)

# Print extracted knowledge triples
print("\nExtracted Knowledge Triples:")
triples = graph.get_triples()
if triples:
    for triple in triples:
        print(triple)
else:
    print("No triples extracted.")

# Inspect underlying NetworkX graph
print("\nGraph Nodes (with attributes):")
for node in graph._graph.nodes(data=True):
    print(node)
print("\nGraph Edges (with attributes):")
for edge in graph._graph.edges(data=True):
    print(edge)

# 4. Customize the Entity Extraction Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# Create a string listing all node names from the graph.
node_list_str = ", ".join(list(graph._graph.nodes()))
# Custom prompt: instruct LLM to extract only entities that match these nodes.
extraction_template = (
    "Given the following list of graph entities: {node_list}\n"
    "Extract all entities (exact phrases) from the question that match these entities. "
    "If no matching entity is found, output 'NONE'.\n\n"
    "Question: {question}\n"
    "Entities (comma-separated):"
)
# Use partial to prefill node_list.
extraction_prompt = PromptTemplate.from_template(extraction_template).partial(node_list=node_list_str)
entity_extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)

# 5. Build the GraphQAChain and override its extraction chain
from langchain.chains import GraphQAChain
qa_chain = GraphQAChain.from_llm(llm, graph=graph, verbose=True)
# Override its entity_extraction_chain with our custom one.
qa_chain.entity_extraction_chain = entity_extraction_chain

# 6. Interactive Q&A Loop
while True:
    user_query = input("Enter your question about the PDF (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        print("Exiting Q&A. Goodbye!")
        break
    response = qa_chain.invoke(user_query)
    print("\n---")
    print("Question:", user_query)
    print("Answer:", response["result"])
    print("---\n")
