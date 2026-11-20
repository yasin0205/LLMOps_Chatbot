# =====================================================================
# WHAT THIS FILE IS
# The RAG chatbot app as a Python module (converted from the notebook).
# It builds (or loads) the knowledge base, creates a retriever, and builds a LangGraph app
# called `graph`. Other scripts import it:  from app.rag_workflow import graph
# (generate_eval_dataset.py does this to run every golden question through the app).
#
# IMPORTANT: everything at the top level of this file runs when the file is IMPORTED.
# That means the knowledge base is built or loaded (and the first build costs embedding calls)
# every time another script imports `graph`. Only the demo question at the very end is protected
# by  if __name__ == "__main__":  so it runs only when you start this file directly.
#
# FLOW OF THE WHOLE SYSTEM
#
# Part A: build or load the knowledge base (runs once on import)
#
#   .env file
#      |  load_dotenv()  -> loads OPENAI_API_KEY into the environment
#      v
#   does the saved-embeddings folder exist?
#      |
#      |-- YES -> load_knowledge_base()   opens the saved Chroma collection (no API cost)
#      |
#      |-- NO  -> upsert_documents(chunk_size, chunk_overlap):
#                    data/processed/*.txt
#                       |  DirectoryLoader + TextLoader     -> reads all .txt files
#                       v
#                    docs (list of Documents, one per file)
#                       |  RecursiveCharacterTextSplitter   -> splits each document into chunks of
#                       |                                      about chunk_size tokens (300)
#                       v
#                    chunks (list of smaller Documents)
#                       |  embedder (text-embedding-3-small) -> turns each chunk into a vector
#                       |                                      of `dimensions` numbers (800)
#                       v
#                    Chroma collection "rag_demo", saved in <root>/saved-embeddings
#      |
#      v
#   vs.as_retriever(k=3)  ->  retriever: returns the 3 most similar chunks for a question
#
# Part B: answer a question (runs on every graph.invoke)
#
#   {"query": "your question"}
#      |
#      v
#   START
#      v
#   retrieve        embed the query, fetch the top 3 chunks, join them into `context`
#      v
#   augmentation    build the prompt template (system rules + context + query)
#      v
#   generation      prompt | llm | StrOutputParser  -> the answer text
#      v
#   END
#      v
#   result["response"]   (result also keeps query, retrieved_docs, context, prompt)
#
# WHAT TO KNOW BEFORE CHANGING SETTINGS
#   The saved-embeddings folder is built ONCE. If it exists, it is loaded as it is.
#   So after you change chunk_size, chunk_overlap, the embedding model or `dimensions`,
#   DELETE the saved-embeddings folder first, otherwise the old store is still used
#   (and a different `dimensions` value raises a dimension-mismatch error when you search).
#   If a first build fails halfway (for example an API error), the folder already exists but is empty or
#   incomplete, and the next run would load it. Delete the folder and run again.
# =====================================================================

# TextLoader reads one text file; DirectoryLoader reads every file in a folder
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# ChatOpenAI = the chat LLM wrapper; OpenAIEmbeddings = the text-to-vector model wrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# ChatPromptTemplate builds a prompt with {placeholders} that are filled in later
from langchain_core.prompts import ChatPromptTemplate
# Document = LangChain's container: .page_content (the text) + .metadata (a dict, e.g. {"source": path})
from langchain_core.documents import Document
# Splits long text into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chroma = the vector database that stores chunks and their embeddings
from langchain_chroma import Chroma
# Path gives a clean, cross-platform way to work with files and folders
from pathlib import Path
# load_dotenv reads the .env file (where OPENAI_API_KEY lives)
from dotenv import load_dotenv

# Reads the .env file and puts each KEY=value line into the environment variables.
# ChatOpenAI and OpenAIEmbeddings then find OPENAI_API_KEY automatically.
# It returns True if it loaded something
# Arguments you can use:
#   load_dotenv(dotenv_path=".env")  -> point to a specific file (default: searches upward from the current folder)
#   load_dotenv(override=True)       -> let the .env values replace variables that already exist (default False)
#   load_dotenv(verbose=True)        -> print a warning if no .env file is found
load_dotenv(verbose=True)

# define the paths
# __file__ = this file's path. .resolve() makes it absolute.
# .parent = go up one folder. Three .parent in a row means three folders up.
# With the file at <root>/src/app/rag_workflow.py, three parents up is <root>.
# If you move the file to a different depth, REPO_ROOT points to the wrong place.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# the folder with the cleaned transcripts (the input; the same folder generate_goldens.py reads)
PROCESSED_TRANSCRIPTS_DIR = REPO_ROOT / "data" / "processed"
# the folder where Chroma saves the vectors on disk
VECTOR_STORE_DIR = REPO_ROOT / "saved-embeddings"


# =====================================================================
# PART A, STEP 1: CREATE THE LLM AND EMBEDDING MODEL
# =====================================================================

# The LLM that writes the final answer.
# model        -> which OpenAI model to call. Here "gpt-5-nano" (the cheapest and fastest of the gpt-5 family,
#                 but also the weakest at following rules like "say I don't know". If faithfulness or answer
#                 correctness drop in the evaluation, try "gpt-5-mini".)
#                 Other choices: "gpt-5", "gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini".
#                 Other providers use other classes (e.g. ChatAnthropic).
# Extra arguments you can add:
#   reasoning_effort="low"   -> for reasoning models like gpt-5-mini / gpt-5-nano: "minimal" | "low" | "medium" | "high"
#                               (less effort = faster and cheaper, more effort = better on hard questions)
#   temperature=0            -> more deterministic answers, but only for non-reasoning models
#                               (gpt-4.1/4o family). gpt-5 models accept only the default, so don't set it.
#   max_retries=2            -> how many times to retry a failed API call
#   timeout=60               -> seconds to wait before giving up
#   api_key="..."            -> pass the key directly (default: reads OPENAI_API_KEY)
#   base_url="..."           -> use a different OpenAI-compatible server
llm = ChatOpenAI(model="gpt-5-nano")

# The model that turns text into vectors (lists of numbers) so similar meanings can be found.
# model      -> other choices: "text-embedding-3-large" (more accurate, 3072 numbers by default, more expensive),
#               "text-embedding-ada-002" (older; does not support `dimensions`)
# dimensions -> how many numbers per vector. text-embedding-3 models can be shortened
#               (any whole number up to the full size). Here 800.
#               Smaller = less storage and faster search, but slightly less accurate.
#               (The API price does not change with dimensions; it is charged per token.)
#               Default for text-embedding-3-small is 1536.
# IMPORTANT: every vector in one Chroma collection must come from the same model AND the same dimensions.
#            If you change either, delete the saved-embeddings folder (or use a new collection_name
#            or persist_directory) and embed again.
# Extra arguments you can add:
#   chunk_size=1000          -> how many texts are sent per API request (a batch size; not the same as the splitter's chunk_size)
#   show_progress_bar=True   -> progress bar while embedding many chunks
#   max_retries, timeout, api_key, base_url -> same meaning as in ChatOpenAI
embedder = OpenAIEmbeddings(model="text-embedding-3-small",
                            dimensions=800)  # smaller = less storage, faster search, slightly less accurate


# =====================================================================
# PART A, STEP 2: BUILD OR LOAD THE KNOWLEDGE BASE
# =====================================================================

# the chunking settings, passed to upsert_documents below
# chunk_size    -> maximum tokens per chunk.
#                  Smaller (200-500) = more precise matches but less context per chunk.
#                  Larger (1000+) = more context but more noise, and more cost per question.
#                  With k=3 and chunk_size=300, the LLM receives roughly 900 tokens of context
#                  (with 1024 it would be roughly 3000).
#                  TIP: generate_goldens.py builds its test contexts with the chunk size from your config
#                  (context_chunk_size). If the two sizes differ a lot, the retriever sees much less text than the
#                  expected answers were built from, which can lower recall. Note the sizes when you compare runs.
# chunk_overlap -> tokens repeated between neighbouring chunks.
#                  0 = no repetition (cheapest). 100-200 is common, so a sentence cut at a
#                  boundary still appears whole in one chunk. More overlap = more chunks to embed.
chunk_size = 300
chunk_overlap = 0

# load and update knowledge base
# Does the whole Part A build: load the documents, chunk them, embed them, and save them in Chroma.
# It returns the vector store. It is called only when the saved-embeddings folder does not exist yet.
# NOTE: despite the name "upsert" it only INSERTS. Calling it a second time by hand stores every chunk twice
# (see the warning at vs.add_documents below).
def insert_documents(chunk_size: int, chunk_overlap: int):

    # --- load the docs from source ---
    # DirectoryLoader walks a folder and loads each file with the loader class you choose.
    # path             -> the folder to read: PROCESSED_TRANSCRIPTS_DIR (<root>/data/processed), an absolute
    #                     path, so it does not depend on where you run the script.
    #                     .as_posix() turns the Path into a string with "/".
    # loader_cls       -> which loader reads each file. TextLoader = plain text.
    #                     Other choices: PyPDFLoader (PDF), CSVLoader (CSV), UnstructuredMarkdownLoader (.md)
    # show_progress    -> show a progress bar (needs tqdm installed)
    # Extra arguments you can add:
    #   glob="*.txt"                         -> only load matching files ("**/*.txt" = also inside sub-folders).
    #                                           Default: every non-hidden file.
    #   recursive=True                       -> also search sub-folders
    #   loader_kwargs={"encoding": "utf-8"}  -> passed to TextLoader. Useful on Windows, where the default
    #                                           encoding can cause UnicodeDecodeError. Or {"autodetect_encoding": True}.
    #   use_multithreading=True              -> load files in parallel (helps with many files)
    #   silent_errors=True                   -> skip files that fail instead of stopping
    #   exclude=["*.tmp"]                    -> patterns to skip
    loader = DirectoryLoader(path=PROCESSED_TRANSCRIPTS_DIR.as_posix(),
                             loader_cls=TextLoader,
                             show_progress=True)

    # Returns a list of Document objects, one per file.
    # (loader.lazy_load() gives them one at a time instead, which saves memory for huge folders.)
    docs = loader.load()

    # --- chunk the text ---
    # chunk the text, sized in tokens (o200k_base matches the gpt-5 model family)
    # to mirror DeepEval's ContextConstructionConfig chunk sizing

    # Why chunk? A whole file is too long and too mixed to match one question well.
    # Small chunks let the retriever return only the relevant part.
    # from_tiktoken_encoder -> measures chunk size in TOKENS (what the model counts), not characters.
    # encoding_name -> the tokenizer. "o200k_base" is used by gpt-4o and gpt-5 models;
    #                  "cl100k_base" is used by older models. Alternative: model_name="gpt-4o"
    # chunk_size    -> maximum tokens per chunk (the value passed into this function, see above)
    # chunk_overlap -> tokens repeated between neighbouring chunks (the value passed into this function)
    # Extra arguments you can add:
    #   separators=["\n\n", "\n", " ", ""]  -> the order it tries to split at: paragraph, line, word, character
    #                                          (this default is why it is called "Recursive")
    #   keep_separator=True                 -> keep the separator text in the chunks
    #   add_start_index=True                -> store each chunk's start position in its metadata
    #   strip_whitespace=True               -> trim spaces at the chunk edges
    # Other splitters: CharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter.
    chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    # Takes a list of Documents and returns a longer list of smaller Documents.
    # Each chunk keeps the metadata (source file) of the document it came from.
    # (split_text(str) works on a plain string and returns a list of strings.)
    chunks = chunker.split_documents(docs)
    # (to see how many chunks were created, which is also how many vectors will be embedded: print(len(chunks)))

    # --- the vector store ---
    # Chroma stores each chunk's text, metadata and vector, and can search by similarity.
    # collection_name    -> the name of the collection (like a table). A new name = a fresh empty collection.
    # embedding_function -> the model used to embed the chunks when you add them
    #                       AND to embed the question later. It must be the same model for both.
    # persist_directory  -> folder where Chroma saves the data on disk, so it survives a restart.
    #                       If you leave it out, the data lives only in memory and is lost when the program stops.
    # Extra arguments you can add:
    #   collection_metadata={"hnsw:space": "cosine"} -> choose the distance measure. Choices: "l2" (default), "cosine", "ip".
    #   client / client_settings                     -> connect to a Chroma server instead of a local folder
    # Other vector stores: FAISS, Qdrant, PGVector, Pinecone, InMemoryVectorStore.
    vs = Chroma(collection_name="rag_demo",
                embedding_function=embedder,
                persist_directory=VECTOR_STORE_DIR.as_posix())

    # add docs
    # Embeds every chunk (one OpenAI API call per batch, so this costs a little money) and stores it.
    # Returns the list of ids it created.
    # WARNING: each call creates new random ids, so calling this twice stores every chunk TWICE
    #          in the saved folder, and the retriever may then return duplicates.
    #          Options: pass your own ids, e.g. vs.add_documents(chunks, ids=[f"chunk-{i}" for i in range(len(chunks))])
    #          (same id = overwritten, not duplicated; that would make this function a real upsert),
    #          or call vs.delete_collection() before rebuilding.
    vs.add_documents(chunks)
    return vs

# Opens the saved collection without embedding anything again (so it costs nothing).
# The Chroma arguments are the same as in upsert_documents. They must match the ones used to build the store,
# especially embedding_function (same model and dimensions) and collection_name.
def load_knowledge_base():
    # Chroma stores each chunk's text, metadata and vector, and can search by similarity.
    vs = Chroma(collection_name="rag_demo",
                embedding_function=embedder,
                persist_directory=VECTOR_STORE_DIR.as_posix())
    return vs

# Build the knowledge base only once: load it if the folder exists, build it if not.
# This also fixes the old notebook problem of adding every chunk twice when you run it again.
# Delete the saved-embeddings folder when you want a fresh build (see the notes at the top).
# (A safer check against an empty or half-built store: also test  len(load_knowledge_base().get(limit=1)["ids"]) > 0 )
if VECTOR_STORE_DIR.exists():
    print("Loading existing knowledge base...")
    vs = load_knowledge_base()
else:
    print("Building knowledge base...")
    vs = insert_documents(chunk_size, chunk_overlap)


# =====================================================================
# PART A, STEP 3: CREATE THE RETRIEVER
# =====================================================================

# A retriever takes a question and returns the most relevant chunks.
# search_type="similarity" -> the k closest chunks (this is the default, written out here)
# search_kwargs={"k":3}    -> return the 3 most similar chunks.
#                             Bigger k = more context but more noise and cost. Smaller k = more focused.
# Extra arguments you can add:
#   search_type="mmr"                       -> closest chunks but with variety (no near-duplicates);
#                                              tune with search_kwargs {"fetch_k": 20, "lambda_mult": 0.5}
#   search_type="similarity_score_threshold" -> only return chunks above a score,
#                                              with search_kwargs {"score_threshold": 0.5}
#   search_kwargs={"filter": {"source": "..."}} -> only search chunks whose metadata matches
# To see the scores yourself: vs.similarity_search_with_score("question", k=3)
# (with Chroma's default l2 distance, a LOWER number means MORE similar)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k":3})


# =====================================================================
# END OF PART A. PART B uses the retriever to answer questions.
# =====================================================================

# WHAT IS LANGGRAPH?
# LangGraph lets you build an app as a GRAPH of small steps instead of one long script.
# Mental model: an assembly line with a shared clipboard.
#   STATE = the shared clipboard (one dict that holds all the information)
#   NODE  = a station on the line (a plain Python function that does one job)
#   EDGE  = a conveyor belt (decides which station runs next)
#   START / END = the door in and the door out
#   compile() = locks the design and gives you a machine you can run with invoke()
#
# FLOW IN THIS FILE:
#   {"query": "..."} -> START -> retrieve -> augmentation -> generation -> END
#
# HOW IT RUNS (graph.invoke):
#   1. You pass the starting state, e.g. {"query": "..."}.
#   2. LangGraph follows the edge from START to the first node and calls it with the state.
#   3. The node does its job and RETURNS A DICT with only the keys it changed.
#   4. LangGraph merges that dict into the state, then follows the next edge.
#   5. At END, invoke() returns the final state (all keys) as a dict.
#
# RULES TO REMEMBER:
#   - A node never calls the next node. It only returns updates; LangGraph decides what runs next.
#   - A returned key OVERWRITES the old value by default (it does not append).
#     To append instead, use a reducer: Annotated[list, operator.add] in the state class.
#   - Each node sees the state as left by the previous nodes, so "generation" can read
#     "context" and "prompt", which were created by earlier nodes.
#
# STATE AFTER EACH STEP:
#   input         -> query
#   retrieve      -> query, retrieved_docs, context
#   augmentation  -> + prompt
#   generation    -> + response
# =====================================================================

# StateGraph builds the workflow; START and END are the special entry and exit points
# StateGraph -> the builder: you give it the state class, then add_node(...), add_edge(...), compile()
# START      -> a marker for "the entry point" (add_edge(START, "first_node"))
# END        -> a marker for "the exit point" (add_edge("last_node", END))
from langgraph.graph import StateGraph, START, END

# TypedDict -> describes a dict with fixed keys and value types. Used for RAGState (the clipboard).
#              It is only a description: Python does not check the types when the code runs.
# Literal   -> a type hint meaning "exactly one of these exact values", e.g. Literal["yes", "no"].
#              It is not a function and it does nothing at runtime.
#              It is used on a ROUTING function (one that picks the next node), so editors
#              and LangGraph know the possible destinations, and the diagram can draw the branches:
#
#                  def route(state: RAGState) -> Literal["augmentation", "no_answer"]:
#                      if state["retrieved_docs"]:
#                          return "augmentation"
#                      return "no_answer"
#
#                  graph_builder.add_conditional_edges("retrieve", route)
#
#              This file uses only fixed edges (add_edge), so Literal is imported but not used yet.
from typing import Literal, TypedDict

# =====================================================================
# WHAT COMES NEXT IN THIS FILE (and the other options you can use)
#   StateGraph(RAGState)                  -> create the builder with the state class
#   add_node("name", function)            -> register a station (add_node(function) also works;
#                                            the name is then taken from the function name)
#   add_edge("a", "b")                    -> fixed order: run b after a
#   add_conditional_edges("a", fn, map)   -> branching: fn(state) returns the next node's name
#                                            (map is optional: {"label": "node_name"})
#   compile()                             -> returns the runnable graph
#                                            (compile(checkpointer=MemorySaver()) saves state between runs)
#   graph.invoke({...})                   -> run once, returns the final state
#   graph.stream({...}, stream_mode="updates") -> see each node's output as it finishes
#   graph.ainvoke({...})                  -> async version of invoke
# =====================================================================


# =====================================================================
# PART B, STEP 4: DEFINE THE STATE (the shared clipboard)
# =====================================================================

# The state is a shared "notebook" that every node can read from and write to.
# Each field is one named slot. Nodes return only the slots they changed.
class RAGState(TypedDict):

    query: str                        # the user's question (the only input you give)

    retrieved_docs: list[Document]    # the chunks the retriever found
    context: str                      # those chunks joined into one text block

    prompt: ChatPromptTemplate        # the prompt template built by the augmentation node (system rules + context + query)
    response: str                     # the final answer from the LLM
# Note: a field with no special rule is OVERWRITTEN each time a node returns it.
# To append instead, you can use a reducer, e.g. messages: Annotated[list, operator.add]


# =====================================================================
# PART B, STEP 5: THE NODES
# =====================================================================

# ---- Node 1: Retrieval ----

# A node is a normal function: it receives the state and returns a dict of updates.
def retrieve(state: RAGState) -> dict:
    # read the question from the state
    query = state["query"]
    # embeds the query with the same embedder, searches Chroma, returns the top k Documents
    retrieved_docs = retriever.invoke(query)

    # join the text of the chunks, with a blank line between them
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    # these two keys are added to the state
    return {"retrieved_docs":retrieved_docs,
            "context":context}

# ---- Node 2: Augmentation ----

def augmentation(state: RAGState) -> dict:

    # from_messages takes a list of (role, text) pairs.
    # Roles: "system" (rules for the model), "human" (the user), "ai" (a previous model reply).
    # {context} and {query} are placeholders, filled in later when the chain is invoked.

    # Other options: ChatPromptTemplate.from_template("...") for a single message,
    #                MessagesPlaceholder("history") to insert earlier chat messages.

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the user query
                      based on the given context only. If you do not know the answer
                      say I don't know. Do not add any preamble to the response"""),
        ("human", "context: {context}\n\nquery: {query}")
    ])
    # ("based on the context only" keeps the model from using its own knowledge, which reduces hallucination.
    #  The spaces at the start of the lines inside the triple quotes are sent too; harmless but you can remove them.)

    # save the prompt into the state
    return {"prompt": prompt}

# ---- Node 3: Generation ----

# StrOutputParser turns the model's reply object (AIMessage) into a plain string
from langchain_core.output_parsers import StrOutputParser

def generation(state: RAGState) -> dict:

    # read what earlier nodes saved in the state
    query = state["query"]
    context = state["context"]
    prompt = state["prompt"]

    # The | operator chains steps, left to right:
    #   prompt            -> fills {context} and {query} and makes the messages
    #   llm               -> sends the messages to the model, returns an AIMessage
    #   StrOutputParser() -> pulls out the text, so you get a plain str
    rag_chain = prompt | llm | StrOutputParser()
    # The dict keys must match the placeholder names in the prompt.
    # Other ways to run a chain: .stream({...}) to get the answer piece by piece,
    #                            .batch([{...}, {...}]) for several questions, .ainvoke({...}) for async.
    response = rag_chain.invoke({"context":context, "query":query})

    return {"response":response}


# =====================================================================
# PART B, STEP 6: BUILD THE GRAPH
# =====================================================================

# StateGraph needs the state class, so it knows which fields exist
graph_builder = StateGraph(RAGState)

# add_node(name, function): the name is used in edges, the function is what runs.
# (add_node(retrieve) alone also works; the name is then taken from the function name)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("augmentation", augmentation)
graph_builder.add_node("generation", generation)

# add_edge(from, to): run "to" after "from" finishes. This creates the fixed order:
# START -> retrieve -> augmentation -> generation -> END
graph_builder.add_edge(START,"retrieve")
graph_builder.add_edge("retrieve","augmentation")
graph_builder.add_edge("augmentation","generation")
graph_builder.add_edge("generation",END)
# For branching, use add_conditional_edges(source, routing_function, {result: next_node}),
# e.g. if no docs were retrieved, jump straight to END with "I don't know".

# compile() checks the graph and turns it into a runnable app.
# Extra argument: compile(checkpointer=...) saves state between runs (for memory / resuming).
# This `graph` is what other files import.
graph = graph_builder.compile()

# In a notebook, writing `graph` alone on a line draws the flow diagram. In a .py file that line does
# nothing, so it was removed. To see the diagram as text instead, use:
#   print(graph.get_graph().draw_mermaid())


# =====================================================================
# PART B, STEP 7: OUTPUT (ASK A QUESTION)
# =====================================================================

# This block runs only when you start this file directly (python rag_workflow.py),
# not when another file imports it. It matters: generate_eval_dataset.py imports `graph` from this file,
# and without this guard every import would also run this question (one extra LLM call and a printed answer).
#if __name__ == "__main__":
    # invoke() runs the whole graph once. You pass only the input field ("query").
    # It returns the FINAL state: a dict with query, retrieved_docs, context, prompt and response.
    # Other ways to run: graph.stream({...}, stream_mode="updates") shows each node's output as it finishes
    # (stream_mode="values" shows the full state after each step), graph.ainvoke(...) for async.
    #result = graph.invoke({"query": "how much fine was imposed on the lawyer who used fake evidences from chatgpt"})

    # only the answer text
    #print(result["response"])
    # the exact text the LLM was given. If the answer is wrong, check here first:
    # was the right information inside the retrieved chunks?
    # print(result["context"])