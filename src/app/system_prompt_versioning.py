# =====================================================================
# WHAT THIS SCRIPT DOES
# It saves the RAG app's SYSTEM PROMPT, together with the pipeline settings (chunk size, overlap,
# embedding dimensions, k), as a new VERSION in Langfuse (Prompt Management).
# Why? Prompts change often. With versioning you can:
#   - see which prompt (and which settings) produced which evaluation scores,
#   - test a new prompt under the label "staging" before it reaches users,
#   - move it to "production" when the metrics improve, and roll back if they get worse,
#   - do all of that without changing the app code (the app asks Langfuse for the prompt).
#
# HOW VERSIONING WORKS IN LANGFUSE
#   - A prompt has a NAME (here "rag_app_system_prompt").
#   - Every create_prompt call with the same name adds a NEW VERSION (1, 2, 3, ...). Old versions are kept.
#   - A LABEL is a movable pointer to one version. A label sits on one version at a time:
#     giving it to a new version removes it from the old one.
#       "production" -> the version the app gets when it asks for the prompt without a label
#       "staging"    -> a version you are still testing (used below)
#       "latest"     -> Langfuse keeps this one on the newest version by itself
#   - The CONFIG is a small dict saved with each version (here the RAG settings), so every prompt
#     version records the pipeline settings it belongs to.
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST are read from the
#                             environment (newer versions also accept LANGFUSE_BASE_URL)
#
#   STEP 1  define the settings (chunk_size, chunk_overlap, output_dimensions, k) and the system_prompt text
#      |
#      v
#   STEP 2  langfuse = get_client()        connects to your Langfuse project using the keys from .env
#      |
#      v
#   STEP 3  langfuse.create_prompt(...)    saves a NEW VERSION of the prompt, with label "staging" and the config
#      |
#      v
#   STEP 4  print the saved prompt text, its version number and its labels
#      |
#      v
#   (later, not in this script)
#      the app asks Langfuse for the prompt:  langfuse.get_prompt("rag_app_system_prompt", label="staging")
#      -> run evaluate_rag_app.py with it -> if the scores improve, move the label "production" to this version
#
# WHERE THIS FITS IN THE PROJECT
#   rag_workflow.py (the app) -> generate_goldens.py / generate_eval_dataset.py -> evaluate_rag_app.py (scores)
#   -> THIS SCRIPT versions the prompt + settings, so each evaluation run can be tied to the exact
#   prompt version and settings that produced it.
#
# MISMATCHES WITH rag_workflow.py (read this before relying on the saved config)
#   The values below do not match the app file you gave me:
#                          this script        rag_workflow.py
#     chunk_size           300                300          -> same
#     chunk_overlap        30                 0            -> DIFFERENT
#     output_dimensions    1024               800 (the embedder's `dimensions`)   -> DIFFERENT
#     k                    3                  3            -> same
#     system prompt        this longer text   the shorter text inside augmentation()  -> DIFFERENT
#   Also, the app does not fetch the prompt from Langfuse yet: it uses its own hard-coded text.
#   So today the saved version describes a setup that the app is not running. Either update this
#   script to the app's real values, or update the app (and delete saved-embeddings, see rag_workflow.py)
#   to the values below. I did not change the values here, because I cannot know which side is the one you want.
# =====================================================================

# get_client = returns the Langfuse client (one shared client, configured from the environment variables)
from langfuse import get_client
# load_dotenv reads the .env file (the Langfuse keys live there)
from dotenv import load_dotenv

# load the api keys
# It must run BEFORE get_client(), because get_client() reads the keys from the environment.
# (arguments of load_dotenv: dotenv_path=".env" to point to a file, override=True to replace
#  existing variables, verbose=True to warn when no .env is found)
load_dotenv()


# =====================================================================
# STEP 1: THE SETTINGS AND THE PROMPT TEXT
# =====================================================================

# The pipeline settings that go with this prompt version. They are stored in the prompt's config below.
# Keep them identical to the values the app really uses (see the mismatch list at the top).
#   chunk_size        -> tokens per chunk (the splitter's chunk_size)
#   chunk_overlap     -> tokens repeated between neighbouring chunks
#   output_dimensions -> the embedder's `dimensions`
#   k                 -> how many chunks the retriever returns
chunk_size = 300
chunk_overlap = 30
output_dimensions = 1024
k = 3

# The system prompt: the fixed rules for the model.
# It is a plain text prompt with no {{variables}}. The question and the context are added by the app in
# the human message, not here. NOTE: Langfuse reads {{name}} (double curly braces) as a variable, so do not
# write that inside the text unless you want a variable.
# The last two sentences ("simple language", "sticks to the input and the context", "factually correct")
# look aimed at the simple explanation, faithfulness and answer correctness metrics.
# Change the text, run this script again, and you get a new version to compare with the old one.
system_prompt = """You are a helpful assistant. Answer the user query
based on the given context only. If you do not know the answer
say I don't know. Do not add any preamble to the response.
Always try to answer in simple language. make sure your answer sticks to the input and the available context,
and be factually correct."""


# =====================================================================
# STEP 2: CONNECT TO LANGFUSE
# =====================================================================

# add system prompt to langfuse

# get_client() -> returns the Langfuse client. It reads these environment variables:
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY -> from your Langfuse project settings
#   LANGFUSE_HOST                            -> the server address. EU cloud (default): https://cloud.langfuse.com,
#                                               US cloud: https://us.cloud.langfuse.com, or your own self-hosted address
# If the keys are missing or wrong, the next call fails with an authentication error.
langfuse = get_client()


# =====================================================================
# STEP 3: SAVE THE PROMPT AS A NEW VERSION
# =====================================================================

# create_prompt(...) -> sends the prompt to Langfuse and returns the saved prompt object.
# Arguments used here:
#   name   -> the prompt's name. The same name = a new version of the same prompt. A new name = a new prompt.
#   type   -> "text" (one string, used here) or "chat" (a list of messages, see below)
#   prompt -> the content. For type="text" it is a string.
#             For type="chat" it is a list like [{"role": "system", "content": "..."},
#                                                {"role": "user", "content": "{{query}}"}]
#   labels -> list of labels to put on this new version. "staging" here, so it is NOT the production version yet.
#             Use "production" only when you want the app to start using it (when the app asks for the
#             default label). Labels you add move away from the older versions that had them.
#   config -> any dict, saved with this version and read back as prompt.config. Here it records the RAG settings.
#             It could also hold the model name or its settings (for example {"model": "gpt-5-nano"}).
# Extra arguments you can add:
#   commit_message="shorter, simpler answers"  -> a note for this version, like a git commit message
#                                                 (in versions of the SDK that support it)
#   tags=["rag", "evaluation"]                 -> tags for the prompt name, for searching in the Langfuse UI
# NOTE: every run of this script creates ANOTHER version. Run it when the prompt or the settings changed.
created_prompt = langfuse.create_prompt(
    name="rag_app_system_prompt",
    type="text",
    prompt=system_prompt,
    labels=["baseline", "staging"],
    config={
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "output_dims": output_dimensions,
        "k": k
    }
)
# (create_prompt is a direct request that finishes before the next line, so no langfuse.flush() is needed here.)


# =====================================================================
# STEP 4: CHECK WHAT WAS SAVED
# =====================================================================

# the saved prompt text
print(created_prompt.prompt)
# the version number Langfuse gave it (1 the first time, then 2, 3, ...)
print(created_prompt.version)
# the labels of this version: ["staging"] (Langfuse may also show its own "latest" label)
print(created_prompt.labels)

# More you can read from created_prompt: .name, .config (the dict above), and
#   created_prompt.compile() -> the text with any {{variables}} filled in (this prompt has none)
#
# HOW THE APP WOULD USE THIS PROMPT (not part of this script; for example inside augmentation() in rag_workflow.py):
#   prompt_obj = langfuse.get_prompt("rag_app_system_prompt", label="staging")
#   system_text = prompt_obj.compile()      # the system prompt text from Langfuse
#   k_value = prompt_obj.config["k"]        # the saved settings
#   get_prompt arguments: label="production" (or leave both out for the production version),
#                         version=2 to pin an exact version,
#                         fallback="..." the text to use if Langfuse cannot be reached,
#                         cache_ttl_seconds=60 how long the app keeps the prompt before asking again.
#
# HOW TO MOVE A VERSION TO PRODUCTION after the evaluation scores look good:
#   langfuse.update_prompt(name="rag_app_system_prompt", version=created_prompt.version, new_labels=["production"])
#   (or change the label in the Langfuse web page). To roll back, give "production" back to an older version.
#
# Run this file with:  uv run python system_prompt_versioning.py