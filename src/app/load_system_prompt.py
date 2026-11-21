# =====================================================================
# WHAT THIS SCRIPT DOES
# It LOADS the system prompt from Langfuse: the other half of system_prompt_versioning.py.
#   system_prompt_versioning.py -> SAVES a new prompt version (text + settings) in Langfuse
#   load_system_prompt.py       -> FETCHES a saved version, so the app can use it
# It prints the version number, the prompt text, the saved settings (config) and the labels.
#
# WHY LOAD THE PROMPT FROM LANGFUSE?
#   The app does not need a code change when the prompt changes. You edit the prompt in Langfuse,
#   give a label to the new version, and the app picks it up the next time it asks.
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST are read from the
#                             environment (newer versions also accept LANGFUSE_BASE_URL)
#      |
#      v
#   STEP 1  langfuse = get_client()      connects to your Langfuse project
#      |
#      v
#   STEP 2  langfuse.get_prompt(name, type, label)
#              asks Langfuse for the version of "rag_app_system_prompt" that has the label "latest"
#              (the SDK keeps a short-lived copy in memory, so repeated calls are fast)
#      |
#      v
#   STEP 3  print the result: .version, .prompt (the text), .config (the settings), .labels
#
# THE WHOLE PROMPT LIFE CYCLE
#   1. system_prompt_versioning.py   create a new version with the label "staging"
#   2. the app (or this script)      load it with label="staging" and run the evaluation (evaluate_rag_app.py)
#   3. scores improved?              move the label "production" to that version (update_prompt, or the Langfuse web page)
#   4. the live app                  loads label="production" and gets the new prompt without a code change
#   5. scores got worse?             give "production" back to the older version (roll back)
#
# WHERE THIS FITS IN THE PROJECT
#   rag_workflow.py (the app) currently has the system prompt written inside augmentation().
#   This is the code that would replace that hard-coded text (an example is in the comments at the end).
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
# STEP 1: CONNECT TO LANGFUSE
# =====================================================================

# get_client() -> returns the Langfuse client. It reads these environment variables:
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY -> from your Langfuse project settings
#   LANGFUSE_HOST                            -> the server address. EU cloud (default): https://cloud.langfuse.com,
#                                               US cloud: https://us.cloud.langfuse.com, or your own self-hosted address
# If the keys are missing or wrong, the next call fails with an authentication error.
langfuse = get_client()


# =====================================================================
# STEP 2: FETCH THE PROMPT
# =====================================================================

# get_prompt(...) -> asks Langfuse for one saved version and returns a prompt object.
# NOTE: in this file `system_prompt` is the prompt OBJECT. The text itself is system_prompt.prompt.
# (In system_prompt_versioning.py the variable with the same name held the text string.)
# Arguments used here:
#   name  -> the prompt's name, exactly as it was created ("rag_app_system_prompt")
#   type  -> "text" (one string, used here) or "chat" (a list of messages).
#            It must match how the prompt was created.
#   label -> which version to get, by label:
#              "latest"     -> the newest version, whatever its other labels are (used here)
#              "staging"    -> the version you are testing
#              "production" -> the version that is live. It is also the DEFAULT when you pass no label and no version.
# Which one to use?
#   "latest" is handy while developing, but in a live app it is risky: any new version you save goes
#   live at once, even a version you have not tested. A live app should load label="production".
#   A test run should load label="staging".
# Use label OR version, not both:
#   version=2 -> an exact version number (the version never changes, so results are repeatable)
# Extra arguments you can add:
#   fallback="You are a helpful assistant ..."  -> the text to use if Langfuse cannot be reached and nothing is
#                                                  cached, so the app keeps working. The prompt object then has
#                                                  is_fallback set to True.
#   cache_ttl_seconds=60   -> how long the SDK keeps the prompt in memory before asking Langfuse again
#                             (default 60). A new version can take up to this long to reach a running app.
#                             0 turns the cache off (every call goes to Langfuse; fine for testing, slower for a live app).
#   max_retries=2          -> how many times to retry a failed request
#   fetch_timeout_seconds=3 -> seconds to wait for Langfuse before giving up
# If the name does not exist, or no version has that label, the call raises an error (unless a fallback is given).
system_prompt = langfuse.get_prompt(
    name="rag_app_system_prompt",
    type="text",
    label="staging"
)


# =====================================================================
# STEP 3: LOOK AT WHAT WAS LOADED
# =====================================================================

# the version number of the prompt that was loaded (1, 2, 3, ...)
print(system_prompt.version)
# the prompt text (what the model gets as its rules)
print(system_prompt.prompt)
# the settings saved with this version: for the current prompt a dict with
# chunk_size, chunk_overlap, output_dims and k (note the key is "output_dims", not "output_dimensions")
print(system_prompt.config)
# the labels this version has, for example ["staging"]. "latest" may be listed too.
print(system_prompt.labels)

# More you can use from the prompt object:
#   system_prompt.name       -> the prompt name
#   system_prompt.compile()  -> the text with any {{variables}} filled in, e.g. compile(topic="RAG").
#                               This prompt has no variables, so it returns the same text as .prompt.
#   system_prompt.get_langchain_prompt() -> the text with {{var}} turned into {var}, for LangChain templates
#
# HOW THE APP WOULD USE IT (not part of this script; for example inside augmentation() in rag_workflow.py):
#   prompt_obj = langfuse.get_prompt("rag_app_system_prompt", label="production")
#   prompt = ChatPromptTemplate.from_messages([
#       ("system", prompt_obj.prompt),
#       ("human", "context: {context}\n\nquery: {query}")
#   ])
#   CAREFUL: ChatPromptTemplate treats {curly braces} in the text as placeholders. The current prompt has none.
#   If you ever add some (for example a JSON example), double them: {{ and }}, or the template raises an error.
#
# THE SAVED SETTINGS (config) AND THE APP:
#   k (how many chunks to retrieve) can be changed while the app runs: vs.as_retriever(search_kwargs={"k": prompt_obj.config["k"]}).
#   chunk_size, chunk_overlap and output_dims only take effect when the knowledge base is BUILT. Reading them from
#   here does not rebuild the saved-embeddings folder. After a change, delete that folder so rag_workflow.py
#   builds a new one (see the notes in rag_workflow.py).
#   Also note: the config stored now says chunk_overlap=30 and output_dims=1024, while rag_workflow.py uses 0 and 800.
#   Make the two match (see system_prompt_versioning.py).
#
# TO TIE AN EVALUATION RUN TO THIS PROMPT VERSION (in evaluate_rag_app.py), pass the version to evaluate(...), for example:
#   hyperparameters={"prompt_version": system_prompt.version, **system_prompt.config}
#
# Run this file with:  uv run python load_system_prompt.py