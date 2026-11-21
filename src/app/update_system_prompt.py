# =====================================================================
# WHAT THIS SCRIPT DOES
# It MOVES a label (here "staging") onto a version of the saved system prompt in Langfuse.
# It does not change the prompt text. It only changes which version carries which label.
# This is the third script of the prompt set:
#   system_prompt_versioning.py -> SAVES a new prompt version (text + settings)
#   load_system_prompt.py       -> FETCHES a version by label (or version number)
#   update_system_prompt.py     -> MOVES labels between existing versions (this file)
#
# WHY MOVE LABELS?
#   Labels are pointers. The app asks for "the version labelled staging" or "labelled production"
#   and never needs to know the version number. To change which version is tested or live,
#   you move the label. No code change in the app is needed.
#
# HOW LABELS BEHAVE
#   - A label sits on ONE version of a prompt at a time. Giving it to another version takes it away
#     from the version that had it.
#   - Example, before and after update_prompt(version=3, new_labels=["staging"]):
#         before:   v1 [production]    v2 [staging]    v3 [latest]
#         after:    v1 [production]    v2 []           v3 [latest, staging]
#   - "latest" is reserved. Langfuse puts it on the newest version by itself, and you cannot assign it.
#   - The labels you pass are ADDED to the version. The version keeps its other labels.
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
#   STEP 2  langfuse.update_prompt(name, version, new_labels)
#              puts the label(s) in new_labels on that version of the prompt called `name`
#      |
#      v
#   (afterwards) load_system_prompt.py with label="staging" now returns that version
#
# THE SAME CALL IS HOW YOU DEPLOY AND ROLL BACK
#   test a version:      new_labels=["staging"]
#   go live:             new_labels=["production"]   (the live app gets it once its cache runs out, see below)
#   roll back:           give "production" to an older version number
#   Be careful with "production": it changes what the live app uses.
#
# WHERE THIS FITS IN THE PROJECT
#   create a version (staging) -> evaluate_rag_app.py scores it -> scores good? move "production" to it.
#   The Langfuse web page can do the same thing with buttons; this script is the same action in code.
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

# shifting the labels in dev

# get_client() -> returns the Langfuse client. It reads these environment variables:
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY -> from your Langfuse project settings
#   LANGFUSE_HOST                            -> the server address. EU cloud (default): https://cloud.langfuse.com,
#                                               US cloud: https://us.cloud.langfuse.com, or your own self-hosted address
# If the keys are missing or wrong, the next call fails with an authentication error.
langfuse = get_client()


# =====================================================================
# STEP 2: MOVE THE LABEL
# =====================================================================

# update_prompt(...) -> puts labels on one version of a prompt. Arguments:
#   name       -> the prompt's name, exactly as it was created ("rag_app_system_prompt").
#                 If the name does not exist, the call raises an error.
#   version    -> the version NUMBER to change, a whole number such as 1, 2 or 3.
#                 It is required. See the warning below about version=None.
#   new_labels -> list of labels to put on that version, for example ["staging"] or ["production"].
#                 Each label moves away from the version that had it. "latest" cannot be assi
# gned.
# What happens around the call:
#   - It changes Langfuse itself, not only your computer, so every app that reads this prompt is affected.
#   - It clears the prompt cache of THIS Python process. Other running apps keep their cached copy until
#     it runs out (cache_ttl_seconds, 60 seconds by default, see load_system_prompt.py).
#   - It returns the updated prompt. This script does not use it, so nothing is printed.
#
# WARNING: version=None is probably an error.
#   update_prompt needs a version number. With None, the request is built for a version called "None",
#   and Langfuse rejects it (in the SDK versions I know). The code below is kept exactly as you gave it.
#   If you get an error, use this instead. It finds the newest version and puts "staging" on it:
#
#       latest_version = langfuse.get_prompt(name="rag_app_system_prompt", type="text",
#                                            label="latest", cache_ttl_seconds=0).version
#       updated_prompt = langfuse.update_prompt(name="rag_app_system_prompt",
#                                               version=latest_version,
#                                               new_labels=["staging"])
#       print(updated_prompt.version, updated_prompt.labels)
#
#   Or write the number yourself, for example version=2 to put "staging" back on version 2.
#   (cache_ttl_seconds=0 makes sure get_prompt asks Langfuse for the newest version, not a cached one.)
langfuse.update_prompt(
    name="rag_app_system_prompt",
    version=None,
    new_labels=["staging"]
)

# To check the result, run load_system_prompt.py with label="staging",
# or open the prompt in the Langfuse web page and look at the labels of each version.
#
# Run this file with:  uv run python update_system_prompt.py