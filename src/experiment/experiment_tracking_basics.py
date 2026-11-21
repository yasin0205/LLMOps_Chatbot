# =====================================================================
# WHAT THIS SCRIPT DOES
# It is a small demo of EXPERIMENT TRACKING with MLflow. It records one "run": the settings you used
# (params), the scores you got (metrics), and the files that belong to it (code and dataset).
# Why? When you change chunk_size, the prompt or the model, you want to compare the scores of all your tries
# in one place instead of remembering them. MLflow keeps the history and shows it in a web page.
#
# THE MLFLOW WORDS
#   Experiment -> a group of runs for one project ("campusx_rag_app")
#   Run        -> one try: one set of settings and the scores it got
#   Param      -> an INPUT setting of the run (chunk_size, chunk_overlap, ...)
#   Metric     -> a number RESULT of the run (recall, faithfulness, ...)
#   Artifact   -> a FILE saved with the run (code, dataset, reports, plots)
#
# FLOW OF THE WHOLE SYSTEM
#
#   STEP 0  start the MLflow server in a separate terminal and keep it running:
#              mlflow server --host 127.0.0.1 --port 5000        (or:  mlflow ui --port 5000)
#              Start it from your project root, so its data folders (mlruns, mlartifacts) land there.
#      |
#      v
#   STEP 1  mlflow.set_tracking_uri(...)    tells this script WHERE to send the data (the server address)
#      |
#      v
#   STEP 2  mlflow.set_experiment(...)      picks the experiment (it is created if it does not exist yet)
#      |
#      v
#   STEP 3  with mlflow.start_run(...)      opens one run. Everything inside the block belongs to this run:
#              log_param     -> the settings
#              log_metric    -> the scores
#              log_artifact  -> the files (code, dataset)
#           when the block ends, the run is closed by itself
#      |
#      v
#   STEP 4  open http://127.0.0.1:5000 in the browser to see the run, and to compare runs
#           (table, charts, sort by a metric)
#
# WHERE THIS FITS IN THE PROJECT
#   Change a setting (chunk_size, prompt version, ...) -> rebuild the app -> evaluate_rag_app.py computes the scores
#   -> log settings + scores + code + dataset as ONE run here -> compare runs and keep the best one.
#   In this demo the metric values are typed by hand (0.89, 0.91, 0.7). In the real pipeline you would
#   take them from the evaluation results (for example the average of each metric over all test cases).
# =====================================================================

# mlflow = the experiment tracking library
import mlflow
# Path gives a clean, cross-platform way to work with files and folders
from pathlib import Path

# __file__ = this script's path. .resolve() makes it absolute.
# .parent = go up one folder. Three .parent in a row means three folders up.
# With the file at <root>/src/<folder>/experiment_tracking_basics.py, three parents up is <root>.
# If you move the file to a different depth, ROOT_DIR points to the wrong place.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# =====================================================================
# STEP 1: SET THE TRACKING SERVER
# =====================================================================

# set the tracking server
# set_tracking_uri(uri) -> where MLflow sends the data. The server from Step 0 must be running at this address,
#                          or the calls below fail with connection errors.
# Choices:
#   "http://127.0.0.1:5000/"      -> an MLflow server on your own computer (used here)
#   "http://<host>:5000"          -> a server on another machine, shared with a team
#   "file:./mlruns"               -> no server: write straight into a local folder (this is the default if you set nothing)
#   "sqlite:///mlflow.db"         -> no server: store in a local database file
# You can also set it once for all scripts with the environment variable MLFLOW_TRACKING_URI.
mlflow.set_tracking_uri("http://127.0.0.1:5000/")


# =====================================================================
# STEP 2: SET THE EXPERIMENT
# =====================================================================

# set the experiment name
# set_experiment(name) -> makes this the active experiment. If it does not exist, it is created.
#                         All runs started afterwards go into it.
# Use one experiment per project or per question you are studying (for example "campusx_rag_app").
# (set_experiment(experiment_id="...") picks an existing experiment by its id instead.)
mlflow.set_experiment("campusx_rag_app")


# =====================================================================
# STEP 3: START A RUN AND LOG EVERYTHING
# =====================================================================

# start a run
# start_run(...) used as `with ... as run` -> the run opens at the start of the block and closes at the end.
# If an error happens inside the block, the run is closed and marked as FAILED.
# Arguments used here:
#   run_name -> the name shown in the web page. It does not have to be unique (every run also has its own run id).
#               A name that says what changed, like "chunk300_overlap30", makes runs easier to tell apart than "test_run".
# Extra arguments you can add:
#   tags={"prompt_version": "2"}   -> labels for searching and filtering runs in the web page
#   description="shorter chunks"   -> a note for the run
#   run_id="..."                   -> continue logging into an existing run
#   nested=True                    -> a child run inside another run
#   log_system_metrics=True        -> also record CPU, memory and GPU use during the run
# `run` holds the run's details: run.info.run_id is the id, run.info.run_name is the name.
with mlflow.start_run(run_name="test_run") as run:
    # params
    # log_param(key, value) -> saves one INPUT setting. Values are stored as text.
    # A param is fixed for the run: logging the same key again with a different value raises an error.
    # Use log_params({...}) to save many at once, for example:
    #   mlflow.log_params({"chunk_size": 300, "chunk_overlap": 30, "output_dims": 1024, "k": 3})
    # NOTE: these numbers are typed here by hand. Keep them the same as the values the app really uses
    # (rag_workflow.py now has chunk_overlap=0 and dimensions=800; the Langfuse prompt config has 30 and 1024).
    # Other settings worth logging: k, the LLM and embedding model names, the judge model, and the prompt version.
    mlflow.log_param("chunk_size", 300)
    mlflow.log_param("chunk_overlap", 30)
    mlflow.log_param("output_dims", 1024)
    # metrics
    # log_metric(key, value) -> saves one number RESULT. The value must be a number.
    # Extra argument: step=0, 1, 2, ... -> log the same metric many times to see it change (for example per epoch).
    # Use log_metrics({...}) to save many at once.
    # NOTE: the values here are examples, not computed scores.
    # NAMES: use the same metric names every time, or runs cannot be compared. DeepEval's custom metric is
    # called "answer correctness" (with a space), and here it is logged as "answer_correctness" (with an underscore).
    # Pick one spelling and always convert to it.
    mlflow.log_metric("recall", 0.89)
    mlflow.log_metric("faithfulness", 0.91)
    mlflow.log_metric("answer_correctness", 0.7)
    # log artifacts
    # log_artifact(local_path, artifact_path) -> copies ONE file into the run's artifact storage.
    # local_path    -> the file to copy. Here the app's code, saved exactly as it was in this run.
    # artifact_path -> a sub-folder inside the run's artifacts ("code"). Leave it out to put the file at the top.
    # Other ways to save files and data:
    #   mlflow.log_artifacts(local_dir=..., artifact_path=...)  -> a whole folder
    #   mlflow.log_dict({...}, "results.json")                  -> a Python dict as a json file
    #   mlflow.log_text("...", "notes.txt")                     -> a string as a text file
    mlflow.log_artifact(local_path=(ROOT_DIR / "src" / "app" / "rag_workflow.py"),
                        artifact_path="code")
    # log dataset
    # Saves the goldens file as an artifact (a copy of the file, at the top of the run's artifacts).
    # NOTE: the file name "golden_dataset_final.json" is typed here. generate_goldens.py takes its file name from your
    # config (golden_dataset_filename). If the two differ, this line fails with a "file not found" error.
    # Another way is to log it as a tracked dataset (mlflow.data + mlflow.log_input), which also records
    # where the data came from and shows it in the run's Datasets section.
    mlflow.log_artifact(local_path=(ROOT_DIR / "data" / "evaluation" / "goldens" / "golden.json"))
    
# After the block the run is closed. Nothing is printed by this script. To see the run, open
# http://127.0.0.1:5000 and click the experiment "campusx_rag_app". To print its id, add inside the block:
#   print(run.info.run_id)
#
# Run this file with:  uv run python experiment_tracking_basics.py
# (the server from Step 0 must already be running in another terminal)