# =====================================================================
# WHAT THIS SCRIPT DOES
# It turns the goldens (test questions + expected answers) into TEST CASES by running
# real RAG app on every golden question and recording what the app answered.
#   golden     = question + expected answer + ideal context   (made by generate_goldens.py)
#   test case  = question + expected answer + ACTUAL answer + ACTUAL retrieved chunks
# The test cases are saved to a file, ready for the evaluation metrics.
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  OPENAI_API_KEY ready (the RAG app calls OpenAI)
#
#   config/parameter_config --> evaluation_dataset_params, golden_dataset_params
#
#   data/evaluation/goldens/<golden_dataset_filename>.json     (made by generate_goldens.py)
#      |  add_goldens_from_json_file
#      v
#   golden_dataset   (list of goldens: input, expected_output, context)
#      |
#      |  for each golden (one loop pass = one test case):
#      |     1. graph.invoke({"query": golden.input})   -> runs the RAG app:
#      |            retrieve -> augmentation -> generation
#      |     2. final_state gives:  response (the answer), retrieved_docs (the chunks used)
#      |     3. LLMTestCase(input, actual_output, expected_output, retrieval_context)
#      |     4. eval_dataset.add_test_case(...)
#      v
#   eval_dataset     (list of test cases)
#      |  save_as(json, include_test_cases=True)
#      v
#   data/evaluation/eval_dataset/<evaluation_dataset_filename>.json
#      |
#      v
#   (later) run metrics on the test cases (faithfulness, answer relevancy,
#           contextual precision / recall, ...)
#
# WHERE THIS FITS IN THE PROJECT
#   clean transcripts -> RAG app (notebook / app.rag_workflow) -> generate_goldens.py
#   -> THIS SCRIPT -> metrics
# =====================================================================

# EvaluationDataset = a container that holds goldens and/or test cases, and can load and save them
# (shorter import also works: from deepeval.dataset import EvaluationDataset)
from deepeval.dataset.dataset import EvaluationDataset
# LLMTestCase = one test case: what went in, what came out, what was expected
# (shorter import also works: from deepeval.test_case import LLMTestCase)
from deepeval.test_case.llm_test_case import LLMTestCase
# load_dotenv reads the .env file (OPENAI_API_KEY lives there)
from dotenv import load_dotenv
# Path gives a clean, cross-platform way to work with files and folders
from pathlib import Path
# logging pieces: getLogger makes a named logger, StreamHandler prints messages to the screen,
# Formatter controls how each message looks, INFO is the message level used below
from logging import getLogger, StreamHandler, Formatter, INFO
# the compiled LangGraph app from your project: the same graph you built in the notebook
# (retrieve -> augmentation -> generation). Importing it must not rebuild or re-add the vector store.
from rag_workflow import graph
# your own config object: it holds all the numbers/settings in one place (not part of DeepEval)
#from config.parameter_config import params_config

# load evaluation dataset params
# evaluation_dataset_params -> the "evaluation dataset" section of your config (output file name)
# golden_dataset_params     -> the "golden dataset" section (to find the goldens file that was saved earlier)
#evaluation_dataset_params = params_config.evaluation_dataset
#golden_dataset_params = params_config.golden_dataset

# load the api keys
# (arguments of load_dotenv: dotenv_path=".env" to point to a file, override=True to replace
#  existing variables, verbose=True to warn when no .env is found)
load_dotenv()

def generate_evaluation_dataset():

    # =================================================================
    # SETUP: logger and paths
    # =================================================================

    # create the logger
    # getLogger(name) -> a logger with this name. The same name always returns the SAME logger object,
    #                    so running this function twice in one session adds a second handler
    #                    and every message prints twice (restart the kernel, or check logger.handlers first).
    logger = getLogger(name="Dataset Logger")
    # add stream handler
    # A handler decides WHERE messages go. StreamHandler() prints to the screen (stderr by default).
    # Other choices: StreamHandler(sys.stdout) to print to stdout,
    #                FileHandler("dataset.log") to write to a file.
    handler = StreamHandler()
    logger.addHandler(handler)
    # setLevel -> the lowest level that is shown. Levels from low to high:
    #             DEBUG < INFO < WARNING < ERROR < CRITICAL. Messages below the level are hidden.
    logger.setLevel(INFO)
    # add formatter
    # fmt placeholders: %(asctime)s time, %(name)s logger name, %(levelname)s level, %(message)s the text.
    # Others you can use: %(filename)s, %(lineno)d. Extra argument: datefmt="%H:%M:%S" for a shorter time.
    formatter = Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt=formatter)

    # create paths
    # __file__ = this script's path. .resolve() makes it absolute.
    # .parent = go up one folder. Three .parent in a row means three folders up.
    # With the file at <root>/src/<folder>/generate_eval_dataset.py, three parents up is <root>.
    # If you move the file to a different depth, ROOT_DIR points to the wrong place.
    ROOT_DIR = Path(__file__).parent

    # the goldens file (the input), saved earlier by generate_goldens.py.
    # .with_suffix(".json") -> sets the extension to .json (it REPLACES any existing extension,
    # so a file name that contains a dot, like "golden_v1.0", would be cut to "golden_v1.json").
    GOLDENS_PATH = (ROOT_DIR / "datasets" / "golden.json")
    # the folder where the test cases file will be saved (the output)
    EVALUATION_DATA_DIR = ROOT_DIR / "datasets" / "eval_dataset"

    # create dir
    # exist_ok=True -> no error if it already exists; parents=True -> create missing parent folders too
    EVALUATION_DATA_DIR.mkdir(exist_ok=True, parents=True)


    # =================================================================
    # STEP 1: LOAD THE GOLDENS  (json file -> golden_dataset)
    # =================================================================

    # dataset to read goldens from
    # EvaluationDataset() with no arguments = an empty container.
    # Other ways to create it: EvaluationDataset(goldens=[...]) or EvaluationDataset(test_cases=[...]).
    golden_dataset = EvaluationDataset()
    # file_path -> the json file to read.
    # Extra arguments for json files with other key names (the defaults match DeepEval's own format):
    #   input_key_name="input", expected_output_key_name="expected_output", context_key_name="context"
    # Other loaders: add_goldens_from_csv_file(...) for csv files.
    golden_dataset.add_goldens_from_json_file(file_path=GOLDENS_PATH)


    # =================================================================
    # STEP 2: RUN THE RAG APP ON EVERY GOLDEN  (golden -> test case)
    # =================================================================

    # dataset to hold produced test cases
    eval_dataset = EvaluationDataset()

    # golden_dataset.goldens = the list of loaded goldens.
    # enumerate(list, 1) -> gives (count, item) pairs, counting from 1 (so the log says "no. 1", not "no. 0").
    for count, golden in enumerate(golden_dataset.goldens, 1):
        # Run the whole RAG graph once, with the golden question as the "query" input.
        # The key "query" must match the field name in RAGState from the notebook.
        # final_state is the full state dict: query, retrieved_docs, context, prompt, response.
        # (one call = one embedding call + one LLM call, and the loop runs one question at a time.
        #  For many goldens, graph.batch([{"query": ...}, ...], config={"max_concurrency": 5}) runs several in parallel.)
        final_state = graph.invoke({"query": golden.input})

        # Build one test case. Arguments:
        #   input             -> the question that was asked (from the golden)
        #   actual_output     -> what the RAG app answered (state key "response")
        #   expected_output   -> the ideal answer written when the golden was generated
        #   retrieval_context -> list of TEXT strings: the chunks the app really retrieved
        #                        (state key "retrieved_docs"; each Document's .page_content is its text)
        # Extra arguments you can add:
        #   context=golden.context   -> the IDEAL context from the golden (the ground-truth source text).
        #                               Do not mix it up with retrieval_context. Some metrics need it,
        #                               for example the hallucination metric.
        #   name="case 1"            -> a label for the test case
        #   additional_metadata={}   -> any extra info you want to keep with the case
        # Which fields a metric needs: answer relevancy -> input + actual_output;
        #   faithfulness -> actual_output + retrieval_context;
        #   contextual precision / recall -> input + actual_output + expected_output + retrieval_context.
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=final_state.get("response"),
            expected_output=golden.expected_output,
            retrieval_context=[doc.page_content for doc in final_state.get("retrieved_docs")]
        )
        # add the finished test case to the eval dataset
        eval_dataset.add_test_case(test_case=test_case)
        # progress message, printed through the logger set up above
        # (logger.log(level=INFO, msg=...) does the same as logger.info(...))
        logger.log(level=INFO, msg=f"Added test case no. {count}")


    # =================================================================
    # STEP 3: SAVE THE TEST CASES  (eval_dataset -> json file)
    # =================================================================

    # file_type          -> "json" (used here), "csv" or "jsonl"
    # directory          -> folder to save into.
    #                       (generate_goldens.py passes a string with .as_posix(); if this raises a TypeError
    #                        in your DeepEval version, use EVALUATION_DATA_DIR.as_posix() here too.)
    # file_name          -> name WITHOUT the extension; the extension comes from file_type.
    #                       If you leave it out, a timestamp is used.
    # include_test_cases -> True = save the TEST CASES. Default is False, which saves only the goldens.
    #                       This dataset holds test cases and no goldens, so with False the file would be EMPTY.
    eval_dataset.save_as(
        file_type="json",
        directory=EVALUATION_DATA_DIR,
        file_name="evaluation_dataset",
        include_test_cases=True
    )


# This block runs only when you execute this file directly (python generate_eval_dataset.py),
# not when another file imports it
if __name__ == "__main__":
    generate_evaluation_dataset()