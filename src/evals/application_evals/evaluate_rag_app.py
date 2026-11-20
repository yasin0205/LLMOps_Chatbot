# =====================================================================
# WHAT THIS SCRIPT DOES
# It is the FINAL STEP of the project: it scores your RAG app.
# It loads the saved test cases (question, expected answer, actual answer, retrieved chunks),
# runs 7 metrics on every test case with a judge LLM, and writes a report.
# This is the script that runs all the metric demos (recall, precision, relevancy, faithfulness,
# answer relevancy, answer correctness, simple explanation) together on the real data.
#
# WHERE THIS FITS IN THE WHOLE PROJECT
#
#   transcripts/*.txt --(transcript_utils.py)--> data/processed/*.txt   (cleaned documents)
#        |
#        v
#   RAG app (notebook / app.rag_workflow): load -> chunk -> embed -> Chroma -> retrieve -> augmentation -> generation
#        |
#        v
#   generate_goldens.py        -> data/evaluation/goldens/<golden_dataset_filename>.json
#        |                        (test questions + expected answers + ideal contexts)
#        v
#   generate_eval_dataset.py   -> data/evaluation/eval_dataset/<evaluation_dataset_filename>.json
#        |                        (runs the RAG app on every golden: adds actual_output + retrieval_context)
#        v
#   evaluate_rag_app.py (THIS FILE) -> scores + reports in <root>/reports/...
#
# FLOW OF THIS SCRIPT
#
#   .env  --load_dotenv()-->  OPENAI_API_KEY ready (the judge LLM is an OpenAI model)
#
#   config/parameter_config --> evaluation_params (judge model, async settings, report folders),
#                               evaluation_dataset_params (the test cases file name)
#
#   STEP 1  create 5 built-in metrics        (3 for the retriever, 2 for the answer)
#   STEP 2  create 2 custom G-Eval metrics   (answer correctness, simple explanation)
#      |
#      v
#   STEP 3  evaluate_app():
#              a. build the path to the saved test cases json
#              b. if the file exists, load the test cases into an EvaluationDataset
#              c. evaluate(test_cases, metrics=[7 metrics]) runs every metric on every test case
#                 (in parallel, controlled by AsyncConfig)
#              d. the results are shown and saved as a report (controlled by DisplayConfig)
#      |
#      v
#   reports/<results_dir> and reports/<report_dir>
#
# WHICH METRIC CHECKS WHAT, AND WHAT TO FIX WHEN IT IS LOW
#   recall                low -> the retriever misses needed info   -> larger k, other chunk_size/overlap, better embeddings
#   precision             low -> good chunks ranked too low         -> reranker, better embeddings, hybrid search
#   contextual_relevancy  low -> the retrieved text is noisy        -> smaller k or chunks, score threshold on the retriever
#   faithfulness          low -> the answer adds/contradicts facts  -> stricter system prompt, stronger LLM
#   answer_relevancy      low -> the answer rambles off topic       -> tighter system prompt, less noisy context
#   answer_correctness    low -> the answer differs from the ideal  -> check the metrics above to find where it breaks
#   simple_explanation    low -> the answer is too technical        -> add "explain simply, avoid jargon" to the system prompt
#
# COST AND TIME
#   Every metric makes several LLM calls per test case (roughly 15 to 25 calls for all 7 metrics together).
#   Total calls = number of test cases x that number, so a big dataset is slow and costs money.
#   Try 2 or 3 test cases first (for example test_cases = dataset.test_cases[:3]) before the full run.
# =====================================================================

# The metric classes:
#   GEval                      -> custom metric where YOU write the rule (used for the 2 custom metrics below)
#   AnswerRelevancyMetric      -> is the answer on topic?
#   FaithfulnessMetric         -> is the answer supported by the retrieved chunks?
#   ContextualPrecisionMetric  -> are the useful chunks ranked first?
#   ContextualRecallMetric     -> were all the needed facts retrieved?
#   ContextualRelevancyMetric  -> how much of the retrieved text is relevant to the question?
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
# evaluate = the function that runs a list of metrics on a list of test cases
from deepeval.evaluate import evaluate
# Rubric = one band of a G-Eval scoring guide: a score range plus a description of what it means
from deepeval.metrics.g_eval import Rubric
# SingleTurnParams = names of the test case fields a G-Eval judge can be shown (INPUT, ACTUAL_OUTPUT, ...)
# NOTE: in older DeepEval versions this enum is called LLMTestCaseParams
from deepeval.test_case.llm_test_case import SingleTurnParams
# EvaluationDataset = a container that holds test cases and can load them from a file
from deepeval.dataset.dataset import EvaluationDataset
# AsyncConfig = settings for running in parallel; DisplayConfig = settings for showing and saving results
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
# Path gives a clean, cross-platform way to work with files and folders
from pathlib import Path
# load_dotenv reads the .env file (OPENAI_API_KEY lives there)
from dotenv import load_dotenv
# your own config object: it holds all the numbers/settings in one place (not part of DeepEval)
#from config.parameter_config import params_config

# load the evaluation params
# NOTE: the variable below is called evaluation_params, and GEval also has an argument called
# evaluation_params=[...]. They are two different things: the variable is your config section,
# the argument is the list of test case fields the judge may see. Python keeps them apart, so it works.
#evaluation_params = params_config.evaluation
# the async settings (parallel runs) and the display settings (report folders) from your config
#async_params = evaluation_params.async_config
#display_params = evaluation_params.display_config
# the "evaluation dataset" section of your config (the test cases file name)
#evaluation_dataset_params = params_config.evaluation_dataset

# load the api keys
# (arguments of load_dotenv: dotenv_path=".env" to point to a file, override=True to replace
#  existing variables, verbose=True to warn when no .env is found)
load_dotenv()

# the judge LLM, read from your config (for example a name like "gpt-4.1").
# The SAME judge is passed to every metric below, so all scores come from one consistent judge.
# A stronger judge gives more reliable scores; a cheaper one saves money. It can also be your own
# class based on DeepEvalBaseLLM (for other providers).
model = "gpt-5-nano"


# =====================================================================
# STEP 1: THE BUILT-IN METRICS
# =====================================================================

# define the metrics
# Each metric gets only model=model here, so every other setting stays at its default.
# Defaults and extra arguments you can add (the same for all five):
#   threshold=0.5       -> pass mark between 0 and 1; success is True when score >= threshold
#   include_reason=True -> also write a short explanation of the score (False = cheaper and faster)
#   strict_mode=False   -> True makes the score all-or-nothing (1 or 0, threshold forced to 1)
#   async_mode=True     -> run the judge calls in parallel
#   verbose_mode=False  -> True prints every step the judge takes
# Extra arguments for FaithfulnessMetric only:
#   penalize_ambiguous_claims=False -> True also counts facts that are not in the context against the answer
#   truths_extraction_limit=None    -> cap on how many facts are taken from the retrieved chunks

# --- the retriever metrics (they judge the chunks) ---
# recall: did the retrieved chunks contain everything needed for the expected answer?
recall = ContextualRecallMetric(model=model)
# precision: are the useful chunks ranked above the useless ones?
precision = ContextualPrecisionMetric(model=model)
# contextual_relevancy: what share of the retrieved text is relevant to the question?
contextual_relevancy = ContextualRelevancyMetric(model=model)

# --- the answer metrics (they judge the LLM's answer) ---
# answer_relevancy: does the answer address the question, without off-topic sentences?
answer_relevancy = AnswerRelevancyMetric(model=model)
# faithfulness: is the answer supported by the retrieved chunks?
faithfulness = FaithfulnessMetric(model=model)


# =====================================================================
# STEP 2: THE CUSTOM METRICS (G-Eval)
# =====================================================================

# define the custom metrics

# ---- answer_correctness ----
# Question: "Is the answer correct?" It compares the actual answer with the expected answer, but it is
# written to be FAIR to the retriever. Goldens are generated by an LLM, so the expected answer can contain
# facts that the retriever never returned. The app cannot be blamed for missing those.
# That is why the judge also sees RETRIEVAL_CONTEXT, and the steps use it as the standard.
#
# evaluation_params -> the judge sees 4 fields: input, expected_output, actual_output, retrieval_context
# evaluation_steps  -> in plain words, the judge:
#   1. finds the PRIMARY thing the question asks for
#   2. builds the list of "key facts": claims in the expected answer that answer the question AND can be
#      found in the retrieval context. A claim with no support in the context is NOT a key fact, so missing
#      it (even saying "I don't know") is never penalised
#   3. lets the retrieval context win when it disagrees with the expected answer on the main point
#   4. checks each key fact: conveyed, contradicted, or omitted
#   5. judges contradictions against both the expected answer and the retrieval context
#   6. penalises extra claims only if they contradict the context or the expected answer, or are clearly false
#   7. ignores differences in wording, order, format and length
#   8. ranks problems by severity (contradiction > missing primary fact > minor secondary detail)
#   9. picks the rubric band for the most severe real problem, and the top of the band if the rest is accurate
# rubric            -> 4 bands (whole numbers 0 to 10, no overlap), converted to 0 to 1 by dividing by 10:
#   0 to 3  = the main claim is missing or contradicted, or key facts are wrong
#   4 to 6  = what it says is correct, but it misses a key fact for the primary question, or adds a claim that contradicts the context
#   7 to 8  = primary answer correct, at most one small secondary problem
#   9 to 10 = every key fact correct, no contradiction
# model             -> the same judge LLM as the other metrics
# Extra arguments you can add: threshold=0.5, strict_mode, async_mode, verbose_mode, top_logprobs
#   (explained in answer_correctness_metric.py). NOTE: with the default threshold 0.5, scores of 5 and 6
#   (the "misses a key fact" band) would still PASS. Use threshold=0.7 so that only 7 and above passes,
#   or set the pass marks where your project keeps them (the comment below mentions thresholds.json).
answer_correctness = GEval(
    name="answer correctness",  # unchanged — feeds MLflow / thresholds.json keys
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    evaluation_steps=[
        "Identify the PRIMARY thing 'Input' asks for (its central question), separately "
        "from any secondary elaboration it invites.",
        "Build the list of 'key facts': claims in 'Expected Output' that answer 'Input' AND "
        "for which you can point to a specific supporting sentence in 'Retrieval Context'. "
        "If you cannot locate a supporting sentence in 'Retrieval Context' for a claim in "
        "'Expected Output', it is NOT a key fact: the system could not have known it, so "
        "its absence from 'Actual Output' — including an explicit 'I don't know' or 'the "
        "context does not say' about it — is never an omission or an error and must not "
        "lower the score.",
        "Pick the rubric band for the most severe real issue found; within that band, "
        "score at the top when the rest of the answer is accurate.",
    ],
    rubric=[
        Rubric(score_range=(0, 3), expected_outcome="The main claim is absent or contradicted, OR 'Actual Output' contradicts a key fact (one you can point to in 'Retrieval Context') on a topic 'Input' directly asks about, OR several key facts are wrong."),
        Rubric(score_range=(4, 6), expected_outcome="Every key fact 'Actual Output' addresses is correct, but it fails to convey a key fact that answers the PRIMARY question of 'Input' and that 'Retrieval Context' supports, OR an added claim directly contradicts 'Retrieval Context'."),
        Rubric(score_range=(7, 8), expected_outcome="Every key fact answering the primary question is conveyed correctly and nothing contradicts 'Retrieval Context'; at most one secondary retrieval-supported detail is missing or slightly imprecise, OR one statement is faithful to 'Retrieval Context' but conflicts with an unsupported generalisation in 'Expected Output'."),
        Rubric(score_range=(9, 10), expected_outcome="Every key fact that 'Input' primarily asks for and that 'Retrieval Context' supports is conveyed correctly, with no contradiction of 'Expected Output' or 'Retrieval Context' and no clearly false statement. Wording, length, format, and any amount of extra grounded or plausible-uncontradicted detail may differ freely. A claim in 'Expected Output' with no locatable support in 'Retrieval Context' is not required; its absence, or an explicit 'not covered' about it, does not lower the score. Loosely-grounded-but-uncontradicted additions also stay in this band."),
    ],
    model=model
)

# ---- simple_explanation ----
# Question: "Is the answer easy to follow for an LLMOps student?" It judges readability only, not correctness.
#
# evaluation_params -> the judge sees 2 fields: input and actual_output
# evaluation_steps  -> in plain words, the judge:
#   1. reads the question (and which terms the asker already uses), then reads the answer as an LLMOps student
#      (comfortable with everyday AI and software words, new to the details)
#   2. finds the OPAQUE jargon: terms such a student would not understand. These do NOT count: terms already
#      used in the question, ordinary English phrases, standard AI/RAG words the audience knows
#      (model, prompt, retriever, embedding ...), and product names
#   3. checks whether each opaque term is avoided, replaced, or explained the first time it appears
#   4. judges structure: short sentences, headings, bullets, examples help; long dense paragraphs hurt
#   5. judges readability only, but does not reward simplicity that comes from leaving out part of the answer
#   6. decides how much effort the student needs, and picks the matching band
# rubric            -> 4 bands (whole numbers 0 to 10, no overlap), converted to 0 to 1 by dividing by 10:
#   0 to 2  = dense with unexplained jargon; the student cannot follow it
#   3 to 5  = three or more unexplained terms, or tangled structure
#   6 to 8  = one essential term left unexplained, or dense and hard to scan; needs real extra effort
#   9 to 10 = well organised, with an example or clear breakdown where it helps; the student can follow it fully
# Extra arguments you can add: threshold=0.5 and the others. NOTE: with the default 0.5, scores 6 to 8
# ("needs real extra effort") would still PASS. Use threshold=0.9 if only the top band should pass.
simple_explanation = GEval(
    name="simple explanation",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Read 'Input' to see what was asked and which terms the asker already uses. Read "
        "'Actual Output' as an LLMOps student: someone taking a course on building and "
        "operating LLM applications, comfortable with everyday AI/software vocabulary but "
        "new to fine detail.",
        "Decide how much effort an LLMOps student needs to follow 'Actual Output' from "
        "start to finish, and pick the matching band.",
    ],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Dense with unhandled opaque jargon or convoluted sentences; the student cannot follow it."),
        Rubric(score_range=(3, 5), expected_outcome="Three or more pieces of opaque jargon are left unhandled, or the structure is tangled; the student follows only parts."),
        Rubric(score_range=(6, 8), expected_outcome="At least one opaque term that is essential to the main point is left unhandled AND this genuinely impedes understanding, or the structure is dense and hard to scan; the student needs real extra effort."),
        Rubric(score_range=(9, 10), expected_outcome="The answer is well organised (short sentences, headings/bullets, at least one concrete example or clear breakdown where it helps) and any unhandled terms are either non-opaque to this audience or inessential to the main point. An LLMOps student can follow it from start to finish. Reusing the Input's own terminology, ordinary domain-flavoured phrasing, standard AI/RAG vocabulary, and product names is expected and fine."),
    ],
    model=model
)


# =====================================================================
# STEP 3: LOAD THE TEST CASES AND RUN THE EVALUATION
# =====================================================================

def evaluate_app():

    # define the dataset path
    # __file__ = this script's path. .resolve() makes it absolute.
    # .parent = go up one folder. FOUR .parent in a row here (the other scripts use three),
    # so this file sits one folder deeper than they do. ROOT_DIR must end up at your project root,
    # because both the input (data/...) and the output (reports/...) are built from it.
    # If a folder is created in the wrong place, check this number of .parent.
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
    # the test cases file (the input), saved earlier by generate_eval_dataset.py.
    # .with_suffix(".json") -> sets the extension to .json (it REPLACES any existing extension)
    DATASET_PATH = (ROOT_DIR / "data" / "evaluation" / "eval_dataset" / "evaluation_dataset.json")

    # Only run if the file exists.
    # NOTE: there is no else branch, so a missing file means the script ends WITHOUT any message.
    # An easy improvement: add  else: print(f"File not found: {DATASET_PATH}")
    if DATASET_PATH.exists():
        # load the dataset
        # EvaluationDataset() with no arguments = an empty container
        dataset = EvaluationDataset()
        
        # load the test cases
        # file_path                  -> the json file to read
        # the *_key_name arguments   -> the names of the keys inside the json file. They must match the file.
        #                               These are DeepEval's own names, so they match what
        #                               generate_eval_dataset.py saved (and they are also the defaults).
        # Other key names you can set: context_key_name (the ideal context), tools_called_key_name, ...
        # Other loaders: add_test_cases_from_csv_file(...) for csv files.
        dataset.add_test_cases_from_json_file(
            file_path=DATASET_PATH,
            input_key_name="input",
            actual_output_key_name="actual_output",
            expected_output_key_name="expected_output",
            retrieval_context_key_name="retrieval_context"
        )
        
        # store the test cases in a list
        # dataset.test_cases = the list of loaded LLMTestCase objects.
        # (to try a small run first: test_cases = dataset.test_cases[:3])
        test_cases = dataset.test_cases
        
        
        # evaluate the dataset
        # evaluate(...) runs every metric on every test case, then shows and saves the results.
        # Arguments used here:
        #   test_cases -> the list of test cases to score
        #   metrics    -> the list of metrics to run on EACH test case (7 here: 3 retriever, 2 answer, 2 custom).
        #                 Every test case must hold the fields the metrics need: input, actual_output,
        #                 expected_output and retrieval_context are all there, because
        #                 generate_eval_dataset.py saved them. To run fewer metrics, remove them from the list.
        #   async_config   -> how to run in parallel:
        #       throttle_value -> seconds to wait between starting test cases (default 0).
        #                         Raise it if you hit OpenAI rate limit errors.
        #       max_concurrent -> how many test cases run at the same time (default 20).
        #                         Lower = slower but gentler on rate limits; higher = faster.
        #       run_async=True -> (extra, default True) False runs one test case at a time, which is slow
        #                         but easier to debug.
        #   display_config -> how results are shown and saved (the exact options depend on your DeepEval
        #                     version; run help(DisplayConfig) to see them all):
        #       results_folder  -> folder where this run's results are saved
        #                          (here <root>/reports/<results_dir>; .as_posix() gives DeepEval a string path)
        #       file_type="md"  -> the format of the saved report file: Markdown
        #       file_output_dir -> folder of the report file (here <root>/reports/<report_dir>)
        #       other options that exist in many versions: show_indicator=False (hide progress bars),
        #                          print_results=False (do not print the results table),
        #                          verbose_mode=True (print every metric's steps),
        #                          display_option to show only all / failing / passing results
        # Extra arguments of evaluate() you can add:
        #   identifier="run 1"                -> a label for this run, so you can tell runs apart
        #   hyperparameters={"k": 3, "chunk_size": 1024, "model": "gpt-5-mini"}
        #                                     -> save the app settings together with the scores,
        #                                        so you can compare runs after changing the app
        #   error_config=ErrorConfig(ignore_errors=True)   -> keep going when one metric fails
        #                                        (from deepeval.evaluate.configs import ErrorConfig)
        #   cache_config=CacheConfig(write_cache=True, use_cache=True) -> reuse earlier results
        #                                        (from deepeval.evaluate.configs import CacheConfig)
        # evaluate() also RETURNS the results object. To use it in code, write: result = evaluate(...)
        evaluate(test_cases=test_cases,
                metrics=[recall,
                        precision,
                        answer_relevancy,
                        faithfulness,
                        contextual_relevancy,
                        answer_correctness,
                        simple_explanation],
                async_config=AsyncConfig(),
                display_config=DisplayConfig(results_folder=(ROOT_DIR / "reports" / "evaluation_results").as_posix(),
                                            file_type="md",
                                            file_output_dir=(ROOT_DIR / "reports" / "evaluation_reports").as_posix())
                                            
        )


# This block runs only when you execute this file directly (python evaluate_rag_app.py),
# not when another file imports it
if __name__ == "__main__":
    evaluate_app()