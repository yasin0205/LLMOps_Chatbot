# =====================================================================
# WHAT THIS SCRIPT DOES
# It builds a CUSTOM metric called "simple explanation" with G-Eval, and tests it on one test case.
# Question it answers: "Is the app's answer simple and student friendly,
# or is it too technical and hard to follow?"
# It judges the STYLE of the answer (how easy it is to read). It does not check if the answer is correct.
# A wrong answer that is easy to read would still score high.
#
# HOW IT DIFFERS FROM THE ANSWER CORRECTNESS METRIC (the other G-Eval file)
#   Answer correctness  -> criteria in one text block; compares actual_output with expected_output
#   Simple explanation  -> evaluation_steps (your own checklist); needs only input + actual_output,
#                          so there is no ideal answer to prepare
#   Both are built with the same GEval class. Only the rule and the fields the judge sees change.
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  OPENAI_API_KEY ready (the metric uses an LLM as a judge)
#
#   STEP 1  define the metric: GEval(name, evaluation_params, evaluation_steps, rubric, ...)
#              evaluation_params  which test case fields the judge may see (input, actual_output)
#              evaluation_steps   your own checklist of what the judge must do, in order
#              rubric             the scoring guide: which score range means what
#      |
#      v
#   STEP 2  LLMTestCase(...)   one test case: the question (input) + the app's answer (actual_output)
#      |
#      v
#   STEP 3  simple_explanation.measure(test_case)   the judge LLM works in these steps:
#              a. use YOUR evaluation_steps as they are
#                 (with criteria instead, the judge would first write its own steps from it)
#              b. read input and actual_output, follow the steps, and pick a score from 0 to 10
#                 inside the rubric range that fits best
#              c. DeepEval converts it to 0 to 1 (score / 10); when the judge model supports it, the score
#                 is also averaged over likely alternatives (token probabilities) to make it more stable
#              d. write a short reason, and compare the score with the threshold
#      |
#      v
#   measure() returns the score, and the results stay in simple_explanation.score, .reason, .success
#   (verbose_mode=True also prints every step)
#
# WHERE THIS FITS IN THE PROJECT
#   goldens (generate_goldens.py) -> test cases from the RAG app (generate_eval_dataset.py) -> metrics.
#   In the real pipeline input is the golden question and actual_output is final_state["response"].
#   Here both are typed by hand to see how the metric behaves.
#   This metric checks a STYLE rule of your system prompt. If scores are low, add a line to the system
#   prompt in the notebook, for example "Explain in simple words for a student and avoid jargon".
#
# WHAT HAPPENS WITH THE EXAMPLE BELOW
#   The answer is correct, but it uses terms a beginner may not know: "supervised machine learning",
#   "dependent (target) variable", "independent (input) variables". It does explain some of them
#   in brackets, and the idea of "the best possible straight line" is easy to picture.
#   So the judge should call it somewhere between "slightly difficult" and "moderately simple":
#   about 0.6 to 0.8 after conversion. The judge is an LLM, so the number can vary between runs.
# =====================================================================

# GEval = the metric class where YOU write the evaluation rule (G-Eval method)
from deepeval.metrics import GEval
# LLMTestCase = one test case: what went in, what came out, what was expected
from deepeval.test_case import LLMTestCase
# Rubric = one band of the scoring guide: a score range plus a description of what it means
from deepeval.metrics.g_eval import Rubric
# SingleTurnParams = names of the test case fields the judge can be shown (INPUT, ACTUAL_OUTPUT, ...)
# NOTE: in older DeepEval versions this enum is called LLMTestCaseParams
# (from deepeval.test_case import LLMTestCaseParams). If you get an ImportError here, use that name.
from deepeval.test_case.llm_test_case import SingleTurnParams
# load_dotenv reads the .env file (OPENAI_API_KEY lives there)
from dotenv import load_dotenv

# load the api key
# (arguments of load_dotenv: dotenv_path=".env" to point to a file, override=True to replace
#  existing variables, verbose=True to warn when no .env is found)
load_dotenv()


# =====================================================================
# STEP 1: DEFINE THE CUSTOM METRIC
# =====================================================================

# define the custom criteria
# Arguments used here:
#   name              -> a label for the metric; it shows up in logs and result tables
#   evaluation_params -> the list of test case fields the judge is allowed to see.
#                        Choices: SingleTurnParams.INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT, CONTEXT,
#                        RETRIEVAL_CONTEXT, TOOLS_CALLED, EXPECTED_TOOLS.
#                        Here the question and the answer are listed, because the steps below mention
#                        both. No expected_output is listed, so the test case does not need one.
#   evaluation_steps  -> your own checklist, a list of short instructions the judge follows in order.
#                        This gives you full control over what is checked and more consistent scores
#                        than criteria. Use EITHER evaluation_steps OR criteria, not both.
#                        Tips: refer to the fields by name, and make each step something you can check.
#                        "Least number of technical jargons" is a bit vague; a clearer step would be
#                        "Count the technical terms a beginner would not know; fewer terms means a higher score".
#   rubric            -> the scoring guide: a list of Rubric(score_range=(low, high), expected_outcome="...").
#                        Ranges are whole numbers from 0 to 10, and they must not overlap.
#                        The judge picks the band that fits and gives a score inside it.
#                        Here: 0 to 3 = too difficult, 4 to 7 = slightly difficult,
#                        8 to 9 = moderately simple, 10 = very simple.
#                        Without a rubric the judge scores freely from 0 to 10.
#   verbose_mode      -> True prints every step the judge takes. Default False.
# Extra arguments you can add:
#   criteria="..."    -> a plain-English description instead of evaluation_steps (the judge then writes its own steps)
#   threshold=0.5     -> pass mark on the converted 0 to 1 score. Default 0.5. NOTE: with this rubric a score
#                        of 4 to 7 means "slightly difficult" (0.4 to 0.7), and 5, 6 and 7 would still PASS at 0.5.
#                        Use threshold=0.8 so that only "moderately simple" and "very simple" pass.
#   model="gpt-4.1"   -> the judge LLM. A model name string, or your own class based on DeepEvalBaseLLM.
#                        The probability-weighted scoring only works with models that return token
#                        probabilities (OpenAI models do); others still work, with a less smooth score.
#   strict_mode=False -> True makes the score all-or-nothing (1 or 0, threshold forced to 1)
#   async_mode=True   -> run the judge calls in parallel (faster). Default True.
#   top_logprobs=20   -> how many alternative scores are used for the probability weighting
simple_explanation = GEval(
    name="simple explanation",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    evaluation_steps=[
        "Read the 'actual output' first and then the 'input'",
        "Check whether the response is simple and easy to understand or not",
        "Make sure the response has least number of technical jargons and is student friendly"
    ],
    rubric=[
        Rubric(score_range=(0,3), expected_outcome="Too Difficult"),
        Rubric(score_range=(4,7), expected_outcome="Slightly difficult"),
        Rubric(score_range=(8,9), expected_outcome="moderately simple"),
        Rubric(score_range=(10,10), expected_outcome="very simple")
    ],
    verbose_mode=True
)


# =====================================================================
# STEP 2: THE TEST CASE
# =====================================================================

# Arguments:
#   input         -> the question (in the real pipeline: golden.input)
#   actual_output -> the app's answer to judge (in the real pipeline: final_state["response"])
# This metric needs ONLY these two fields, because evaluation_params lists only these two.
# Extra arguments you can add (ignored by this metric unless listed in evaluation_params):
#   expected_output="..."    -> the ideal answer
#   retrieval_context=[...]  -> the retrieved chunk texts
#   context=[...]            -> the ideal source text (ground truth)
#   name="linear regression" -> a label for the test case
#   additional_metadata={}   -> any extra info you want to keep with the case
test_case = LLMTestCase(
    input="What is Linear Regression",
    actual_output="Linear regression is a foundational supervised machine learning algorithm used to predict continuous numerical values. It models the linear relationship between a dependent (target) variable and one or more independent (input) variables by fitting the best possible straight line to the data"
)


# =====================================================================
# STEP 3: RUN THE METRIC
# =====================================================================

# measure(test_case) -> runs the judge steps (a to d in the flow above), stores the results inside the
#                       metric object, and RETURNS the score as a number from 0 to 1.
#                       print(...) shows that number. verbose_mode=True also prints the judge's steps.
#                       It uses LLM calls, so it costs a little money.
# (a_measure(test_case) is the async version.)
print(simple_explanation.measure(test_case))

# To read more of the results, add:
#   print(simple_explanation.reason)           # the judge's explanation of the score
#   print(simple_explanation.is_successful())  # True if score >= threshold
#
# Other ways to run the same metric:
#   from deepeval import evaluate
#   evaluate(test_cases=[test_case], metrics=[simple_explanation])  -> runs many test cases and metrics and prints a table
#   (in the real pipeline: evaluate(test_cases=eval_dataset.test_cases, metrics=[simple_explanation, ...]))
#   from deepeval import assert_test
#   assert_test(test_case, [simple_explanation])                    -> use inside a pytest function (run with: deepeval test run <file>)
#
# Run this file with:  uv run python simple_explanation_metric.py