# =====================================================================
# WHAT THIS SCRIPT DOES
# It builds a CUSTOM metric called "answer correctness" with G-Eval, and tests it on one test case.
# Question it answers: "Is the app's answer CORRECT compared with the ideal answer?"
# This is the only metric in the set that compares actual_output with expected_output.
# So it checks the whole app end to end (retriever + LLM), not one part.
#
# WHY A CUSTOM METRIC?
#   The ready-made metrics each check one fixed thing:
#     Contextual Recall / Precision / Relevancy -> the retriever (the chunks)
#     Answer Relevancy                          -> is the answer on topic?
#     Faithfulness                              -> is the answer supported by the chunks?
#   None of them asks "does the answer match the ideal answer?". G-Eval lets you describe your own
#   rule in plain English (criteria) and a scoring guide (rubric), and an LLM judge applies it.
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  OPENAI_API_KEY ready (the metric uses an LLM as a judge)
#
#   STEP 1  define the metric: GEval(name, evaluation_params, criteria, rubric, ...)
#              evaluation_params  which test case fields the judge may see (expected_output, actual_output)
#              criteria           what to judge, in plain English
#              rubric             the scoring guide: which score range means what
#      |
#      v
#   STEP 2  LLMTestCase(...)   one test case: expected_output (the ideal answer) + actual_output (the app's answer)
#      |
#      v
#   STEP 3  answer_correctness.measure(test_case)   the judge LLM works in these steps:
#              a. turn the criteria into a short list of evaluation steps
#                 (skipped if you pass evaluation_steps yourself)
#              b. read the fields listed in evaluation_params, follow the steps, and pick a score from 0 to 10
#                 inside the rubric range that fits best
#              c. DeepEval converts it to 0 to 1 (score / 10); when the judge model supports it, the score
#                 is also averaged over likely alternatives (token probabilities) to make it more stable
#              d. write a short reason, and compare the score with the threshold
#      |
#      v
#   measure() returns the score, and the results stay in answer_correctness.score, .reason, .success
#   (verbose_mode=True also prints every step)
#
# WHERE THIS FITS IN THE PROJECT
#   goldens (generate_goldens.py) -> test cases from the RAG app (generate_eval_dataset.py) -> metrics.
#   In the real pipeline expected_output comes from the golden and actual_output is the app's response.
#   Here both are typed by hand to see how the metric behaves.
#
# WHAT HAPPENS WITH THE EXAMPLE BELOW
#   The ideal answer lists 3 approaches (manual, historical data, synthetic data) with pros and cons.
#   The app's answer also names all three, but:
#     - Historical data: it says "reuse previous-year question papers", while the ideal answer says
#       "real previous USER questions". That is a wrong fact.
#     - It leaves out most of the ideal pros and cons (bias, cost, coverage, knowledge base updates ...)
#       and gives different pros for manual creation.
#     - It adds unrelated notes (retriever accuracy, RAGas) at the end.
#   The criteria say "penalize wrong facts strictly", so the judge should land in the lower rubric bands
#   (about 0.3 to 0.6 after conversion). The judge is an LLM, so the number can vary between runs.
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

# define the custom metric
# Arguments used here:
#   name              -> a label for the metric; it shows up in logs and result tables
#   evaluation_params -> the list of test case fields the judge is allowed to see.
#                        Choices: SingleTurnParams.INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT, CONTEXT,
#                        RETRIEVAL_CONTEXT, TOOLS_CALLED, EXPECTED_TOOLS.
#                        Here only the two answers are listed, so the judge does not see the question.
#                        Add SingleTurnParams.INPUT if the judge should also read the question.
#                        Only list what you need: every extra field costs tokens and can distract the judge.
#   criteria          -> what to judge, in plain English. Use the field names (expected_output,
#                        actual_output) so the judge knows what each one is. Be specific: vague criteria
#                        give unstable scores. Use EITHER criteria OR evaluation_steps, not both.
#   rubric            -> the scoring guide: a list of Rubric(score_range=(low, high), expected_outcome="...").
#                        Ranges are whole numbers from 0 to 10, and they must not overlap.
#                        The judge picks the band that fits and gives a score inside it.
#                        Here: 0 to 5 = wrong facts, 6 to 9 = mostly correct, 10 = fully correct.
#                        Without a rubric the judge scores freely from 0 to 10.
#   verbose_mode      -> True prints every step the judge takes. Default False.
# Extra arguments you can add:
#   evaluation_steps=["Step 1 ...", "Step 2 ..."] -> your own list of steps instead of criteria.
#                          More control and more consistent scores, because the judge does not
#                          invent its own steps.
#   threshold=0.5     -> pass mark on the converted 0 to 1 score. Default 0.5. NOTE: a rubric score of 5
#                        (the top of the "incorrect facts" band) becomes 0.5, which still PASSES at 0.5.
#                        Use threshold=0.6 so that only "mostly correct" (6 and above) passes.
#   model="gpt-4.1"   -> the judge LLM. A model name string, or your own class based on DeepEvalBaseLLM.
#                        The probability-weighted scoring only works with models that return token
#                        probabilities (OpenAI models do); others still work, with a less smooth score.
#   strict_mode=False -> True makes the score all-or-nothing (1 or 0, threshold forced to 1)
#   async_mode=True   -> run the judge calls in parallel (faster). Default True.
#   top_logprobs=20   -> how many alternative scores are used for the probability weighting
answer_correctness = GEval(
    name="answer correctness",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    criteria="""Evaluate the LLM response based on correctness of answer. Compare
    the 'expected_output' with the 'actual_output'. Penalize wrong facts strictly""",
    rubric=[Rubric(score_range=(0,5), expected_outcome="Answer has incorrect facts"),
            Rubric(score_range=(6,9), expected_outcome="Answer is mostly correct but has minor differences"),
            Rubric(score_range=(10,10), expected_outcome=r"100% correct")],
    verbose_mode=True
)


# =====================================================================
# STEP 2: THE TEST CASE
# =====================================================================

# Arguments:
#   input           -> the question. Empty here on purpose: INPUT is not in evaluation_params, so the
#                      judge never sees it. The field must still exist, so "" is passed.
#   expected_output -> the ideal answer (in the real pipeline: golden.expected_output)
#   actual_output   -> the app's answer to judge (in the real pipeline: final_state["response"])
# (Each \n inside the strings below is a line break. The strings are on one line each, as in the original.)
# Extra arguments you can add (ignored by this metric unless listed in evaluation_params):
#   retrieval_context=[...]  -> the retrieved chunk texts
#   context=[...]            -> the ideal source text (ground truth)
#   name="question paper"    -> a label for the test case
#   additional_metadata={}   -> any extra info you want to keep with the case
test_case = LLMTestCase(
    input="",
    expected_output="The 3 question-paper setup approaches are:\n\n1. **Manual creation**\n   - **Pros:** Human control, can ensure coverage and balanced difficulty.\n   - **Cons:** Biased, costly, hard to scale, depends on human availability, and breaks when the knowledge base updates.\n\n2. **Historical data**\n   - **Pros:** Uses real previous user questions, so it reflects actual usage.\n   - **Cons:** May have low coverage, still needs correct answers for scoring, and can be limited by one type of user behavior.\n\n3. **Synthetic data**\n   - **Pros:** Automated, scalable, lower cost, less bias, no fatigue/illness issues, and adapts better to knowledge base updates.\n   - **Cons:** Needs a smart generation method, which was only briefly mentioned in the context.",
    actual_output="- Manual (human-crafted question papers)\n  - Pro: can create tailored exams across multiple subjects.\n  - Con: high manual effort and poor scalability (you must create many exams/subjects by hand).\n\n- Synthetic data (automatically generated Q&A pairs)\n  - Pro: generate question–answer pairs programmatically (e.g., using a framework like DeepEval) to set the question paper — reduces manual creation.\n  - Con: no specific downsides are mentioned in the provided context.\n\n- Historical data (reuse previous-year question papers)\n  - Pro: reuse/fetch existing questions from past papers so you can assemble papers without creating everything anew.\n  - Con: no specific downsides are mentioned in the provided context.\n\n(Additional context notes mentioned: evaluation is based on retriever accuracy, tests may run offline or in production, and tools like RAGas are part of the LLM eval setup.)",
)


# =====================================================================
# STEP 3: RUN THE METRIC
# =====================================================================

# measure(test_case) -> runs the judge steps (a to d in the flow above), stores the results inside the
#                       metric object, and RETURNS the score as a number from 0 to 1.
#                       print(...) shows that number. verbose_mode=True also prints the judge's steps.
#                       It uses LLM calls, so it costs a little money.
# (a_measure(test_case) is the async version.)
print(answer_correctness.measure(test_case=test_case))

# To read more of the results, add:
#   print(answer_correctness.reason)           # the judge's explanation of the score
#   print(answer_correctness.is_successful())  # True if score >= threshold
#
# Other ways to run the same metric:
#   from deepeval import evaluate
#   evaluate(test_cases=[test_case], metrics=[answer_correctness])  -> runs many test cases and metrics and prints a table
#   (in the real pipeline: evaluate(test_cases=eval_dataset.test_cases, metrics=[answer_correctness, ...]))
#   from deepeval import assert_test
#   assert_test(test_case, [answer_correctness])                    -> use inside a pytest function (run with: deepeval test run <file>)
#
# Run this file with:  uv run python answer_correctness_metric.py