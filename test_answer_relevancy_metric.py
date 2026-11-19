# =====================================================================
# WHAT THIS SCRIPT DOES
# It is a small demo of ONE evaluation metric: Answer Relevancy.
# Question it answers: "Does the app's ANSWER actually address the question,
# or does it contain off-topic sentences?"
# It checks the GENERATION part (the LLM's answer), not the retriever.
# Unlike the three contextual metrics, it needs no retrieved chunks and no expected answer:
# only the question and the answer.
#
# HOW IT DIFFERS FROM THE CONTEXTUAL METRICS
#   Contextual Recall / Precision / Relevancy  -> judge the RETRIEVER (the chunks)
#   Answer Relevancy                           -> judges the ANSWER against the question
#   It does NOT check if the answer is true. A wrong answer that stays on topic still scores high.
#   For truthfulness use Faithfulness (is the answer supported by retrieval_context?).
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  OPENAI_API_KEY ready (the metric uses an LLM as a judge)
#
#   STEP 1  write the inputs by hand
#              rag_input    the question
#              actual_ouput the answer to judge (in the real pipeline: final_state["response"] from the RAG graph)
#      |
#      v
#   STEP 2  LLMTestCase(...)   packs the inputs into one test case
#      |
#      v
#   STEP 3  AnswerRelevancyMetric(...)   creates the metric (the judge settings)
#      |
#      v
#   STEP 4  answer_relevancy.measure(test_case)   the judge LLM works in these steps:
#              a. split the answer into separate statements (about one per sentence)
#              b. for each statement ask: "is this relevant to the question?"
#                 yes / no, or "idk" for borderline statements (+ a reason for the "no" ones)
#              c. score = relevant statements / total statements      (0 to 1, higher is better)
#                 (borderline statements are normally counted as relevant; only clear "no" verdicts lower the score)
#              d. write a short reason, and compare the score with the threshold
#      |
#      v
#   results: answer_relevancy.score, .reason, .success   (verbose_mode=True also prints every step)
#
# WHERE THIS FITS IN THE PROJECT
#   goldens (generate_goldens.py) -> test cases from the RAG app (generate_eval_dataset.py)
#   -> metrics like this one. Here the question and answer are typed by hand to see how the metric behaves;
#   in the real pipeline they come from the saved test cases.
#
# WHAT HAPPENS WITH THE EXAMPLE BELOW
#   The question asks what boosting is and how it works. The answer has 5 statements:
#     1. boosting combines weak learners into a strong learner     -> relevant
#     2. models are trained one after another, fixing earlier errors -> relevant
#     3. AdaBoost was one of the first popular boosting algorithms -> related, may be borderline
#     4. Random Forests build trees independently and average them -> OFF TOPIC (a different technique)
#     5. Gradient Boosting uses gradient descent to minimize a loss -> relevant (explains how boosting works)
#   So the score should be about 0.8 (4 of 5). It could drop to 0.6 if the judge rejects the AdaBoost sentence.
#   The judge is an LLM, so the exact number can vary a little. The reason text names the off-topic sentence.
#   LOW ANSWER RELEVANCY means the answer rambles or drifts. Things to try: tighten the system prompt
#   (for example "answer only what was asked, in 2-3 sentences"), reduce noisy retrieved chunks
#   (noise in the context often leaks into the answer), or use a stronger LLM.
# =====================================================================

# AnswerRelevancyMetric = the metric that measures how relevant the answer is to the question
# (a related metric for truthfulness: FaithfulnessMetric)
from deepeval.metrics import AnswerRelevancyMetric
# LLMTestCase = one test case: what went in, what came out, what was expected
from deepeval.test_case.llm_test_case import LLMTestCase
# load_dotenv reads the .env file (OPENAI_API_KEY lives there)
from dotenv import load_dotenv

# load the api keys
# (arguments of load_dotenv: dotenv_path=".env" to point to a file, override=True to replace
#  existing variables, verbose=True to warn when no .env is found)
load_dotenv()


# =====================================================================
# STEP 1: THE INPUTS
# =====================================================================

# the question that was asked
# Every statement of the answer is compared against THIS question.
rag_input =  "What is boosting in machine learning and how does it work?"

# the answer to judge: in the real pipeline, this is what your RAG app produced (final_state["response"]).
# Here it is written by hand, with one deliberately off-topic sentence (the Random Forests one).
actual_ouput = """Boosting is an ensemble technique that combines multiple weak learners to form a strong learner. 
It trains models sequentially, where each new model focuses on correcting the errors made by the previous ones. 
AdaBoost was one of the first popular boosting algorithms. Random Forests, on the other hand, build trees independently and 
average their results. Gradient Boosting uses gradient descent to minimize a loss function when adding new learners."""


# =====================================================================
# STEP 2: THE TEST CASE
# =====================================================================

# test case
# Arguments:
#   input         -> the question
#   actual_output -> the answer to judge (the keyword is spelled correctly here; only the variable is "actual_ouput")
# This metric needs ONLY these two fields. There is no retrieval_context and no expected_output,
# because it never looks at the chunks or at an ideal answer.
# Extra arguments you can add (ignored by this metric, used by others):
#   expected_output="..."      -> the ideal answer
#   retrieval_context=[...]    -> the retrieved chunk texts
#   context=[...]              -> the ideal source text (ground truth)
#   name="boosting case"       -> a label for the test case
#   additional_metadata={}     -> any extra info you want to keep with the case
test_case = LLMTestCase(
    input=rag_input,
    actual_output=actual_ouput
)


# =====================================================================
# STEP 3: THE METRIC
# =====================================================================

# metric
# Only verbose_mode is set here; everything else uses the defaults.
# Arguments:
#   verbose_mode=True -> print every step the judge takes (statements, yes/no verdicts, reasons,
#                        final score) to the console. Default False. Use True while learning or debugging.
# Extra arguments you can add:
#   threshold=0.5       -> pass mark between 0 and 1. success is True when score >= threshold. Default 0.5.
#                          Use a higher value (0.7 to 0.9) if off-topic sentences are not acceptable.
#   model="gpt-4.1"     -> the judge LLM. A model name string, or your own class based on DeepEvalBaseLLM
#                          (for other providers). Default: an OpenAI model that depends on your DeepEval
#                          version. A stronger judge gives more reliable scores; a cheaper one saves money.
#   include_reason=True -> also write a short explanation of the score, including which statements
#                          were off topic (default True). False skips that LLM call, which is cheaper and faster.
#   strict_mode=False   -> True makes the score all-or-nothing: 1 if every statement is relevant, otherwise 0
#                          (the threshold is forced to 1).
#   async_mode=True     -> run the judge calls in parallel (faster). Default True.
answer_relevancy = AnswerRelevancyMetric(verbose_mode=True)


# =====================================================================
# STEP 4: RUN THE METRIC
# =====================================================================

# test the metric
# measure(test_case) -> runs the judge steps (a to d in the flow above) and stores the results
#                       inside the metric object. It uses LLM calls, so it costs a little money.
# (a_measure(test_case) is the async version.)
answer_relevancy.measure(test_case=test_case)

# The script above prints the steps because verbose_mode=True. To read the results yourself, add:
#   print(answer_relevancy.score)            # the number between 0 and 1
#   print(answer_relevancy.reason)           # the judge's explanation (names the off-topic statements)
#   print(answer_relevancy.is_successful())  # True if score >= threshold
#
# Other ways to run the same metric:
#   from deepeval import evaluate
#   evaluate(test_cases=[test_case], metrics=[answer_relevancy])  -> runs many test cases and metrics and prints a table
#   from deepeval import assert_test
#   assert_test(test_case, [answer_relevancy])                    -> use inside a pytest function (run with: deepeval test run <file>)
#
# Run this file with:  uv run python test_answer_relevancy_metric.py