# =====================================================================
# WHAT THIS SCRIPT DOES
# It is a small demo of ONE evaluation metric: Contextual Precision.
# Question it answers: "Are the useful chunks ranked ABOVE the useless ones?"
# ORDER MATTERS here: the list retrieval_context is read from first chunk to last,
# like the ranking your retriever returns (best similarity first).
# It checks the RETRIEVER only (ranking quality, embeddings, k, reranking).
# It does not judge the LLM's answer.
#
# HOW IT DIFFERS FROM THE OTHER TWO RETRIEVAL METRICS
#   Contextual Recall     did we retrieve EVERYTHING needed for the ideal answer?     (needs expected_output)
#   Contextual Precision  are the relevant chunks ranked ABOVE the irrelevant ones?   (needs expected_output)
#   Contextual Relevancy  how much of the retrieved text is RELEVANT to the question? (needs only the question)
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  OPENAI_API_KEY ready (the metric uses an LLM as a judge)
#
#   STEP 1  write the inputs by hand
#              rag_input          the question
#              expected_output    the ideal answer (in the real pipeline: golden.expected_output)
#              retrieval_context  the chunks the retriever returned, IN RANK ORDER
#                                 (in the real pipeline: retrieved_docs)
#      |
#      v
#   STEP 2  LLMTestCase(...)   packs the inputs into one test case
#      |
#      v
#   STEP 3  ContextualPrecisionMetric(...)   creates the metric (the judge settings)
#      |
#      v
#   STEP 4  context_precision.measure(test_case)   the judge LLM works in these steps:
#              a. go through the chunks in order, from rank 1 to the last
#              b. for each chunk ask: "is it useful to reach the expected answer for this question?"
#                 yes / no (+ a reason)
#              c. score = average of the precision at every position that holds a relevant chunk
#                 (a weighted score: a relevant chunk at rank 1 counts more than the same chunk at rank 5)
#              d. write a short reason, and compare the score with the threshold
#      |
#      v
#   results: context_precision.score, .reason, .success   (verbose_mode=True also prints every step)
#
# WHERE THIS FITS IN THE PROJECT
#   goldens (generate_goldens.py) -> test cases from the RAG app (generate_eval_dataset.py)
#   -> metrics like this one. Here the inputs are typed by hand to see how the metric behaves;
#   in the real pipeline they come from the saved test cases.
#   The order of retrieved_docs from your retriever is the similarity ranking (best match first),
#   so this metric tests how good that ranking is.
#
# HOW THE SCORE IS CALCULATED (simple version)
#   For every position k that holds a RELEVANT chunk:  precision at k = relevant chunks in the top k / k
#   score = the average of these values over all relevant chunks.
#
# WHAT HAPPENS WITH THE EXAMPLE BELOW
#   Chunks 1, 2 and 3 are all about linear regression (relevant). Chunks 4 and 5 (logistic regression,
#   decision trees) are noise, but they come LAST.
#     rank 1 relevant: 1/1,  rank 2 relevant: 2/2,  rank 3 relevant: 3/3   ->  average = 1.0
#   So the score should be about 1.0: the noise exists, but it sits at the bottom, so ranking is perfect.
#   (The judge is an LLM, so it may rate a chunk differently and the score can vary a little.)
#
#   TRY THIS to see the metric work: put the two irrelevant chunks FIRST in retrieval_context
#   (order: chunk 4, chunk 5, chunk 1, chunk 2, chunk 3). Relevant chunks are now at ranks 3, 4, 5:
#     rank 3: 1/3,  rank 4: 2/4,  rank 5: 3/5   ->  (0.33 + 0.50 + 0.60) / 3  =  about 0.48
#   Same chunks, worse order, much lower score. Recall would stay the same, because it ignores order.
#   LOW PRECISION means good chunks are buried under bad ones. Things to try: a better embedding model,
#   a reranker, hybrid (keyword + vector) search, or cleaner and smaller chunks.
# =====================================================================

# The three retrieval metrics. Only ContextualPrecisionMetric is used in this file.
# ContextualRelevancyMetric and ContextualRecallMetric are imported but not used here (see the end of the file
# for how to run all three on the same test case).
from deepeval.metrics import ContextualPrecisionMetric, ContextualRelevancyMetric, ContextualRecallMetric
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
rag_input = "What is linear regression?"

# the ideal answer. Precision uses it to decide which chunks are "useful":
# a chunk is useful if it helps to reach this answer for this question.
expected_output = """Linear regression is a supervised learning algorithm that models the relationship 
between a dependent variable and one or more independent variables by fitting a straight line to 
the data, typically using least squares to minimize the error."""

# the chunks the retriever returned: a list of TEXT strings, one string per chunk, IN RANK ORDER
# (in the real pipeline this is [doc.page_content for doc in retrieved_docs])
# Chunks 1 to 3 are on topic. Chunks 4 and 5 are noise, and they are ranked last.
retrieval_context = [
    "Linear regression assumes a linear relationship between input features and the target variable, and fits a line by minimizing sum of squared residuals.",
    "In linear regression, the model estimates coefficients (slope and intercept) that best fit the training data using the least squares method.",
    "Linear regression is a foundational supervised learning technique used to predict a continuous dependent variable from one or more independent variables.",
    "Logistic regression is used for classification tasks, not regression, despite the name.",
    "Decision trees split data based on feature thresholds to minimize impurity at each node.",
    ]


# =====================================================================
# STEP 2: THE TEST CASE
# =====================================================================

# test case
# Arguments:
#   input             -> the question
#   retrieval_context -> list of the retrieved chunk texts, in the order the retriever ranked them
#   expected_output   -> the ideal answer (used to decide which chunks are useful)
#   actual_output     -> what the RAG app answered. Contextual Precision does NOT use it for the score:
#                        it only compares the question and expected answer with the ranked chunks.
#                        The field is still required by the metric, so an empty string "" is passed.
#                        In the real pipeline it holds the app's response.
# Extra arguments you can add:
#   context=[...]              -> the ideal source text (ground truth), used by other metrics such as hallucination
#   name="linear regression"   -> a label for the test case
#   additional_metadata={}     -> any extra info you want to keep with the case
test_case = LLMTestCase(
    input=rag_input,
    retrieval_context=retrieval_context,
    expected_output=expected_output,
    actual_output=""
)


# =====================================================================
# STEP 3: THE METRIC
# =====================================================================

# metric
# Only verbose_mode is set here; everything else uses the defaults.
# Arguments:
#   verbose_mode=True -> print every step the judge takes (yes/no verdict for each chunk in order,
#                        reasons, final score) to the console. Default False.
#                        Use True while learning or debugging.
# Extra arguments you can add:
#   threshold=0.5       -> pass mark between 0 and 1. success is True when score >= threshold. Default 0.5.
#                          Use a higher value (0.7 to 0.9) for a stricter ranking test.
#   model="gpt-4.1"     -> the judge LLM. A model name string, or your own class based on DeepEvalBaseLLM
#                          (for other providers). Default: an OpenAI model that depends on your DeepEval
#                          version. A stronger judge gives more reliable scores; a cheaper one saves money.
#   include_reason=True -> also write a short explanation of the score (default True).
#                          False skips that LLM call, which is cheaper and faster.
#   strict_mode=False   -> True makes the score all-or-nothing: 1 only if every relevant chunk is ranked
#                          above every irrelevant one, otherwise 0 (the threshold is forced to 1).
#   async_mode=True     -> run the judge calls in parallel (faster). Default True.
context_precision= ContextualPrecisionMetric(verbose_mode=True)


# =====================================================================
# STEP 4: RUN THE METRIC
# =====================================================================

# test the metric
# measure(test_case) -> runs the judge steps (a to d in the flow above) and stores the results
#                       inside the metric object. It uses LLM calls, so it costs a little money.
# (a_measure(test_case) is the async version.)
context_precision.measure(test_case=test_case)

# The script above prints the steps because verbose_mode=True. To read the results yourself, add:
#   print(context_precision.score)            # the number between 0 and 1
#   print(context_precision.reason)           # the judge's explanation
#   print(context_precision.is_successful())  # True if score >= threshold
#
# Run all three retrieval metrics on the same test case (uses the two imports that are otherwise unused):
#   from deepeval import evaluate
#   context_relevancy = ContextualRelevancyMetric()
#   context_recall = ContextualRecallMetric()
#   evaluate(test_cases=[test_case], metrics=[context_precision, context_relevancy, context_recall])
#   For this example you would expect: precision high (noise is ranked last), relevancy lower (2 of 5 chunks are noise),
#   recall high (the expected answer is covered by chunks 1 to 3).
#
# Other way to run it as a test:
#   from deepeval import assert_test
#   assert_test(test_case, [context_precision])   -> use inside a pytest function (run with: deepeval test run <file>)
#
# Run this file with:  uv run python test_precision_metric.py