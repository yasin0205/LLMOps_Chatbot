# =====================================================================
# WHAT THIS SCRIPT DOES
# It is a small demo of ONE evaluation metric: Contextual Recall.
# Question it answers: "Did the retriever bring back ALL the information
# that is needed to write the ideal answer?"
# It checks the RETRIEVER only (chunk size, k, embeddings). It does not judge the LLM's answer.
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  OPENAI_API_KEY ready (the metric uses an LLM as a judge)
#
#   STEP 1  write the inputs by hand
#              rag_input          the question
#              expected_output    the ideal answer (in the real pipeline: golden.expected_output)
#              retrieval_context  the chunks the retriever returned (in the real pipeline: retrieved_docs)
#      |
#      v
#   STEP 2  LLMTestCase(...)   packs the inputs into one test case
#      |
#      v
#   STEP 3  ContextualRecallMetric(...)   creates the metric (the judge settings)
#      |
#      v
#   STEP 4  recall.measure(test_case)   the judge LLM works in these steps:
#              a. split expected_output into separate statements (about one per sentence)
#              b. for each statement ask: "can this be found in the retrieval_context?"  yes / no
#              c. score = statements found / total statements        (0 to 1, higher is better)
#              d. write a short reason, and compare the score with the threshold
#      |
#      v
#   results: recall.score, recall.reason, recall.success   (verbose_mode=True also prints every step)
#
# WHERE THIS FITS IN THE PROJECT
#   goldens (generate_goldens.py) -> test cases from the RAG app (generate_eval_dataset.py)
#   -> metrics like this one. Here the three fields are typed by hand to see how the metric behaves;
#   in the real pipeline they come from the saved test cases.
#
# WHAT HAPPENS WITH THE EXAMPLE BELOW
#   The expected answer has 5 statements:
#     1. bootstrap samples with replacement        -> found in chunk 1
#     2. a separate model per subset               -> found in chunk 2
#     3. classification uses majority voting       -> NOT in any chunk
#     4. regression averages the predictions       -> found in chunk 2
#     5. reduces variance, prevents overfitting    -> NOT in any chunk
#   So the score should be about 3/5 = 0.6 (the judge is an LLM, so the exact split can vary slightly).
#   Chunk 3 (random forests) is off-topic. Recall does NOT punish irrelevant chunks;
#   that is the job of Contextual Precision.
#   LOW RECALL means the retriever missed information. Things to try: larger k, a different
#   chunk_size or chunk_overlap, a better embedding model.
# =====================================================================

# ContextualRecallMetric = the metric that measures how much of the expected answer
# is supported by the retrieved chunks
# (other retrieval metrics in the same module: ContextualPrecisionMetric, ContextualRelevancyMetric)
from deepeval.metrics import ContextualRecallMetric
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
rag_input = "How does bagging work?"

# the ideal answer. Recall splits THIS text into statements and checks each one against the chunks.
# A longer, more detailed expected answer = more statements to find = a stricter test.
expected_output = """Bagging, short for bootstrap aggregating, works by creating 
multiple subsets of the training data through random sampling with 
replacement. A separate model is trained on each subset independently. 
For classification, the final prediction is made by majority voting 
across all models. For regression, the predictions are averaged. This 
process reduces variance and helps prevent overfitting."""

# the chunks the retriever returned: a list of TEXT strings, one string per chunk
# (in the real pipeline this is [doc.page_content for doc in retrieved_docs])
# Here the first two chunks hold part of the answer, and the third is a distractor.
retrieval_context = [
    "Bagging creates multiple bootstrap samples from the training dataset — each sample is drawn randomly with replacement, meaning some data points may appear multiple times while others may be left out.",
    "In bagging, an independent model is trained on each bootstrap sample. For regression tasks, the final output is the average of all individual model predictions.",
    "Random forests are an extension of bagging that also introduces random feature selection at each split, in addition to bootstrap sampling."
    ]


# =====================================================================
# STEP 2: THE TEST CASE
# =====================================================================

# test case
# Arguments:
#   input             -> the question
#   retrieval_context -> list of the retrieved chunk texts (what the retriever really returned)
#   expected_output   -> the ideal answer
#   actual_output     -> what the RAG app answered. Contextual Recall does NOT use it for the score:
#                        it only compares expected_output with retrieval_context.
#                        The field is still required by the metric, so an empty string "" is passed.
#                        In the real pipeline it holds the app's response.
# Extra arguments you can add:
#   context=[...]              -> the ideal source text (ground truth), used by other metrics such as hallucination
#   name="bagging case"        -> a label for the test case
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
#   verbose_mode=True -> print every step the judge takes (statements, yes/no verdicts, reasons,
#                        final score) to the console. Default False. Use True while learning or debugging.
# Extra arguments you can add:
#   threshold=0.5       -> pass mark between 0 and 1. success is True when score >= threshold. Default 0.5.
#                          Use a higher value (0.7 to 0.9) for a stricter retriever test.
#   model="gpt-4.1"     -> the judge LLM. A model name string, or your own class based on DeepEvalBaseLLM
#                          (for other providers). Default: an OpenAI model that depends on your DeepEval
#                          version. A stronger judge gives more reliable scores; a cheaper one saves money.
#   include_reason=True -> also write a short explanation of the score (default True).
#                          False skips that LLM call, which is cheaper and faster.
#   strict_mode=False   -> True makes the score all-or-nothing: 1 if everything is found, otherwise 0
#                          (the threshold is forced to 1).
#   async_mode=True     -> run the judge calls in parallel (faster). Default True.
recall = ContextualRecallMetric(verbose_mode=True)


# =====================================================================
# STEP 4: RUN THE METRIC
# =====================================================================

# test the metric
# measure(test_case) -> runs the judge steps (a to d in the flow above) and stores the results
#                       inside the metric object. It uses LLM calls, so it costs a little money.
# (a_measure(test_case) is the async version.)
recall.measure(test_case=test_case)

# The script above prints the steps because verbose_mode=True. To read the results yourself, add:
#   print(recall.score)            # the number between 0 and 1
#   print(recall.reason)           # the judge's explanation
#   print(recall.is_successful())  # True if score >= threshold
#
# Other ways to run the same metric:
#   from deepeval import evaluate
#   evaluate(test_cases=[test_case], metrics=[recall])    -> runs many test cases and many metrics and prints a table
#   from deepeval import assert_test
#   assert_test(test_case, [recall])                      -> use inside a pytest function (run with: deepeval test run <file>)
#
# Run this file with:  uv run python test_recall_metric.py