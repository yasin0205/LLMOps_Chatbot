# =====================================================================
# WHAT THIS SCRIPT DOES
# It is a small demo of ONE evaluation metric: Contextual Relevancy.
# Question it answers: "How much of what the retriever brought back is actually
# relevant to the question?"  (low score = the retrieved context is full of noise)
# It checks the RETRIEVER only (k, chunk size, search type). It does not judge the LLM's answer.
#
# HOW IT DIFFERS FROM THE OTHER TWO RETRIEVAL METRICS
#   Contextual Recall     did we retrieve EVERYTHING needed for the ideal answer?   (needs expected_output)
#   Contextual Precision  are the relevant chunks ranked ABOVE the irrelevant ones? (needs expected_output)
#   Contextual Relevancy  how much of the retrieved text is RELEVANT to the question? (needs only the question)
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  OPENAI_API_KEY ready (the metric uses an LLM as a judge)
#
#   STEP 1  write the inputs by hand
#              rag_input          the question
#              retrieval_context  the chunks the retriever returned (in the real pipeline: retrieved_docs)
#      |
#      v
#   STEP 2  LLMTestCase(...)   packs the inputs into one test case
#      |
#      v
#   STEP 3  ContextualRelevancyMetric(...)   creates the metric (the judge settings)
#      |
#      v
#   STEP 4  context_relevancy.measure(test_case)   the judge LLM works in these steps:
#              a. split every retrieved chunk into separate statements
#              b. for each statement ask: "is this relevant to the question?"  yes / no (+ a reason)
#              c. score = relevant statements / total statements      (0 to 1, higher is better)
#              d. write a short reason, and compare the score with the threshold
#      |
#      v
#   results: context_relevancy.score, .reason, .success   (verbose_mode=True also prints every step)
#
# WHERE THIS FITS IN THE PROJECT
#   goldens (generate_goldens.py) -> test cases from the RAG app (generate_eval_dataset.py)
#   -> metrics like this one. Here the inputs are typed by hand to see how the metric behaves;
#   in the real pipeline they come from the saved test cases.
#
# WHAT HAPPENS WITH THE EXAMPLE BELOW
#   Chunk 1 explains how KNN works (majority vote of the K neighbours, no training, slow prediction): relevant.
#   Chunk 2 explains how to choose K: mostly relevant to "how does KNN work".
#   Chunk 3 is mainly about Support Vector Machines (hyperplane, margin): mostly NOT relevant to the question.
#   So the score should land somewhere in the middle (roughly 0.5 to 0.7). The exact number depends on
#   how the judge LLM splits and rates the statements, so it can vary a little between runs.
#   The reason text lists which statements were judged irrelevant.
#   LOW RELEVANCY means the context carries noise. Things to try: a smaller k, a smaller chunk_size
#   (big chunks hold many unrelated sentences), a score threshold on the retriever
#   (search_type="similarity_score_threshold"), or a reranker.
# =====================================================================

# ContextualRelevancyMetric = the metric that measures how relevant the retrieved chunks are to the question
# (other retrieval metrics in the same module: ContextualRecallMetric, ContextualPrecisionMetric)
from deepeval.metrics import ContextualRelevancyMetric
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
# Relevancy compares every retrieved statement against THIS question,
# so it is the only "truth" the metric needs (no expected answer).
rag_input =  "How does the K-Nearest Neighbors (KNN) algorithm work?"


# the chunks the retriever returned: a list of TEXT strings, one string per chunk
# (in the real pipeline this is [doc.page_content for doc in retrieved_docs])
# Here chunks 1 and 2 are on topic, and chunk 3 is mostly about SVMs, so it adds noise.
retrieval_context = [
    "KNN is a non-parametric algorithm that classifies a data point based on the majority class among its K nearest neighbors in the feature space. It requires no training phase, since the entire dataset is stored and used at prediction time. This makes prediction slow for large datasets, since distance must be computed to every point.",
    "Choosing the right value of K is important — small K makes the model sensitive to noise, while large K smooths the decision boundary but may include points from other classes.",
    "Support Vector Machines find the optimal hyperplane that maximizes the margin between classes. KNN is often compared to SVMs as both are used for classification, but SVMs are parametric and build an explicit decision boundary during training, unlike KNN's lazy learning approach."
    ]


# =====================================================================
# STEP 2: THE TEST CASE
# =====================================================================

# test case
# Arguments:
#   input             -> the question
#   retrieval_context -> list of the retrieved chunk texts (what the retriever really returned)
#   actual_output     -> what the RAG app answered. Contextual Relevancy does NOT use it for the score:
#                        it only compares the question with the retrieved chunks.
#                        The field is still required by the metric, so an empty string "" is passed.
#                        In the real pipeline it holds the app's response.
# There is no expected_output here on purpose: this metric does not need an ideal answer
# (Contextual Recall and Contextual Precision do).
# Extra arguments you can add:
#   expected_output="..."      -> the ideal answer (needed by other metrics, ignored by this one)
#   context=[...]              -> the ideal source text (ground truth), used by other metrics such as hallucination
#   name="knn case"            -> a label for the test case
#   additional_metadata={}     -> any extra info you want to keep with the case
test_case = LLMTestCase(
    input=rag_input,
    retrieval_context=retrieval_context,
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
#   include_reason=True -> also write a short explanation of the score, including which statements
#                          were irrelevant (default True). False skips that LLM call, which is cheaper and faster.
#   strict_mode=False   -> True makes the score all-or-nothing: 1 if everything is relevant, otherwise 0
#                          (the threshold is forced to 1).
#   async_mode=True     -> run the judge calls in parallel (faster). Default True.
context_relevancy = ContextualRelevancyMetric(verbose_mode=True)


# =====================================================================
# STEP 4: RUN THE METRIC
# =====================================================================

# test the metric
# measure(test_case) -> runs the judge steps (a to d in the flow above) and stores the results
#                       inside the metric object. It uses LLM calls, so it costs a little money.
#                       More chunks and longer chunks mean more statements to judge, so more cost.
# (a_measure(test_case) is the async version.)
context_relevancy.measure(test_case=test_case)

# The script above prints the steps because verbose_mode=True. To read the results yourself, add:
#   print(context_relevancy.score)            # the number between 0 and 1
#   print(context_relevancy.reason)           # the judge's explanation (which statements were irrelevant)
#   print(context_relevancy.is_successful())  # True if score >= threshold
#
# Other ways to run the same metric:
#   from deepeval import evaluate
#   evaluate(test_cases=[test_case], metrics=[context_relevancy])  -> runs many test cases and metrics and prints a table
#   from deepeval import assert_test
#   assert_test(test_case, [context_relevancy])                    -> use inside a pytest function (run with: deepeval test run <file>)
#
# Run this file with:  uv run python test_relevancy_metric.py