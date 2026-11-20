# =====================================================================
# WHAT THIS SCRIPT DOES
# It is a small demo of ONE evaluation metric: Faithfulness.
# Question it answers: "Does the app's ANSWER stick to the retrieved chunks,
# or does it contradict them or add facts that are not in them?"  (hallucination check)
# It checks the GENERATION part (the LLM's answer) against the chunks the retriever returned.
#
# HOW IT DIFFERS FROM THE OTHER METRICS
#   Contextual Recall / Precision / Relevancy  -> judge the RETRIEVER (the chunks)
#   Answer Relevancy   -> is the answer ON TOPIC for the question?
#   Faithfulness       -> is the answer SUPPORTED by the retrieved chunks?
#   It compares the answer with the retrieved chunks, not with the real world. If the retriever
#   returns a wrong chunk and the LLM repeats it faithfully, the score is still high.
#   It also tests whether the LLM obeyed the system prompt rule "answer based on the given context only".
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  OPENAI_API_KEY ready (the metric uses an LLM as a judge)
#
#   STEP 1  write the inputs by hand
#              rag_input          the question
#              actual_output      the answer to judge (in the real pipeline: final_state["response"])
#              retrieval_context  the chunks the retriever returned (in the real pipeline: retrieved_docs)
#      |
#      v
#   STEP 2  LLMTestCase(...)   packs the inputs into one test case
#      |
#      v
#   STEP 3  FaithfulnessMetric(...)   creates the metric (the judge settings)
#      |
#      v
#   STEP 4  faithfulness.measure(test_case)   the judge LLM works in these steps:
#              a. extract "truths" (short facts) from the retrieval_context
#                 (at most truths_extraction_limit of them)
#              b. extract "claims" (statements) from the actual_output
#              c. for each claim, compare it with the truths and give a verdict:
#                   yes = supported by the truths
#                   no  = contradicts the truths
#                   idk = not mentioned in the truths (cannot be confirmed)
#              d. score = faithful claims / total claims          (0 to 1, higher is better)
#                 with penalize_ambiguous_claims=False (used below) "idk" claims still count as faithful,
#                 so only claims that CONTRADICT the context lower the score
#              e. write a short reason, and compare the score with the threshold
#      |
#      v
#   results: faithfulness.score, .reason, .success   (verbose_mode=True also prints every step)
#
# WHERE THIS FITS IN THE PROJECT
#   goldens (generate_goldens.py) -> test cases from the RAG app (generate_eval_dataset.py)
#   -> metrics like this one. Here the three fields are typed by hand to see how the metric behaves;
#   in the real pipeline they come from the saved test cases.
#
# WHAT HAPPENS WITH THE EXAMPLE BELOW
#   The answer has about 6 claims:
#     1. kNN is non-parametric, for classification and regression        -> supported
#     2. uses the k closest points, Euclidean distance, vote or average   -> supported
#     3. stores the data, no explicit model, "lazy learner"               -> supported
#     4. a very SMALL k makes the model MORE robust to noise              -> CONTRADICTS the context
#                                                                            (the context says small k is sensitive to noise)
#     5. feature scaling matters because kNN uses distances              -> supported
#     6. training time complexity is O(n log n) because of tree structures -> NOT in the context ("idk"),
#                                                                            or judged as a contradiction, because kNN builds no model in training
#   Expected score with the settings below: about 0.83 (5 of 6) if claim 6 is "idk", or about 0.67 (4 of 6)
#   if the judge marks it as a contradiction. The judge is an LLM, so the exact numbers can vary.
#   WITH penalize_ambiguous_claims=True the "idk" claim 6 would also count against the answer
#   (about 4 of 6 = 0.67), so the invented O(n log n) fact would no longer slip through.
#   LOW FAITHFULNESS means the LLM adds or changes facts. Things to try: a stricter system prompt
#   ("use only the context, say I don't know otherwise"), a lower temperature on models that allow it
#   (not gpt-5 models), better retrieval so the LLM does not need to guess, or a stronger LLM.
# =====================================================================

# FaithfulnessMetric = the metric that measures how well the answer agrees with the retrieved chunks
# (a related metric that uses a ground-truth context instead: HallucinationMetric)
from deepeval.metrics import FaithfulnessMetric
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
rag_input = "Explain how the k-Nearest Neighbors algorithm works, including how the value of k affects performance."

# the answer to judge: in the real pipeline, this is what your RAG app produced (final_state["response"]).
# Here it is written by hand, with two deliberate problems:
#   - the "small k is more robust to noise" sentence contradicts the context
#   - the "O(n log n) training time" sentence adds a fact that the context does not contain
actual_output = """k-Nearest Neighbors is a non-parametric algorithm that can be used for classification and regression tasks. It predicts the label of a query point by looking at the k closest training points, usually measured using Euclidean distance, and taking a majority vote or average. Since kNN doesn't learn an explicit model during training and just stores the data, it's often referred to as a lazy learner. Choosing a very small value of k typically makes the model more robust to noisy data. Feature scaling matters a lot for kNN since it relies on distance calculations. kNN has a training time complexity of O(n log n) due to the tree structures used internally."""


# the chunks the retriever returned: a list of TEXT strings, one string per chunk
# (in the real pipeline this is [doc.page_content for doc in retrieved_docs])
# These are the "source of truth" for this metric. The answer is judged only against them.
retrieval_context = [
    "k-Nearest Neighbors is a non-parametric, instance-based learning algorithm used for both classification and regression.",
    "It works by finding the k closest training examples to a query point, typically using Euclidean distance, and predicting based on majority vote (for classification) or averaging (for regression).",
    "The algorithm does not build an explicit model during training — it simply stores the training data, which is why it is called a 'lazy learner.'",
    "The choice of k significantly affects performance: a small k makes the model sensitive to noise, while a large k can smooth out decision boundaries too much.",
    "Feature scaling is important for kNN because distance calculations are sensitive to the magnitude of feature values."
]



# =====================================================================
# STEP 2: THE TEST CASE
# =====================================================================

# test case
# Arguments:
#   input             -> the question
#   retrieval_context -> list of the retrieved chunk texts (the facts the answer must stay within)
#   actual_output     -> the answer to judge
# There is no expected_output here on purpose: faithfulness never uses an ideal answer.
# Extra arguments you can add (ignored by this metric, used by others):
#   expected_output="..."      -> the ideal answer
#   context=[...]              -> the ideal source text (ground truth), used by the hallucination metric
#   name="knn case"            -> a label for the test case
#   additional_metadata={}     -> any extra info you want to keep with the case
test_case = LLMTestCase(
    input=rag_input,
    retrieval_context=retrieval_context,
    actual_output=actual_output
)


# =====================================================================
# STEP 3: THE METRIC
# =====================================================================

# metric
# Arguments used here:
#   verbose_mode=True               -> print every step the judge takes (truths, claims, verdicts,
#                                      reasons, final score) to the console. Default False.
#                                      Use True while learning or debugging.
#   penalize_ambiguous_claims=False -> what to do with "idk" claims (true-sounding facts that are NOT in the context).
#                                      False (the default): "idk" claims count as faithful, so only
#                                      contradictions lower the score. This is lenient.
#                                      True: "idk" claims count against the answer too. Use True when you
#                                      want to catch invented extra facts, like the O(n log n) claim above.
#   truths_extraction_limit=7       -> the most facts ("truths") the judge extracts from the retrieval context.
#                                      Default None = no limit. A limit saves cost and time on long contexts.
#                                      The judge keeps the most important facts, so a limit that is too LOW
#                                      can drop facts, and a correct claim then looks unsupported.
#                                      The 5 chunks above hold more than 7 small facts, so 7 may cut some.
# Extra arguments you can add:
#   threshold=0.5       -> pass mark between 0 and 1. success is True when score >= threshold. Default 0.5.
#                          Use a higher value (0.8 to 0.9) if hallucinations are not acceptable.
#   model="gpt-4.1"     -> the judge LLM. A model name string, or your own class based on DeepEvalBaseLLM
#                          (for other providers). Default: an OpenAI model that depends on your DeepEval
#                          version. A stronger judge gives more reliable scores; a cheaper one saves money.
#   include_reason=True -> also write a short explanation of the score, including which claims were
#                          contradicted (default True). False skips that LLM call, which is cheaper and faster.
#   strict_mode=False   -> True makes the score all-or-nothing: 1 if every claim is supported, otherwise 0
#                          (the threshold is forced to 1).
#   async_mode=True     -> run the judge calls in parallel (faster). Default True.
faithfulness = FaithfulnessMetric(verbose_mode=True,
                                  penalize_ambiguous_claims=False,
                                  truths_extraction_limit=7)


# =====================================================================
# STEP 4: RUN THE METRIC
# =====================================================================

# test the metric
# measure(test_case) -> runs the judge steps (a to e in the flow above) and stores the results
#                       inside the metric object. It uses several LLM calls (truths, claims, verdicts,
#                       reason), so it costs a little more than the simpler metrics.
# (a_measure(test_case) is the async version.)
faithfulness.measure(test_case=test_case)

# The script above prints the steps because verbose_mode=True. To read the results yourself, add:
#   print(faithfulness.score)            # the number between 0 and 1
#   print(faithfulness.reason)           # the judge's explanation (names the contradicting claims)
#   print(faithfulness.is_successful())  # True if score >= threshold
#
# TRY THIS: change penalize_ambiguous_claims to True and run again. If the judge marked the
# O(n log n) claim as "idk", the score drops, because that invented fact now counts against the answer.
#
# Other ways to run the same metric:
#   from deepeval import evaluate
#   evaluate(test_cases=[test_case], metrics=[faithfulness])  -> runs many test cases and metrics and prints a table
#   from deepeval import assert_test
#   assert_test(test_case, [faithfulness])                    -> use inside a pytest function (run with: deepeval test run <file>)
#
# Run this file with:  uv run python test_faithfulness_metric.py