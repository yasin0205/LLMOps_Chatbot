# =====================================================================
# WHAT THIS SCRIPT DOES
# It builds the TEST DATA SET for the RAG app. A "golden" is one test case:
# a question (input) + the expected answer (expected_output) + the context + actual output (null for now) 
# it came from later you run your RAG graph on each golden question
#
# FLOW OF THE WHOLE SYSTEM
#
#   .env  --load_dotenv()-->  OPENAI_API_KEY ready (DeepEval calls OpenAI models)
#
#   config/parameter_config --> params_config.golden_dataset --> golden_params
#                               (all numbers: thresholds, retries, models, chunk size ...)
#
#   data/processed/*.txt   (your cleaned transcripts)
#      |
#      v
#   PART A: CONTEXT GENERATION   (documents -> contexts)
#      1. load the documents, cut them into chunks (chunk_size tokens), embed, store in a temporary vector store
#      2. randomly sample chunks, find related chunks by cosine similarity
#      3. a critic model scores each chunk (clarity, depth, structure, relevance); weak chunks are dropped
#      4. group related chunks into contexts of 1 to 3 chunks (max 3, matching the retriever's k=3)
#      |
#      v
#   PART B: DATASET GENERATION   (contexts -> goldens), repeated for every context
#      1. Input generation   the LLM writes a candidate question from the context
#      2. Filtration         a critic scores the question (self-contained? clear?)
#                            score >= threshold -> passes
#                            score <  threshold -> try again (up to max_quality_retries)
#                            retries used up    -> the best-scoring candidate goes forward
#      3. Evolution          rewrite the question into a harder or more varied one
#                            (multicontext / concretizing / constrained / comparative),
#                            the type is picked by probability, repeated num_evolutions times
#      4. Styling            optional (tone, format, persona). SKIPPED here: no StylingConfig is passed
#      +  the LLM also writes the expected answer for each question
#      |
#      v
#   save_as(...)  -->  data/evaluation/goldens/<golden_dataset_filename>.json
#      |
#      v
#   (later) graph.invoke on each golden question, then compare with the expected answer
# =====================================================================

# Synthesizer = the engine that creates the goldens (test questions + expected answers)
from deepeval.synthesizer.synthesizer import Synthesizer
# The three config classes control the stages:
#   ContextConstructionConfig -> Part A (how chunks are built into contexts)
#   FiltrationConfig          -> Part B step 2 (how questions are filtered)
#   EvolutionConfig           -> Part B step 3 (how questions are rewritten)
from deepeval.synthesizer.config import EvolutionConfig, FiltrationConfig, ContextConstructionConfig
# Evolution = an enum listing the question-rewriting types
from deepeval.synthesizer.types import Evolution
# load_dotenv reads the .env file (OPENAI_API_KEY lives there)
from dotenv import load_dotenv
# Path gives a clean, cross-platform way to work with files and folders
from pathlib import Path
# your own config object: it holds all the numbers/settings in one place (not part of DeepEval)
#from config.parameter_config import params_config

# load golden dataset params
# params_config.golden_dataset = the "golden dataset" section of your config.
# Every setting below is read from golden_params, so you change numbers in the config, not in the code.
#golden_params = params_config.golden_dataset

# load the api keys
# (arguments of load_dotenv: dotenv_path=".env" to point to a file, override=True to replace
#  existing variables, verbose=True to warn when no .env is found)
load_dotenv()

def generate_golden_dataset():

    # =================================================================
    # SETUP: paths
    # =================================================================

    # define paths
    # __file__ = this script's path. .resolve() makes it absolute.
    # .parent = go up one folder. Three .parent in a row means three folders up.
    # With the file at <root>/src/<folder>/generate_goldens.py, three parents up is <root>.
    # If you move the file to a different depth, ROOT_PATH points to the wrong place.
    ROOT_PATH = Path(__file__).resolve().parent
    # the folder with your cleaned .txt documents (the input)
    DOCS_PATH = ROOT_PATH / "cleaned-transcripts"
    # the folder where the goldens file will be saved (the output)
    GOLDENS_DIR = ROOT_PATH / "datasets"

    # create the directory
    # exist_ok=True -> no error if it already exists; parents=True -> create missing parent folders too
    GOLDENS_DIR.mkdir(exist_ok=True, parents=True)

    # Helper: returns the paths of all .txt files in a folder as a list of strings.
    # DeepEval wants plain strings, so .as_posix() converts each Path to "folder/file.txt".
    # Supported document types in DeepEval: .txt, .docx, .pdf (and .md in newer versions).
    # To load more types, collect several patterns: glob("*.txt"), glob("*.pdf"),
    # or use glob("**/*.txt") to include sub-folders.
    def get_docs_path(directory_path: Path | str) -> list[str]:
        directory_path = Path(directory_path)
        if directory_path.exists() and directory_path.is_dir():
            paths = directory_path.glob("*.txt")
        # BUG: if the folder does not exist, `paths` is never created, and the next line crashes
        # with UnboundLocalError. Safer: raise FileNotFoundError when the folder is missing.
        return [path.as_posix() for path in paths]


    # =================================================================
    # PART A: CONTEXT GENERATION  (documents -> contexts)
    # =================================================================

    # load the config parameters for this part
    #context_construction_params = golden_params.context_construction_config

    # create the config
    # Controls how documents become "contexts" (groups of related chunks).
    # critic_model               -> scores chunk quality (clarity, depth, structure, relevance).
    #                               Default: the Synthesizer's model.
    # max_contexts_per_document  -> upper limit of contexts built from one document. Default 3.
    #                               More = more goldens from each file.
    # context_quality_threshold  -> minimum critic score (0 to 1) for a chunk to be used. Default 0.5.
    #                               Higher = only better chunks, but more chunks rejected.
    # max_retries                -> how many times to try sampling a good chunk before giving up. Default 3.
    # chunk_size                 -> tokens per chunk. Default 1024. Your notebook uses 1024 too,
    #                               so the golden contexts match the chunks your RAG app retrieves.
    # chunk_overlap              -> tokens shared between neighbouring chunks. Default 0 (same as the notebook).
    # Extra arguments you can add (defaults can differ by DeepEval version):
    #   min_contexts_per_document=1      -> lower limit of contexts per file
    #   max_context_length=3             -> max chunks grouped into one context. The default 3 matches your
    #                                       retriever's k=3, so a test context is as big as what the app retrieves.
    #   min_context_length=1             -> min chunks per context
    #   context_similarity_threshold=0.5 -> minimum cosine similarity for chunks to be grouped together
    #                                       (higher = tighter, more related groups)
    #   embedder=...                     -> the model that embeds the chunks (default: an OpenAI embedding model)
    #   encoding=...                     -> the tokenizer used to count chunk_size tokens
    context_config = ContextConstructionConfig(
        critic_model="gpt-5-nano",
        max_contexts_per_document=2,
        context_quality_threshold=0.5,
        max_retries=2,
        chunk_size=200,
        chunk_overlap=10
    )
    # (this config is only created here; it is used in Part B when generate_goldens_from_docs runs,
    #  because that method builds the contexts first and then writes the questions)


    # =================================================================
    # PART B: DATASET GENERATION  (contexts -> goldens)
    # =================================================================

    # load the config parameters for this part
    #filtration_params = golden_params.filtration_config
    #evolution_params = golden_params.evolution_config

    # ---- B2: FILTRATION ----
    # After the LLM writes a candidate question, a critic model scores it
    # (is it self-contained? is it clear?). Low-scoring questions are regenerated.
    # synthetic_input_quality_threshold -> minimum score (0 to 1) to pass. Default 0.5.
    #                                      Higher = stricter and better questions, but more retries and more cost.
    # max_quality_retries               -> how many times to regenerate a failing question. Default 3.
    #                                      After the last retry, the best-scoring candidate goes forward
    #                                      even if it is below the threshold.
    # critic_model                      -> the model that scores the questions.
    #                                      Default: the same model as the Synthesizer. Use a stronger model for stricter scoring.
    #                                      A name like "gpt-4.1" or your own DeepEvalBaseLLM.
    filtration_config = FiltrationConfig(
        synthetic_input_quality_threshold=0.5,
        max_quality_retries=2,
        critic_model="gpt-5-nano"
    )

    # ---- B3: EVOLUTION ----
    # Evolution rewrites a plain question into a harder or more varied one.
    # num_evolutions -> how many rewrite rounds each question goes through. Default 1.
    #                   0 would keep the plain questions. More rounds = harder, but less natural and more cost.
    # evolutions     -> dict {Evolution type: probability}. Each round picks ONE type by these probabilities.
    #                   Probabilities should add up to 1.
    #                   Types used here:
    #                     MULTICONTEXT -> needs information from several chunks of the context
    #                     CONCRETIZING -> makes an abstract question more specific and detailed
    #                     CONSTRAINED  -> adds a condition or restriction
    #                     COMPARATIVE  -> asks to compare things
    #                   Other types you can add: REASONING (multi-step thinking),
    #                     HYPOTHETICAL ("what if ..."), IN_BREADTH (a related but different question).
    #                   A type you leave out is never used.
    # The values come from your config dict, e.g. {"multicontext": 0.25, "concretizing": 0.25, ...}
    evolution_config = EvolutionConfig(
        num_evolutions=2,
        evolutions={
            Evolution.MULTICONTEXT: 0.25,
            Evolution.CONCRETIZING: 0.25,
            Evolution.CONSTRAINED: 0.25,
            Evolution.COMPARATIVE: 0.25
        }
    )

    # ---- create the synthesizer (the engine that runs Part A and Part B) ----
    # model             -> the LLM that writes the questions and expected answers. A name like "gpt-4.1",
    #                      or your own class based on DeepEvalBaseLLM for other providers.
    # filtration_config -> the filtration settings above
    # evolution_config  -> the evolution settings above
    # Extra arguments you can add:
    #   styling_config=StylingConfig(...) -> optional styling step: set the tone, format, scenario or
    #                                        persona of the questions. NOT passed here, so styling is skipped.
    #   async_mode=True                   -> run LLM calls in parallel (faster). Default True.
    #   max_concurrent=20                 -> how many calls run at once (lower it if you hit rate limits)
    #   cost_tracking=True                -> print the estimated cost of the run
    synthesizer = Synthesizer(
        model="gpt-5-nano",
        filtration_config=filtration_config,
        evolution_config=evolution_config
    )

    # ---- generate the goldens (runs Part A, then Part B, end to end) ----
    # generate_goldens_from_docs -> builds contexts from the files (Part A), then writes questions (Part B).
    # document_paths              -> list of file path strings (from get_docs_path)
    # max_goldens_per_context     -> how many questions to write for each context. Default 2.
    #                                Total goldens is about contexts x this number.
    # context_construction_config -> the Part A settings (context_config above)
    # Extra argument:
    #   include_expected_output=True -> also write the expected answer for each question (default True).
    #                                   Keep it True if your metrics compare against an expected answer.
    # Returns the list of goldens. Each has: input (question), expected_output, context, source_file.
    # Other methods with the same pipeline:
    #   generate_goldens_from_contexts(contexts=[[...], ...], ...) -> you give the contexts yourself
    #   generate_goldens_from_scratch(num_goldens=...)             -> no documents, needs a styling_config
    #   generate_goldens_from_goldens(goldens=[...])               -> expand an existing dataset
    
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=get_docs_path(DOCS_PATH),
        max_goldens_per_context=3,
        context_construction_config=context_config)
    # (the `goldens` variable is not used below: save_as saves the goldens the synthesizer
    #  stored internally, so the code works the same without it)

    # ---- save the goldens ----
    # file_type -> "json" (used here) or "csv"
    # directory -> folder to save into (string, so .as_posix())
    # file_name -> name WITHOUT the extension; the extension comes from file_type.
    #              If you leave it out, a timestamp is used.
    # Extra argument: quiet=True -> do not print the save message
    synthesizer.save_as(
        file_type="json",
        directory=GOLDENS_DIR.as_posix(),
        file_name="golden"
    )

# This block runs only when you execute this file directly (python generate_goldens.py),
# not when another file imports it
if __name__ == "__main__":
    generate_golden_dataset()