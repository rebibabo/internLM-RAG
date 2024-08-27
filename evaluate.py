from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics

import os
import json
from dotenv import load_dotenv
load_dotenv()
from loguru import logger

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

# set-up the evaluator
evaluator = RAGChecker(
    extractor_name="gpt-4o-mini-2024-07-18",
    checker_name="gpt-4o-mini-2024-07-18",
    batch_size_extractor=5,
    batch_size_checker=5,
    openai_api_key=api_key,
    extractor_api_base=api_base,
    checker_api_base=api_base
)

file_type = "txt"
output_dir = f"result/{file_type}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(1, 11):
    if i == 9:
        continue

    output_path = f'{output_dir}/{i}.json'
    if os.path.exists(output_path):
        continue

    # initialize ragresults from json/dict
    with open(f"output/{file_type}/{i}.json") as fp:
        rag_results = RAGResults.from_json(fp.read())

    logger.info(f'Evaluating {file_type}/{i}.json')

    # evaluate results with selected metrics or certain groups, e.g., retriever_metrics, generator_metrics, all_metrics
    try:
        evaluator.evaluate(rag_results, all_metrics)
    except Exception as e:
        logger.critical(f"Evaluation failed for {i}.json")
        logger.exception(e)
        continue

    # save results to file
    with open(output_path, "w") as fp:
        json.dump(rag_results.to_dict(), fp, indent=4, ensure_ascii=False)
    logger.success(f"Evaluation completed for {i}.json")