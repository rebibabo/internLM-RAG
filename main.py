import pandas as pd
import json
import os
from loguru import logger
from RAG import RAG

def load_dataset(file_id: int) -> tuple[list[str], list[str]]:
    questions, gt_answers = [], []
    df = pd.read_excel('dataset/Q&A.xlsx')
    for index, row in df.iterrows():
        if index == 0 or int(row[0][:2]) != file_id:
            continue
        _, _, question, gt_answer = row
        questions.append(question)
        gt_answers.append(gt_answer)
    return questions, gt_answers

if __name__ == "__main__":
    file_type = "txt"

    fourbit_models = [
        # "unsloth/Phi-3.5-mini-instruct",
        # "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        # "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        # "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        # "unsloth/mistral-7b-v0.3-bnb-4bit",       
        # "unsloth/Qwen2.5-7B-bnb-4bit", 
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",    
        # "unsloth/gemma-2-9b-bnb-4bit",
    ]

    # rag = RAG(model="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", huggingface=True, n=2)
    rag = RAG(model="gpt-4o-mini", n=2, huggingface=False)
    for file_id in range(1, 11):
        output_dir = os.path.join('output', file_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = f'{output_dir}/{file_id}.json'
        if os.path.exists(output_path):
            continue
        logger.info(f"Processing {file_type} file {file_id}...")
        questions, gt_answers = load_dataset(file_id)
        for file in os.listdir(f'dataset/{file_type}'):
            if file.startswith(f'{file_id:02d}'):
                src_path = f'dataset/{file_type}/{file}'
                break
        rag.indexing(result_type="text", src_path=src_path)
        for question, gt_answer in zip(questions, gt_answers):
            rag.query(question, gt_answer)
            output = rag.output
            print(json.dumps(output, indent=4, ensure_ascii=False))

        with open(output_path, 'w') as f:
            json.dump({"results": rag.output_jsons}, f, indent=4, ensure_ascii=False)
