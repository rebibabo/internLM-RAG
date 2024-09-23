import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()
import os
from loguru import logger
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import (PromptTemplate, Settings, 
                              VectorStoreIndex, StorageContext, 
                              load_index_from_storage)
from transformers import AutoTokenizer, BitsAndBytesConfig
from llama_parse import LlamaParse
from typing import Literal
from copy import deepcopy

import os

class RAG:
    # This will wrap the default prompts that are internal to llama-index
    # taken from https://huggingface.co/Writer/camel-5b-hf
    query_wrapper_prompt = PromptTemplate(
        "下面是一份任务描述。请写出一份【简单扼要】的回答，以完成请求。\n\n"
        "### 任务描述：{query_str}\n\n### 回答："
    )
    query_time: int = 1
    output_json = {
      "query_id": "<query id>", # string
      "query": "<input query>", # string
      "gt_answer": "<ground truth answer>", # string
      "response": "<response generated by the RAG generator>", # string
      "retrieved_context": [ # a list of retrieved chunks by the retriever
        # {
        #   "doc_id": "<doc id>", # string, optional
        #   "text": "<content of the chunk>" # string
        # },
        # ...
      ]
    }
    output_json_cp = deepcopy(output_json)
    output_jsons = []

    def __init__(self, 
            model: str,                  # the name of the model to use
            huggingface: bool = True,    # whether to use huggingface to load the model
            chunk_size: int = 512,       # the maximum number of tokens to generate per request
            chunk_overlap: int = 10,     # the number of tokens to overlap between chunks
            max_length: int = 512,      # the maximum length of the generated text
            temperature: float = 0.25,   # the temperature of the generated text
            n: int = 1,                  # the number of responses to generate
        ) -> None:
        logger.info(f"chunk size: {chunk_size}, overlap: {chunk_overlap}, max length: {max_length}, temperature: {temperature}")
        if huggingface:
            logger.info(f"Using HuggingFace model {model}")
            llm = HuggingFaceLLM(
                context_window=2048,
                max_new_tokens=max_length,
                generate_kwargs={"temperature": temperature, "do_sample": False},
                query_wrapper_prompt=self.query_wrapper_prompt,
                tokenizer_name=model,
                tokenizer_kwargs={"trust_remote_code": True, "max_length":max_length},
                model_name=model,
                model_kwargs={"trust_remote_code": True, "quantization_config": BitsAndBytesConfig(load_in_4bit=True)},
                device_map="auto",
            )
            logger.info(f"Finished loading model!")
        else:    # use openai
            logger.info(f"Using OpenAI model {model}")
            llm = OpenAI(
                model=model,
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=os.getenv("OPENAI_BASE_URL"),
                temperature=temperature,
                max_tokens=max_length,
            )
        self.n = n
        Settings.llm = llm
        self.llm = llm
        text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        Settings.text_splitter = text_splitter

    def indexing(self, 
            result_type: Literal["text", "markdown"],
            src_path: str
        ) -> None:
        logger.info(f"Indexing {src_path}...")
        self.output_jsons.clear()
        self.query_time = 1
        dst_path = os.path.join("storage", src_path.replace('.', '_'))
        if not os.path.exists(dst_path):
            documents = LlamaParse(result_type=result_type).load_data(src_path)
            index = VectorStoreIndex.from_documents(documents, show_progress=True)
            index.storage_context.persist(dst_path)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=dst_path)
            index = load_index_from_storage(storage_context)
        logger.info(f"Finished indexing!")
        self.retriever = index.as_retriever(choice_batch_size=self.n)
        self.query_engine = index.as_query_engine(choice_batch_size=self.n)

    def retrieve(self, question: str) -> None:
        nodes = self.retriever.retrieve(question)
        for node in nodes:
            text = node.text
            doc_id = node.node_id
            self.output_json["retrieved_context"].append({"doc_id": doc_id, "text": text})

    def query(self, 
        question: str,
        gt_answer: str = '',
    ) -> str:
        self.retrieve(question)
        response = self.query_engine.query(question)
        self.output_json["query"] = question
        self.output_json["response"] = response.response.replace('\n', ' ')
        self.output_json["gt_answer"] = gt_answer
        self.output_json["query_id"] = str(self.query_time)
        self.query_time += 1
        self.output_jsons.append(deepcopy(self.output_json))
        self.output_json = deepcopy(self.output_json_cp)
        return response.response

    @ property
    def output(self):
        return self.output_jsons[-1]
