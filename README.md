# 基于LlamaIndex框架的RAG
模型：internLM
环境配置：
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
在RAG目录下添加.env文件，里面存放所需要的环境变量，示例模板如下：
```
OPENAI_API_KEY="sk-your api key"
OPENAI_API_BASE="https://api.gptsapi.net/v1"
HF_HOME="/root/autodl-tmp/huggingface"
HF_ENDPOINT="https://hf-mirror.com"
LLAMA_CLOUD_API_KEY="llx-your api key"
# DASHSCOPE_API_KEY="sk-your api key"
# OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
```
- HF_HOME为huggingface缓存加载位置
- HF_ENDPOINT为镜像网站，国内服务器可以下载huggingface模型
- DASHSCOPE_API_KEY为[阿里云百炼](https://help.aliyun.com/zh/model-studio/getting-started/)的api
- LLAMA_CLOUD_API_KEY为llamacloud的api，用于indexing，[注册网站](https://cloud.llamaindex.ai/landing)

运行RAG.py，对dataset中指定的`file_type`类型的文件进行RAG问答，问答结果将保存在output目录下。

运行evaluate.py，对output目录下的json文件进行评估，使用了[RAGChecker](https://github.com/amazon-science/RAGChecker)指标，结果保存在result目录下。
