import modal

stub = modal.Stub("llama_index")
stub.volume = modal.Volume.persisted("llama_index_volume") # Beta版だけどまぁ（Commit必要）　https://modal.com/docs/guide/volumes
# volume = modal.NetworkFileSystem.persisted("llama_index-volume") # クソ遅い
model_dir = "/content/model"


@stub.function(
    image=modal.Image.from_dockerhub("python:3.8-slim")
    .pip_install(
        "llama-index==0.8.29.post1",
        "transformers==4.33.2",
        "accelerate==0.23.0",
        "bitsandbytes==0.41.1",
        "sentence_transformers==2.2.2"
    ),
    volumes={model_dir: stub.volume},
    gpu="a10g",
    timeout=6000,
    mounts=[modal.Mount.from_local_dir(
        r"./", remote_path="/root/")]
)
async def run_llama_index():
    from datetime import datetime
    import logging
    import sys

    # ログレベルの設定
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # トークナイザーとモデルの準備
    print(f"{str(datetime.now())} : トークナイザーとモデルの準備-AutoTokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "elyza/ELYZA-japanese-Llama-2-7b-instruct",
        cache_dir=f"{model_dir}/AutoTokenizer"
    )

    print(f"{str(datetime.now())} : トークナイザーとモデルの準備-AutoModelForCausalLM")
    model = AutoModelForCausalLM.from_pretrained(
        "elyza/ELYZA-japanese-Llama-2-7b-instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=f"{model_dir}/AutoModelForCausalLM" # AutoModelForCausalLMに関してはキャッシュしないほうが早いかもしれない・・・
    )

    from transformers import pipeline
    from langchain.llms import HuggingFacePipeline

    # パイプラインの準備
    print(f"{str(datetime.now())} : パイプラインの準備")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    # LLMの準備
    print(f"{str(datetime.now())} : LLMの準備")
    llm = HuggingFacePipeline(pipeline=pipe)

    from langchain.embeddings import HuggingFaceEmbeddings
    from llama_index import LangchainEmbedding
    from typing import Any, List

    # 埋め込みクラスにqueryを付加
    class HuggingFaceQueryEmbeddings(HuggingFaceEmbeddings):
        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return super().embed_documents(["query: " + text for text in texts])

        def embed_query(self, text: str) -> List[float]:
            return super().embed_query("query: " + text)

    # 埋め込みモデルの準備
    print(f"{str(datetime.now())} : 埋め込みモデルの準備")
    embed_model = LangchainEmbedding(
        HuggingFaceQueryEmbeddings(
            model_name="intfloat/multilingual-e5-large", cache_folder=f"{model_dir}/LangchainEmbedding")
    )

    from llama_index import ServiceContext
    from llama_index.text_splitter import SentenceSplitter
    from llama_index.node_parser import SimpleNodeParser

    # ノードパーサーの準備
    print(f"{str(datetime.now())} : ノードパーサーの準備")
    text_splitter = SentenceSplitter(
        chunk_size=500,
        paragraph_separator="\n\n",
        tokenizer=tokenizer.encode
    )
    node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)

    # サービスコンテキストの準備
    print(f"{str(datetime.now())} : サービスコンテキストの準備")
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )

    from llama_index import SimpleDirectoryReader

    # ドキュメントの読み込み
    print(f"{str(datetime.now())} : ドキュメントの読み込み")
    documents = SimpleDirectoryReader(
        input_files=["dataset.txt"]
    ).load_data()

    from llama_index import VectorStoreIndex

    # インデックスの作成
    print(f"{str(datetime.now())} : インデックスの作成")
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
    )

    from llama_index.prompts.prompts import QuestionAnswerPrompt

    # QAテンプレートの準備
    print(f"{str(datetime.now())} : QAテンプレートの準備")
    qa_template = QuestionAnswerPrompt("""<s>[INST] <<SYS>>
    質問に100文字以内で答えだけを回答してください。
    <</SYS>>
    {query_str}

    {context_str} [/INST]
    """)

    # クエリエンジンの作成
    print(f"{str(datetime.now())} : クエリエンジンの作成")
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=qa_template,
    )

    # 入力
    inputs = [
        "主人公の性格は？",
        "主人公の家族構成は？",
        "主人公はどこに住んでる？",
    ]

    # チェーンの実行
    print("========チェーンの実行========")
    for input in inputs:
        print("================")
        print(input)
        print(query_engine.query(input))

    # モデルデータを永続化 TODO モデルデータダウンロード時のみにしたい
    print("モデルデータを永続化")
    stub.volume.commit()

@stub.local_entrypoint()
def main():
    run_llama_index.remote()

# if __name__ == '__main__':
#     main()
