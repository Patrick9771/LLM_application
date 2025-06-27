# llm_router.py

from config import *
from sparkai.llm.llm import ChatSparkLLM
from sparkai.core.messages import ChatMessage
import requests

# ===== 通义千问 =====
from openai import OpenAI as QwenClient

QWEN_CLIENT = QwenClient(
    api_key=os.getenv("QWEN_API_KEY") or "sk-f8c157427a204f498f146f2ad401a804",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def call_qwen(prompt: str) -> str:
    response = QWEN_CLIENT.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        # 若调用 Qwen3 模型且非流式，请开启此选项：
        # extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content

def call_llm(model_name: str, prompt: str) -> str:
    if model_name == "deepseek-chat":
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_CHAT_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": DEEPSEEK_CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    elif model_name == "deepseek-reasoner":
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_REASONER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": DEEPSEEK_REASONER_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    elif model_name == "spark":
        spark = ChatSparkLLM(
            spark_api_url=SPARKAI_URL,
            spark_app_id=SPARKAI_APP_ID,
            spark_api_key=SPARKAI_API_KEY,
            spark_api_secret=SPARKAI_API_SECRET,
            spark_llm_domain=SPARKAI_DOMAIN,
            streaming=False,
        )
        messages = [ChatMessage(role="user", content=prompt)]
        result = spark.generate([messages])
        return result.generations[0][0].text.strip()

    else:
        raise ValueError(f"未知模型: {model_name}")