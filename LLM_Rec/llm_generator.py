import requests
from config import DEEPSEEK_CHAT_API_KEY, DEEPSEEK_CHAT_MODEL
import json

# 加载推荐候选电影
with open("D:/测试/pythonProject1/data/movie_titles.json", "r", encoding="utf-8") as f:
    candidate_movies = json.load(f)
# 构造可读的候选电影字符串
MOVIE_LIST_TEXT = "\n".join([f"- {title}" for title in candidate_movies])

def generate_recommendation(user_profile: str) -> str:
    # 强化提示，明确要求LLM返回推荐数量
    prompt = f"""
    Based on the user profile: {user_profile}, recommend 3 movies and provide explanations for each recommendation. 
    CThere is no limit on the year of the film, but it must be recommended from the following list of candidate movies:
    {MOVIE_LIST_TEXT}
    
    Each explanation count should be controlled within 50 words.
    Please format your response as follows:
    
    1. **Movie Title 1 (the year)**  
       - **Why?** Explanation...

    2. **Movie Title 2 (the year)**  
       - **Why?** Explanation...
    """
    # 假设 DeepSeek API 的请求 URL
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
    result = response.json()["choices"][0]["message"]["content"]
    print("evaluate_generator 返回结果:", result)  # 打印实际响应内
    return result