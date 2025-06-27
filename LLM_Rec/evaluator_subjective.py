import requests
import json
from config import QWEN_PLUS_API_KEY, QWEN_PLUS_MODEL
from llm_router import call_llm, call_qwen


def evaluate_subjective(user_profile, recommendation_text):
    prompt = f"""
    用户画像：{user_profile}
    推荐内容：{recommendation_text}
    You are simulating a user on a movie recommendation platform.
    You are shown recommended movie and a brief explanation of why it was recommended to you.

    Generate the ratings for each film in sequence, and the rating criteria are as follows:

    1.    Relevance: Does the explanation match the user's interests?
    2.    Clarity: Is the explanation easy to understand?
    3.    Persuasiveness: Does the explanation make you more likely to watch the movie?

    Rate each aspect from 1 to 5, where 1 = strongly disagree and 5 = strongly agree.

    Try to immerse yourself in this user's experience and provide feedback (no extra text):

    The word count should be controlled within 50 words.
    Please return the output result in JSON array format. Each item has the following structure:
    {{
      "Movie": "Name of the film (the year)",
      "Explanation": "Reason",
      "Relevance": Scores (integers 1 to 5),
      "Clarity": Scores (integers 1 to 5),
      "Persuasiveness": Scores (integers 1 to 5)
    }}
    """

    result = call_qwen(prompt=prompt)
    print("evaluator_subjective 返回结果:", result)
    return result