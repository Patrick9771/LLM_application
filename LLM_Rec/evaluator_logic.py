# evaluator_logic.py
import requests
import json

from llm_router import call_llm


def evaluate_logic(recommendation_text, model_name="spark", subjective=None):
    prompt = f"""
    You are evaluating the logical consistency of a recommendation explanation.
    The recommended text is as follows: {recommendation_text}
    The subjective evaluation of an evaluator is as follows: {subjective}.

    Generate the ratings for each film in sequence, and the rating criteria are as follows:

    1. Does the explanation accurately reflect the movie's actual genre, themes, or story?
    2. Is the explanation logically structured, coherent, and easy to follow?
    3. Does the explanation align with the subjective scores provided?    For example, if the subjective rating is high, is the justification strong?

    Rate each aspect from 1 to 5, where 1 = strongly disagree and 5 = strongly agree.

    The word count should be controlled within 100 words.
    Please return the output result in JSON array format.      Each item has the following structure:
    {{
    "Movie": "Name of the film",
    "Explanation": "Reason",
    "Content-Matching": Scores (integers 1 to 5),
    "Logic-Clarity": Scores (integers 1 to 5)
    }}
    """

    result = call_llm(prompt=prompt, model_name=model_name)
    print("evaluate_logic 返回结果:", result)  # 添加调试信息
    return result