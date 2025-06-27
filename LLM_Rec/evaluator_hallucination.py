# evaluator_hallucination.py
import requests
from config import DEEPSEEK_REASONER_API_KEY, DEEPSEEK_REASONER_MODEL
from llm_router import call_llm


def evaluate_hallucination(recommendation_text, model_name="deepseek-reasoner", logic=None, subjective=None):
    prompt = f"""
    You are checking for hallucinations in a movie recommendation explanation.
    The recommended text is as follows: {recommendation_text}
    The subjective evaluation of an evaluator is as follows: {subjective}.
    The logic evaluation of an other evaluator is as follows: {logic}.

    Generate the ratings for each film in sequence, and the rating criteria are as follows:

    1. **Hallucination-Risk**: Does the explanation contain factual inaccuracies, fabricated details, or unsupported claims about the movie?
    2. **Explanatory Validity**: Does the explanation provide a clear and reasonable justification for the recommendation, grounded in reality?

    The word count should be controlled within 50 words.Only output the content in the following format. No additional content needs to be output.
    Please return the output result in JSON array format.   Each item has the following structure:
    {{
    "Movie": "Name of the film",
    "Explanation": "Reason",
    "Hallucination-Risk": Scores (integers 0 to 5, scale where 0 = no hallucination, 5 = severe hallucination),
    "Explanatory Validity": Scores (integers 0 to 5, scale where 0 = Ineffective, 5 = very effective)
    }}
    """

    result = call_llm(prompt=prompt, model_name=model_name)
    print("evaluate_hallucination 返回结果:", result)  # 添加调试信息
    return result