import json
import re


def calculate_stage_score(scores, weights=None, default_score=3.0):
    """
    计算某个评估阶段的综合得分

    参数:
    scores (dict): 包含该阶段各项分数的字典
    weights (dict): 各项分数的权重，默认为等权重
    default_score (float): 当分数缺失时使用的默认值

    返回:
    float: 该阶段的综合得分
    """
    if not scores:
        return default_score

    # 如果没有提供权重，使用等权重
    if weights is None:
        weights = {k: 1.0 for k in scores}

    total_weight = sum(weights.values())
    weighted_sum = 0.0

    for key, score in scores.items():
        # 处理分数缺失或无效的情况
        if score is None:
            score = default_score

        # 确保分数在有效范围内
        try:
            score = float(score)
            score = max(0.0, min(5.0, score))
        except (ValueError, TypeError):
            score = default_score

        # 累加加权分数
        weighted_sum += score * weights.get(key, 1.0)

    # 计算加权平均
    return weighted_sum / total_weight


# reward_utils.py
def compute_reward(subjective_scores, logic_result, hallucination_result, verbose=False):
    """计算综合奖励分数，增强数据类型验证和字段过滤"""
    stages = {}

    # 定义评估指标的有效键
    VALID_SUBJECTIVE_KEYS = {"Relevance", "Clarity", "Persuasiveness",
                             "relevance", "clarity", "persuasiveness","score"}
    VALID_LOGIC_KEYS = {"Logic-Clarity", "Content-Matching", "logic-clarity", "content-matching", "score"}
    VALID_HALLUCINATION_KEYS = {"Hallucination-Risk", "Explanatory Validity",
                                "hallucination-risk", "explanatory validity", "score"}

    # 处理主观评估分数
    subjective_avg = 3.0  # 默认值
    valid_scores = []
    if isinstance(subjective_scores, list):
        for movie_info in subjective_scores:
            for key, value in movie_info.items():
                # 只处理预定义的有效指标键
                if key in VALID_SUBJECTIVE_KEYS:
                    try:
                        valid_scores.append(float(value))
                    except (ValueError, TypeError) as e:
                        if verbose:
                            print(f"警告: 主观评估中键 '{key}' 的值 '{value}' 无法转换为数值，忽略该值")
    elif isinstance(subjective_scores, dict):
        for key, value in subjective_scores.items():
            # 只处理预定义的有效指标键
            if key in VALID_SUBJECTIVE_KEYS:
                try:
                    valid_scores.append(float(value))
                except (ValueError, TypeError) as e:
                    if verbose:
                        print(f"警告: 主观评估中键 '{key}' 的值 '{value}' 无法转换为数值，忽略该值")

    if valid_scores:
        subjective_avg = sum(valid_scores) / len(valid_scores)

    stages["subjective"] = max(1.0, min(5.0, subjective_avg))  # 确保在1-5范围内

    # 处理逻辑评估分数
    logic_score = 3.0  # 默认值
    if logic_result is not None:
        try:
            if isinstance(logic_result, str):
                try:
                    logic_json = json.loads(logic_result)
                except json.JSONDecodeError:
                    logic_json = {}

            elif isinstance(logic_result, dict):
                logic_json = logic_result

            scores = []
            if isinstance(logic_json, dict):
                for key, val in logic_json.items():
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            if sub_key in VALID_LOGIC_KEYS:
                                try:
                                    scores.append(float(sub_val))
                                except:
                                    pass
                    elif key in VALID_LOGIC_KEYS:
                        try:
                            scores.append(float(val))
                        except:
                            pass

            if scores:
                logic_score = sum(scores) / len(scores)

        except Exception as e:
            if verbose:
                print(f"解析逻辑评估结果时出错: {e}")
    stages["logic"] = max(1.0, min(5.0, logic_score))  # 确保在1-5范围内

    # 处理幻觉评估分数
    # 处理幻觉评估分数
    hallucination_score = 3.0  # 默认值
    try:
        if isinstance(hallucination_result, str):
            try:
                hallucination_result = json.loads(hallucination_result)
            except json.JSONDecodeError:
                hallucination_result = {}

        hallucination_risk_scores = []
        explanatory_validity_scores = []

        if isinstance(hallucination_result, list):
            for item in hallucination_result:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if key in VALID_HALLUCINATION_KEYS:
                            try:
                                val = float(value)
                                if key.lower() in {"hallucination-risk", "hallucination_risk"}:
                                    hallucination_risk_scores.append(1.0 - min(max(val, 0.0), 5.0) / 5.0)  # 越小越好 → 越大越好
                                elif key.lower() in {"explanatory validity", "explanatory_validity"}:
                                    explanatory_validity_scores.append(min(max(val, 0.0), 5.0) / 5.0)  # 越大越好
                            except:
                                pass

        elif isinstance(hallucination_result, dict):
            for key, value in hallucination_result.items():
                if key in VALID_HALLUCINATION_KEYS:
                    try:
                        val = float(value)
                        if key.lower() in {"hallucination-risk", "hallucination_risk"}:
                            hallucination_risk_scores.append(1.0 - min(max(val, 0.0), 5.0) / 5.0)
                        elif key.lower() in {"explanatory validity", "explanatory_validity"}:
                            explanatory_validity_scores.append(min(max(val, 0.0), 5.0) / 5.0)
                    except:
                        pass

        final_scores = hallucination_risk_scores + explanatory_validity_scores
        if final_scores:
            hallucination_score = sum(final_scores) / len(final_scores) * 5.0  # 还原回1~5区间

    except Exception as e:
        if verbose:
            print(f"解析幻觉评估结果时出错: {e}")
    stages["hallucination"] = max(1.0, min(5.0, hallucination_score))  # 保证范围

    # 计算综合奖励
    reward = (
            stages["subjective"] * 0.4 +
            stages["logic"] * 0.3 +
            stages["hallucination"] * 0.3
    )

    final_reward = max(1.0, min(5.0, reward))

    if verbose:
        print(f"\n=== 奖励计算详情 ===")
        print(f"主观评估: {stages['subjective']:.2f} (权重: 40%)")
        print(f"逻辑评估: {stages['logic']:.2f} (权重: 30%)")
        print(f"幻觉评估: {stages['hallucination']:.2f} (权重: 30%)")
        print(f"加权综合分数: {reward:.2f}")
        print(f"最终奖励分数: {final_reward:.2f}\n")

    return final_reward