import json

# 简单示例：更新评估权重
def update_model(user_description, recommendation, manual_score, reasons):
    # 读取当前权重
    try:
        with open('weights.json', 'r', encoding='utf-8') as f:
            weights = json.load(f)
    except FileNotFoundError:
        # 如果文件不存在，初始化权重
        weights = {
            "alpha": 0.4,  # 主观满意度权重
            "beta": 0.3,   # 逻辑一致性权重
            "gamma": 0.3   # 幻觉风险反向权重
        }

    # 根据用户反馈简单调整权重
    if "类别错误" in reasons:
        weights["alpha"] += 0.1  # 增加主观满意度权重
        weights["beta"] -= 0.05
        weights["gamma"] -= 0.05
    elif "逻辑错误" in reasons:
        weights["beta"] += 0.1
        weights["alpha"] -= 0.05
        weights["gamma"] -= 0.05
    elif "存在幻觉" in reasons:
        weights["gamma"] += 0.1
        weights["alpha"] -= 0.05
        weights["beta"] -= 0.05

    # 确保权重在合理范围内
    for key in weights:
        if weights[key] < 0:
            weights[key] = 0
        elif weights[key] > 1:
            weights[key] = 1

    # 保存更新后的权重
    with open('weights.json', 'w', encoding='utf-8') as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)

    print("模型评估权重已更新:", weights)