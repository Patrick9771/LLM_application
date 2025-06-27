import json
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluator_hallucination import evaluate_hallucination
from evaluator_logic import evaluate_logic
from evaluator_subjective import evaluate_subjective
from llm_generator import generate_recommendation
from reward_utils import compute_reward

# 路径设置
MERGED_FILE = "D:/测试/pythonProject1/ml-1m/merged_ratings_movies.csv"
RESULT_PATH = "D:/测试/pythonProject1/data/result.json"
REWARD_PATH = "D:/测试/pythonProject1/data/rewards.json"
os.makedirs("D:/测试/pythonProject1/data", exist_ok=True)

# JSON 序列化处理器
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 标准化电影标题
def normalize_title(title):
    return re.sub(r"[^\w\s]", "", title.lower().strip()) if isinstance(title, str) else ""

# 加载数据
merged_data = pd.read_csv(MERGED_FILE)
merged_data['Title_Normalized'] = merged_data['Title'].apply(normalize_title)

with open("data/user_profiles.json", "r", encoding="utf-8") as f:
    user_profiles = json.load(f)

user_profiles = user_profiles[5:7]  # 用于测试，仅第5个用户

final_results = []
matched_ratings = []
rewards = []

# 步骤 1: 生成推荐与评估
for profile in user_profiles:
    uid = profile["uid"]
    desc = profile["description"]
    recommendation = generate_recommendation(desc)

    movies = re.findall(r"\*\*(.+?)\*\*", recommendation)
    subjective = evaluate_subjective(desc, recommendation)
    logic = evaluate_logic(recommendation, "spark", subjective)
    hallucination = evaluate_hallucination(recommendation, "deepseek-reasoner", logic, subjective)

    final_results.append({
        "user": profile,
        "rec": recommendation,
        "recommendations": movies,
        "subjective_result": subjective,
        "logic_result": logic,
        "hallucination_result": hallucination
    })

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False, cls=NpEncoder)
print("✅ 已生成 result.json")

# 步骤 2: 匹配用户评分
for result in final_results:
    uid = result["user"]["uid"]
    for movie in result["recommendations"]:
        norm_title = normalize_title(movie)
        match = merged_data[merged_data["Title_Normalized"] == norm_title]
        if not match.empty:
            row = match.iloc[0]
            movie_id = int(row["MovieID"])
            rating_row = match[match["UserID"] == int(uid)]
            if not rating_row.empty:
                rating = int(rating_row["Rating"].values[0])
                matched_ratings.append({
                    "user_id": uid,
                    "movie": movie,
                    "movie_id": movie_id,
                    "rating": rating
                })

# 步骤 3: 计算奖励
for r in matched_ratings:
    uid = r["user_id"]
    movie_name = r["movie"]
    norm_movie = normalize_title(movie_name)

    result = next((res for res in final_results if res["user"]["uid"] == uid), None)
    if not result:
        continue

    try:
        subj_raw = result["subjective_result"].replace("```json", "").replace("```", "").strip()
        logic_raw = result["logic_result"].replace("```json", "").replace("```", "").strip()
        halluc_raw = result["hallucination_result"].replace("```json", "").replace("```", "").strip()

        subj = json.loads(subj_raw)
        logic = json.loads(logic_raw)
        halluc = json.loads(halluc_raw)
    except Exception as e:
        print(f"[!] 解析JSON失败: {e}")
        continue

    subj_match = next((item for item in subj if normalize_title(item.get("Movie", "")) == norm_movie), None)
    logic_match = next((item for item in logic if normalize_title(item.get("Movie", "")) == norm_movie), None)

    if subj_match and logic_match:
        reward = compute_reward(
            subjective_scores=subj_match,
            logic_result=logic_match,
            hallucination_result=halluc,
            verbose=True
        )
        rewards.append({
            "user_id": uid,
            "movie_id": r["movie_id"],
            "movie_name": movie_name,
            "reward": reward,
            "user_rating": r["rating"]
        })

with open(REWARD_PATH, "w", encoding="utf-8") as f:
    json.dump(rewards, f, indent=2, ensure_ascii=False, cls=NpEncoder)
print("✅ 已生成 rewards.json")

# 步骤 4: 打印对比
print("\n开始对比LLM推荐分数与用户真实评分...")
for r in matched_ratings:
    for rw in rewards:
        if r["user_id"] == rw["user_id"] and r["movie_id"] == rw["movie_id"]:
            print(f"用户 {r['user_id']}，电影ID {r['movie_id']}：用户评分 {r['rating']}，LLM推荐分数 {rw['reward']:.2f}")


# 步骤 5: 计算 MAE / STD 并可视化对比
movie_names = [r["movie_id"] for r in rewards]
user_scores = [r["user_rating"] for r in rewards]
llm_scores = [r["reward"] for r in rewards]

mae = np.mean([abs(u - l) for u, l in zip(user_scores, llm_scores)])
std = np.std([u - l for u, l in zip(user_scores, llm_scores)])

print(f"\n📈 MAE (平均绝对误差): {mae:.3f}")
print(f"📊 STD (标准差): {std:.3f}")

# 可视化
x = np.arange(len(movie_names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width/2, user_scores, width, label='User Score')
rects2 = ax.bar(x + width/2, llm_scores, width, label='LLM Score')

ax.set_ylabel('Score')
ax.set_title('LLM vs User Scores by Movie')
ax.set_xticks(x)
ax.set_xticklabels(movie_names, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.show()