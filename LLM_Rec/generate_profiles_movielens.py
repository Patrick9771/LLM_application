import pandas as pd
import json
from collections import Counter
import os
from datetime import datetime

# 文件路径
RATINGS_FILE = "ml-1m/ratings.dat"  # 使用 ml-1m 数据集
MOVIES_FILE = "ml-1m/movies.dat"
USERS_FILE = "ml-1m/users.dat"
OUTPUT_JSON = "data/user_profiles.json"

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

# 用户画像参数
TOP_K_GENRES = 5  # 每个用户最多取K个偏好类型
MIN_RATINGS = 20  # 最少有多少条评分记录才考虑
MAX_USERS = 200  # 最多生成多少个用户画像
RECENT_MONTHS = 12  # 近几个月的评分权重加倍
HIGH_RATING_THRESHOLD = 4  # 高评分阈值(>=4分)

# 检查文件是否存在
for file in [RATINGS_FILE, MOVIES_FILE, USERS_FILE]:
    if not os.path.exists(file):
        print(f"错误：文件 {file} 不存在！")
        exit(1)

# 加载数据
ratings = pd.read_csv(RATINGS_FILE, sep="::", engine="python",
                      names=["UserID", "MovieID", "Rating", "Timestamp"],
                      encoding="latin-1")
movies = pd.read_csv(MOVIES_FILE, sep="::", engine="python",
                     names=["MovieID", "Title", "Genres"],
                     encoding="latin-1")
users = pd.read_csv(USERS_FILE, sep="::", engine="python",
                    names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
                    encoding="latin-1")

# 数据预处理
# 1. 将时间戳转换为日期
ratings['Date'] = pd.to_datetime(ratings['Timestamp'], unit='s')
# 2. 计算距今的月数
now = datetime.now()
ratings['MonthsAgo'] = ((now.year - ratings['Date'].dt.year) * 12 +
                        (now.month - ratings['Date'].dt.month))

# 合并评分与电影数据
merged = pd.merge(ratings, movies, on="MovieID")
merged = pd.merge(merged, users, on="UserID")

# 年龄映射
AGE_MAP = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+"
}

# 性别映射
GENDER_MAP = {
    'F': '女',
    'M': '男'
}

# 职业映射 (ml-1m数据集中的职业ID到名称的映射)
OCCUPATION_MAP = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
}

# 准备生成画像
user_profiles = []
for uid, group in merged.groupby("UserID"):
    if len(group) < MIN_RATINGS:
        continue

    # 计算加权类型偏好
    genre_scores = Counter()

    for _, row in group.iterrows():
        genres = row["Genres"].split("|")
        rating = row["Rating"]
        months_ago = row["MonthsAgo"]

        # 计算基础分数 (高评分电影权重更高)
        base_score = rating / 5.0

        # 时间衰减：近12个月的评分权重加倍
        time_factor = 2.0 if months_ago <= RECENT_MONTHS else 1.0

        # 最终分数 = 基础分数 × 时间因子
        final_score = base_score * time_factor

        # 累加每个类型的分数
        for genre in genres:
            genre_scores[genre] += final_score

    # 选择分数最高的前K个类型
    top_genres = [g for g, _ in genre_scores.most_common(TOP_K_GENRES)]
    tag_str = "、".join(top_genres)

    # 提取用户信息
    user_info = group.iloc[0]
    gender = GENDER_MAP.get(user_info["Gender"], user_info["Gender"])
    age_group = AGE_MAP.get(user_info["Age"], "Unknown")
    occupation = OCCUPATION_MAP.get(user_info["Occupation"], "Unknown")

    # 计算用户偏好统计信息
    avg_rating = group["Rating"].mean()
    high_rating_ratio = len(group[group["Rating"] >= HIGH_RATING_THRESHOLD]) / len(group)
    recent_movies = group.sort_values("Date", ascending=False).head(3)["Title"].tolist()

    # 构建更丰富的用户画像
    user_profiles.append({
        "uid": str(uid),
        "gender": gender,
        "age_group": age_group,
        "occupation": occupation,
        "interests": top_genres,
        "stats": {
            "total_ratings": len(group),
            "avg_rating": round(avg_rating, 2),
            "high_rating_ratio": f"{high_rating_ratio:.0%}"
        },
        "recent_movies": recent_movies,
        "description": (f"{uid}号用户（{gender}, {age_group}, {occupation}）"
                        f"是个电影爱好者（平均评分{avg_rating:.1f}分，{high_rating_ratio:.0%}的电影评分为4星及以上）"
                        f"，最近观看的电影包括《{', '.join(recent_movies)}》。"
                        f"偏好题材包括：{tag_str}")
    })

    if len(user_profiles) >= MAX_USERS:
        break

# 保存为 JSON 文件
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(user_profiles, f, ensure_ascii=False, indent=2)

print(f"✅ 成功生成 {len(user_profiles)} 条用户画像，保存至 {OUTPUT_JSON}")