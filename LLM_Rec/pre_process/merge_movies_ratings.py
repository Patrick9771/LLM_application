import pandas as pd

# 加载数据
RATINGS_FILE = "D:/测试/pythonProject1/ml-1m/ratings.dat"
MOVIES_FILE = "D:/测试/pythonProject1/ml-1m/movies.dat"
MERGED_FILE = "D:/测试/pythonProject1/ml-1m/merged_ratings_movies.csv"

ratings = pd.read_csv(RATINGS_FILE, sep="::", engine="python",
                      names=["UserID", "MovieID", "Rating", "Timestamp"],
                      encoding="latin-1")
movies = pd.read_csv(MOVIES_FILE, sep="::", engine="python",
                     names=["MovieID", "Title", "Genres"],
                     encoding="latin-1")

# 合并评分与电影数据
merged = pd.merge(ratings, movies, on="MovieID")

# 去掉时间戳列
merged = merged.drop(columns=["Timestamp"])

# 保存合并后的数据
merged.to_csv(MERGED_FILE, index=False)

print(f"✅ 成功合并数据并保存至 {MERGED_FILE}")