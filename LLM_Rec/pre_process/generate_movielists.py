import pandas as pd
import json

# 加载电影数据
MOVIES_FILE = "D:/测试/pythonProject1/ml-1m/movies.dat"
movies = pd.read_csv(MOVIES_FILE, sep="::", engine="python",
                     names=["MovieID", "Title", "Genres"],
                     encoding="latin-1")

# 提取电影标题
movie_titles = movies["Title"].tolist()

# 保存为JSON文件
with open('D:/测试/pythonProject1/data/movie_titles.json', 'w', encoding='utf-8') as f:
    json.dump(movie_titles, f, ensure_ascii=False, indent=2)

print("✅ 成功保存电影标题到 data/movie_titles.json")