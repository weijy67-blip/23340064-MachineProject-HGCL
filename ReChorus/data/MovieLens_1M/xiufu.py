import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random


def convert_movielens_with_negs():
    """将 MovieLens-1M 转换为符合 ReChorus 框架的格式，包含负样本"""

    print("=" * 60)
    print("MovieLens-1M 数据转换 - 包含负样本")
    print("=" * 60)

    # 1. 检查当前目录
    print(f"\n当前目录: {os.getcwd()}")
    files = os.listdir('.')

    # 2. 查找原始数据文件
    ratings_file = None
    movies_file = None

    for f in files:
        if 'rating' in f.lower() and f.endswith('.dat'):
            ratings_file = f
        elif 'movie' in f.lower() and f.endswith('.dat'):
            movies_file = f

    if not ratings_file:
        print("错误: 未找到 ratings.dat 文件!")
        return False

    print(f"\n找到评分文件: {ratings_file}")
    print(f"找到电影文件: {movies_file}")

    # 3. 读取原始数据
    print("\n读取原始数据...")

    try:
        ratings = pd.read_csv(ratings_file, sep='::', engine='python',
                              names=['user_id', 'item_id', 'rating', 'time'])
        print(f"成功读取 {len(ratings)} 条评分记录")

        # 读取电影数据
        if movies_file:
            movies = pd.read_csv(movies_file, sep='::', engine='python',
                                 names=['item_id', 'title', 'genres'], encoding='latin-1')
            print(f"成功读取 {len(movies)} 条电影记录")
        else:
            movies = pd.DataFrame({'item_id': ratings['item_id'].unique()})

    except Exception as e:
        print(f"读取数据失败: {e}")
        return False

    # 4. 转换为隐式反馈（所有交互的评分为1）
    print("\n转换为隐式反馈格式...")

    # 对于隐式反馈，我们将所有交互的评分设为1
    ratings['rating'] = 1.0

    # 5. 获取所有用户和物品的列表
    all_users = ratings['user_id'].unique()
    all_items = ratings['item_id'].unique()
    num_users = len(all_users)
    num_items = len(all_items)

    print(f"唯一用户数: {num_users}")
    print(f"唯一物品数: {num_items}")

    # 6. 为每个用户构建交互物品集合（用于负采样）
    print("\n构建用户交互集合...")
    user_interacted_items = {}
    for user_id in all_users:
        user_items = set(ratings[ratings['user_id'] == user_id]['item_id'].unique())
        user_interacted_items[user_id] = user_items

    # 7. 拆分数据集（按时间顺序）
    print("\n拆分数据集 (按时间顺序)...")

    # 按用户和时间排序
    ratings = ratings.sort_values(by=['user_id', 'time']).reset_index(drop=True)

    # 拆分训练集、验证集、测试集
    train_list, dev_list, test_list = [], [], []

    for user_id in all_users:
        user_data = ratings[ratings['user_id'] == user_id]

        if len(user_data) >= 5:
            # 按时间顺序拆分：80%训练，10%验证，10%测试
            train_size = int(len(user_data) * 0.8)
            dev_size = int(len(user_data) * 0.1)

            train = user_data.iloc[:train_size]
            dev = user_data.iloc[train_size:train_size + dev_size]
            test = user_data.iloc[train_size + dev_size:]

            train_list.append(train)
            dev_list.append(dev)
            test_list.append(test)
        else:
            # 数据太少，全部放入训练集
            train_list.append(user_data)

    # 合并数据
    train_data = pd.concat(train_list, ignore_index=True)
    dev_data = pd.concat(dev_list, ignore_index=True) if dev_list else pd.DataFrame(columns=ratings.columns)
    test_data = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=ratings.columns)

    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(dev_data)}")
    print(f"测试集大小: {len(test_data)}")

    # 8. 创建负样本
    print("\n为训练集创建负样本...")

    # 获取所有物品的列表用于负采样
    all_items_list = list(all_items)

    def create_negative_samples(data, user_interacted_items, num_negatives=1):
        """为数据创建负样本"""
        neg_items_list = []

        for idx, row in data.iterrows():
            user_id = row['user_id']
            interacted_items = user_interacted_items[user_id]

            # 采样负样本
            neg_samples = []
            attempts = 0
            while len(neg_samples) < num_negatives and attempts < 100:
                neg_item = random.choice(all_items_list)
                if neg_item not in interacted_items:
                    neg_samples.append(neg_item)
                attempts += 1

            # 如果找不到足够的负样本，用-1填充
            while len(neg_samples) < num_negatives:
                neg_samples.append(-1)

            neg_items_list.append(neg_samples[0])  # 只取第一个负样本

        return neg_items_list

    # 为训练集创建负样本
    train_data['neg_item_id'] = create_negative_samples(train_data, user_interacted_items, num_negatives=1)

    # 9. 保存文件（制表符分隔）
    print("\n保存文件...")

    # 保存训练集（包含负样本）
    train_data.to_csv('train.csv', sep='\t', index=False)

    # 验证集和测试集不需要负样本（在评估时动态生成）
    dev_data.to_csv('dev.csv', sep='\t', index=False)
    test_data.to_csv('test.csv', sep='\t', index=False)

    # 保存电影元数据
    movies.to_csv('item_meta.csv', sep='\t', index=False)

    # 10. 创建 info.txt
    info_content = f"""dataset=MovieLens_1M
user_num={num_users}
item_num={num_items}
"""

    with open('info.txt', 'w') as f:
        f.write(info_content)

    # 11. 显示生成的文件信息
    print("\n转换完成! 生成的文件:")
    print("  - train.csv (包含正样本和负样本)")
    print("  - dev.csv")
    print("  - test.csv")
    print("  - item_meta.csv")
    print("  - info.txt")

    print(f"\ntrain.csv 列名: {list(train_data.columns)}")
    print(f"train.csv 前几行:")
    print(train_data[['user_id', 'item_id', 'neg_item_id', 'rating', 'time']].head())

    print(f"\ninfo.txt 内容:")
    print(info_content)

    return True


def check_csv_format():
    """检查生成的 CSV 文件格式"""
    print("\n" + "=" * 60)
    print("检查 CSV 文件格式")
    print("=" * 60)

    for file_name in ['train.csv', 'dev.csv', 'test.csv']:
        if os.path.exists(file_name):
            df = pd.read_csv(file_name, sep='\t')
            print(f"\n{file_name}:")
            print(f"  列名: {list(df.columns)}")
            print(f"  形状: {df.shape}")
            if len(df) > 0:
                print(f"  前3行:")
                print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    # 转换数据
    success = convert_movielens_with_negs()

    if success:
        # 检查格式
        check_csv_format()