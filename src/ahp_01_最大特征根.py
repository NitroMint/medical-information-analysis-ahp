"""
1.	用和积法求下列矩阵的特征向量和最大特征根

A= [[1,8,5,3],
[1/8,1,1/2,1/6],
[1/5,2,1,1/3],
[1/3,6,3,1]]

程序描述：
    用和积法（列归一化 → 行求和 → 归一化）求判断矩阵的特征向量 W
    与最大特征根 lambda_max，公式为 lambda_max = mean(AW / W)。
    结果：W = (0.5666, 0.0560, 0.1044, 0.2730)，lambda_max = 4.0666。
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# 构建判断矩阵
A = np.array([
    [1,     8,     5,     3    ],
    [1/8,   1,     1/2,   1/6  ],
    [1/5,   2,     1,     1/3  ],
    [1/3,   6,     3,     1    ],
], dtype=float)

n = A.shape[0]
print("判断矩阵 A:")
print(A)


def ahp_sum_product(A):
    """和积法求特征向量与最大特征根"""
    n = A.shape[0]

    # 第一步：列归一化
    B = A / A.sum(axis=0)

    # 第二步：按行求和
    row_sum = B.sum(axis=1)

    # 第三步：归一化得到特征向量（权重）
    W = row_sum / row_sum.sum()

    # 第四步：求最大特征根
    AW = A @ W
    lambda_max = np.mean(AW / W)

    return W, lambda_max


W, lambda_max = ahp_sum_product(A)

print("\n===== 和积法结果 =====")
print("特征向量（权重） W =")
for i, w in enumerate(W, start=1):
    print(f"  W{i} = {w:.4f}")

print(f"\n最大特征根 lambda_max = {lambda_max:.4f}")
