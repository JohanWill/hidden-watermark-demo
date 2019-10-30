import numpy as np

# 假设这是原始水印
original_ls = np.array([100,200,300,400,500])
shuffled_ls = original_ls.copy() # 拷贝一份等待shuffle

# 第一次打乱以后
random_ls = np.random.RandomState(15)
random_ls.shuffle(shuffled_ls)
# print(shuffled_ls) # [300 400 500 200 100]

# 原始索引顺序
index = np.arange(5)
# print(index)# [0 1 2 3 4]

# 经历相同的一次打乱
random_index = np.random.RandomState(15)
random_index.shuffle(index)
# print(index) # [2 3 4 1 0]


# 将打乱的水印还原
shuffled_ls[index] = shuffled_ls.copy()
print(shuffled_ls)

