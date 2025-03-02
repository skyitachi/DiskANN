import numpy as np

# 生成10000个128维的随机向量
num_vectors = 10000
dimension = 128
data = np.random.rand(num_vectors, dimension).astype(np.float32)

target_file = "random_128d_10k_vectors.bin"
# 保存为二进制文件
with open(target_file, "wb") as file_handler:
    file_handler.write(np.array(data.shape, dtype=np.int32).tobytes())
    _ = file_handler.write(data.tobytes())
