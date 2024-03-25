import ray

# 初始化 Ray
ray.init()

# 获取并打印 Ray 可用资源信息
resources = ray.available_resources()
for i in resources:
    print(i,':',resources[i])

# 检查 GPU 是否可用
if "GPU" in resources:
    print(f"Ray detected {resources['GPU']} GPUs.")
else:
    print("Ray did not detect any GPUs.")
