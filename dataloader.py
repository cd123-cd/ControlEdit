from datasets import load_dataset

# 加载整个数据集
dataset = load_dataset('chengzhiyuan/onlyclothe', cache_dir='/root/autodl-tmp/cache')

# 查看数据集信息
print(dataset)


# 获取数据集的数据，假设整个数据集作为一个数据集
data = dataset['train']

# 打印数据示例
print(data[0])