import torch
import torch.nn as nn

# 定义参数
vocab_size = 128 * 1000  # 词汇表大小
intermediate_size = 1560  # 中间隐藏层大小
batch_size = 1
seq_len = 4096  # 假设序列长度

# 创建一个随机隐层输出张量 (batch_size, seq_len, intermediate_size)
hidden_output = torch.randn(batch_size, seq_len, intermediate_size).cuda()

# 创建最后一层线性层，将隐层大小映射到词汇表大小
# linear_layer = nn.Linear(intermediate_size, vocab_size).cuda()

# 对每个时间步进行最后一层的计算，可以使用 reshape
# 如果需要按照 seq_len 进行计算，可以使用 batched 为 (batch_size * seq_len, intermediate_size)
# logits = linear_layer(hidden_output.view(-1, intermediate_size))  # 变形为 (batch_size * seq_len, intermediate_size)

# 直接构造权重矩阵 W (intermediate_size, vocab_size)
weight_matrix = torch.randn(intermediate_size, vocab_size).cuda()
# 计算 logits，进行矩阵乘法
# 对每个时间步进行最后一层的计算，可以使用 reshape
logits = torch.matmul(hidden_output.view(-1, intermediate_size), weight_matrix)


# 重新调整 logits 的形状为 (batch_size, seq_len, vocab_size)
logits = logits.view(batch_size, seq_len, vocab_size)

# 计算 softmax 以得到概率分布
probabilities = nn.functional.softmax(logits, dim=-1)  # 计算每个时间步的概率分布

del logits, probabilities, weight_matrix, hidden_output
torch.cuda.empty_cache()
