# Transformer Architecture Notes

## 问题1：在批量样本的transformer架构中，如何统一输入序列的长度，以及统一进行self attention的计算？

回答：未达到最大长度的，用padding补齐，并计算对应的mask向量，用于attention matrix的时候进行对相关无效部分进行mask免除相关计算
因此，编码器解码器架构中src有mask，target也有mask，同时target还有causual mask，最终解码器的mask即为target_mask和causual mask的交集计算。

