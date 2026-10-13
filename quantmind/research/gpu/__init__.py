"""quantmind.research.gpu — T-030 日频 GPU 预测模型的研究工具（不进生产链路，不写 registry）.

- folds：与主线 B 方案 C 逐折对齐的外层折 + 每折内层验证（交易日索引空间，标签成熟约束）
- transforms：当日截面 gauss-rank、市场级特征的训练窗 z-score
- losses：同日 IC 损失、ListNet（一个 batch = 若干完整交易日）
- batching：按交易日分组的 batch
- metrics：逐日 RankIC、移动块自助法（与 T-028 eval 相同的实现）、pinball
- runtime：显存上限（防 WSL 把显存溢出到共享内存）、原子 checkpoint、完成单元台账
"""
