# OP Summary

# conv

# topK

好的，下面详细讲解 **topK 算子** 的概念、原理、常见用法和注意事项。

---

## 1. 概念

**topK 算子**，即“取前 K 大（或小）元素”的操作。  
在深度学习和数据处理中，常用于从一组数值中选出最大的 K 个（或最小的 K 个）元素及其索引。

---

## 2. 典型应用场景

- **分类任务**：取概率最大的前 K 个类别（Top-1、Top-5 accuracy）
- **检索/推荐**：返回得分最高的 K 个候选项
- **非极大值抑制（NMS）**：选取置信度最高的 K 个候选框
- **注意力机制**：选取权重最大的 K 个位置

---

## 3. 算子原理

- **输入**：一个张量（如 `[N, C]`，N 个样本，每个有 C 个分数）
- **参数**：
  - `k`：要选取的元素个数
  - `dim`：在哪个维度上取 topK
  - `largest`：取最大（True）还是最小（False）
  - `sorted`：输出是否按值排序
- **输出**：
  - `values`：topK 的数值
  - `indices`：topK 的原始索引

---

## 4. 代码示例（以 PyTorch 为例）

### 4.1 基本用法

```python
import torch

x = torch.tensor([[1.2, 3.4, 2.1, 0.5],
                  [0.9, 2.2, 4.1, 1.7]])  # shape [2, 4]

# 取每行最大的2个元素及其索引
values, indices = torch.topk(x, k=2, dim=1, largest=True, sorted=True)
print(values)   # [[3.4, 2.1], [4.1, 2.2]]
print(indices)  # [[1, 2], [2, 1]]
```

### 4.2 取最小的 K 个

```python
values, indices = torch.topk(x, k=2, dim=1, largest=False)
print(values)   # [[0.5, 1.2], [0.9, 1.7]]
print(indices)  # [[3, 0], [0, 3]]
```

---

## 5. 注意事项

- **排序**：`sorted=True` 时，输出按值从大到小（或小到大）排列。
- **维度**：`dim` 决定在哪个轴上取 topK，常见如分类 logits 的最后一维。
- **性能**：topK 算法比完全排序快，常用堆排序/快速选择等。
- **可微性**：topK 的索引操作不可微，但值的选择可用于后续可微分操作。

---

## 6. 相关算子

- **argmax/argmin**：topK 的特例（K=1）
- **sort**：返回全排序结果，topK 只取前 K 个
- **argsort**：返回排序后的索引

---

## 7. 总结

**topK 算子**是深度学习和数据处理中非常常用的操作，能高效地从大批量数据中选出最重要的 K 个元素及其索引，广泛用于分类、检索、注意力等场景。  
常见于 PyTorch（`torch.topk`）、TensorFlow（`tf.nn.top_k`）、NumPy（`np.argpartition`+切片）等主流框架。

# Gather

**gather 算子** 是深度学习框架中常用的“按索引取值”操作。它允许你根据给定的索引，从输入张量的指定维度上，批量地选取元素，输出的形状与索引张量相同。

---

## 1. 基本概念

- **功能**：从输入张量的某个维度上，按照索引张量指定的位置，收集元素。
- **常见接口**（以 PyTorch 为例）：
  ```python
  torch.gather(input, dim, index)
  ```
  - `input`：原始张量
  - `dim`：在哪个维度上 gather
  - `index`：索引张量，形状与输出相同

---

## 2. 例子

### 2.1 二维示例

```python
import torch

x = torch.tensor([[10, 11, 12],
                  [20, 21, 22]])  # shape [2, 3]

idx = torch.tensor([[2, 0, 1],
                    [1, 1, 0]])   # shape [2, 3]

y = torch.gather(x, dim=1, index=idx)
print(y)
# 输出：
# tensor([[12, 10, 11],
#         [21, 21, 20]])
```
解释：  
- 第一行：取 x[0,2], x[0,0], x[0,1] → 12, 10, 11  
- 第二行：取 x[1,1], x[1,1], x[1,0] → 21, 21, 20

### 2.2 分类概率选取

假设有分类概率 `probs`，标签 `target`，想取每个样本的正确类别概率：

```python
probs = torch.tensor([[0.1, 0.7, 0.2],
                      [0.3, 0.4, 0.3]])  # shape [2, 3]
target = torch.tensor([1, 2])            # shape [2]

picked = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
print(picked)  # tensor([0.7, 0.3])
```

---

## 3. 常见用途

- 分类任务中，按标签选取概率
- 多通道特征中，按 mask/索引选取特定通道
- 实现高级索引、ROI采样等

---

## 4. 注意事项

- `index` 的 shape 必须与 `input` 除 `dim` 以外的其他维度一致
- 索引超界会报错
- 与 `scatter` 相反，`gather` 是“按索引读”，`scatter` 是“按索引写”

---

## 5. 相关算子

- `scatter`：按索引写入
- `index_select`：按一维索引选取整个切片
- `take`：按扁平索引取值

---

**总结**：  
gather 算子是高效的批量索引工具，广泛用于深度学习模型的输出处理、标签选取、特征采样等场景。

# Scatter

**scatter 算子** 是深度学习框架中常用的“按索引写入”操作。它与 gather 相反：gather 是“按索引读”，scatter 是“按索引写”。

---

## 1. 基本概念

- **功能**：将给定的 values 按照指定的索引 index，写入到目标张量的指定维度上。
- **常见接口**（以 PyTorch 为例）：
  ```python
  tensor.scatter_(dim, index, src)
  ```
  - `dim`：在哪个维度上 scatter
  - `index`：索引张量，指定写入位置
  - `src`：要写入的数据（可以是标量或张量）

---

## 2. 例子

### 2.1 基本用法

```python
import torch

x = torch.zeros(2, 4, dtype=torch.int)
index = torch.tensor([[2, 1, 0, 3],
                      [0, 2, 3, 1]])
src = torch.tensor([[9, 10, 11, 12],
                    [13, 14, 15, 16]])

# 按行写入
x.scatter_(dim=1, index=index, src=src)
print(x)
# 输出：
# tensor([[11, 10,  9, 12],
#         [13, 16, 14, 15]])
```
解释：  
- 第一行：x[0,2]=9, x[0,1]=10, x[0,0]=11, x[0,3]=12  
- 第二行：x[1,0]=13, x[1,2]=14, x[1,3]=15, x[1,1]=16

### 2.2 用于 one-hot 编码

```python
labels = torch.tensor([2, 0, 1])
one_hot = torch.zeros(3, 4)
one_hot.scatter_(1, labels.unsqueeze(1), 1)
print(one_hot)
# 输出：
# tensor([[0., 0., 1., 0.],
#         [1., 0., 0., 0.],
#         [0., 1., 0., 0.]])
```

---

## 3. 常见用途

- one-hot 编码
- 生成 mask、标签分布
- 按索引批量写入数据
- 反向 gather 操作

---

## 4. 注意事项

- scatter_ 是原地操作（会修改原张量），scatter 返回新张量
- 如果同一位置被多次写入，后写入的值会覆盖前面的（可用 scatter_add 实现累加）
- index 的 shape 必须与 src 的 shape 匹配
- 索引超界会报错

---

## 5. 相关算子

- `gather`：按索引读
- `scatter_add`：按索引累加写入
- `index_add`：按索引累加（整轴）

---

**总结**：  
scatter 算子是高效的批量索引写入工具，广泛用于 one-hot 编码、标签分布生成、掩码构建等场景，是 gather 的“逆操作”。

# GRU

好的，下面详细讲解 **GRU（Gated Recurrent Unit）算子** 的原理、结构、公式、常见用法和注意事项。

---

## 1. 概念简介

**GRU（门控循环单元）** 是一种循环神经网络（RNN）结构，专为解决传统 RNN 难以捕捉长期依赖、梯度消失/爆炸等问题而设计。  
GRU 结构比 LSTM 更简单，参数更少，计算更快，但效果相近。

---

## 2. 结构与原理

GRU 通过**重置门（reset gate）**和**更新门（update gate）**来控制信息流动：

- **重置门**：决定当前输入和前一时刻隐藏状态结合的程度
- **更新门**：决定前一时刻隐藏状态有多少保留到当前时刻

### 结构图  
```
        ┌─────────────┐
xₜ ───▶│             │
       │    GRU      │───▶ hₜ
hₜ₋₁──▶│             │
        └─────────────┘
```

---

## 3. 公式推导

设输入为 \( x_t \)，前一时刻隐藏状态为 \( h_{t-1} \)，当前隐藏状态为 \( h_t \)：

1. **更新门** $z_t$
   $$
   z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
   $$
   
2. **重置门** \($ r_t$ \)：
   $$
   r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
   $$
3. **候选隐藏状态** $ \tilde{h}_t$
   $$
   {h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
   $$
4. **最终隐藏状态** \( $h_t$ \)：
   $$
   h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
   $$

---

其中 \($\sigma$\) 是 sigmoid，\($\odot$\) 是逐元素乘法。

## 4. 代码示例（PyTorch）

### 4.1 单步计算

```python
import torch
import torch.nn as nn

gru_cell = nn.GRUCell(input_size=10, hidden_size=20)
x_t = torch.randn(3, 10)      # batch=3, input_size=10
h_prev = torch.randn(3, 20)   # batch=3, hidden_size=20
h_t = gru_cell(x_t, h_prev)   # h_t: [3, 20]
```

### 4.2 序列计算

```python
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)
x = torch.randn(3, 5, 10)     # batch=3, seq_len=5, input_size=10
output, h_n = gru(x)          # output: [3, 5, 20], h_n: [1, 3, 20]
```

---

## 5. 常见用途

- 序列建模（文本、语音、时间序列等）
- 机器翻译、语音识别、对话系统
- 作为编码器/解码器的基本单元

---

## 6. 优缺点

**优点：**
- 结构简单，参数少，训练快
- 能捕捉长期依赖，缓解梯度消失

**缺点：**
- 仍有一定的长期依赖建模局限
- 对于极长序列，Transformer 等结构更优

---

## 7. 相关算子

- **LSTM**：更复杂的门控结构
- **RNN**：无门控，最基础的循环结构

---

**总结**：  
GRU 算子是高效的门控循环神经网络单元，广泛用于各种序列建模任务。它通过更新门和重置门灵活控制信息流动，兼顾性能和效率，是深度学习中常用的时序建模工具。

# SIFT

**SIFT（Scale-Invariant Feature Transform，尺度不变特征变换）算子** 是计算机视觉领域经典的特征提取算法，常用于图像的关键点检测与描述。它具有**尺度不变性**和**旋转不变性**，广泛应用于图像匹配、拼接、三维重建等任务。

---

## 1. SIFT 算子的主要流程

1. **尺度空间极值检测**  
   - 构建高斯金字塔，对图像做多尺度模糊。
   - 计算相邻尺度的差分高斯（DoG, Difference of Gaussian）。
   - 在 DoG 金字塔中寻找极值点（即潜在的关键点）。

2. **关键点精确定位**  
   - 对极值点进行亚像素级精确定位。
   - 剔除边缘响应弱、对比度低的点。

3. **方向分配**  
   - 计算关键点邻域的梯度方向直方图。
   - 给每个关键点分配主方向，实现旋转不变性。

4. **关键点描述符生成**  
   - 以关键点为中心，按照主方向旋转，提取邻域的梯度信息。
   - 生成 128 维的特征向量（描述符），用于后续匹配。

---

## 2. 算子特点

- **尺度不变性**：能检测不同大小的目标。
- **旋转不变性**：对图像旋转鲁棒。
- **部分仿射不变性**：对视角变化有一定适应性。
- **抗噪声、抗遮挡**：特征点稳定，适合匹配。

---

## 3. 典型应用

- 图像拼接（全景图）
- 目标识别与跟踪
- 三维重建
- 物体检测与识别
- 机器人视觉导航

---

## 4. 代码示例（OpenCV）

```python
import cv2

img = cv2.imread('test.jpg', 0)
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)

# keypoints: 关键点位置、尺度、方向等
# descriptors: 每个关键点的128维特征向量
```

---

## 5. 与其他特征算子的对比

| 算子  | 尺度不变 | 旋转不变 | 描述符长度 | 速度 | 备注             |
| ----- | -------- | -------- | ---------- | ---- | ---------------- |
| SIFT  | 是       | 是       | 128        | 慢   | 经典，专利已过期 |
| SURF  | 是       | 是       | 64         | 较快 | 专利             |
| ORB   | 否       | 是       | 32         | 快   | 免费，适合实时   |
| AKAZE | 是       | 是       | 64         | 快   | 免费             |

---

## 6. 总结

**SIFT 算子** 是一种强大的图像局部特征提取方法，具有良好的不变性和鲁棒性。虽然速度较慢，但在高精度图像匹配等场景仍被广泛使用。  
在深度学习兴起前，SIFT 是传统视觉领域的“黄金标准”特征算子之一。

# KNN

**KNN 算子**（K-Nearest Neighbors，K近邻）是机器学习和数据处理中的一种基础操作，既可以指经典的KNN分类/回归算法，也可以指在深度学习/点云处理中常用的“最近邻搜索”算子。下面详细讲解其原理、常见用法和实现方式。

---

## 1. 基本概念

- **KNN（K近邻）**：给定一个查询点，在数据集中找到距离它最近的K个点（邻居）。
- **常见用途**：
  - 传统KNN分类/回归
  - 点云特征提取（如PointNet++、DGCNN等）
  - 图神经网络（构建邻接关系）
  - 图像检索、聚类等

---

## 2. 算子原理

- **输入**：
  - 查询点集（query points），形状 [M, D]
  - 数据点集（reference points），形状 [N, D]
  - K：邻居个数
- **输出**：
  - 每个查询点的K个最近邻的索引和/或距离，形状 [M, K]（索引）

- **距离度量**：常用欧氏距离（L2），也可用L1、余弦等。

---

## 3. 代码示例

### 3.1 传统KNN分类（sklearn）

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

### 3.2 点云/深度学习中的KNN（PyTorch实现）

```python
import torch

def knn(x, k):
    # x: [B, N, D] (batch, 点数, 维度)
    inner = -2 * torch.matmul(x, x.transpose(2, 1))  # [B, N, N]
    xx = torch.sum(x ** 2, dim=-1, keepdim=True)     # [B, N, 1]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # [B, N, N]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]     # [B, N, k]
    return idx
```

### 3.3 点云库（Open3D）KNN

```python
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)  # points: [N, 3]
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
[k, idx, _] = pcd_tree.search_knn_vector_3d(query_point, K)
```

---

## 4. 常见用途

- **点云特征聚合**：如DGCNN中，KNN用于构建局部邻域特征
- **图神经网络**：KNN构图
- **聚类/降维**：如t-SNE、UMAP等
- **图像检索**：找最相似的K张图片

---

## 5. 性能与实现

- **暴力法**：O(NM)复杂度，适合小数据
- **KD树/Ball树/Annoy/FAISS**：大规模数据用高效索引结构加速
- **GPU实现**：如PyTorch3D、Open3D、FAISS等支持大规模并行KNN

---

## 6. 总结

**KNN算子**是“找最近邻”的基础工具，既可用于传统机器学习，也广泛用于点云、图神经网络、检索等深度学习场景。  
主流框架（sklearn、Open3D、PyTorch3D、FAISS等）均有高效实现。