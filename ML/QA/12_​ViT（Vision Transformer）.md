# **ViT（Vision Transformer）网络详解**

------

## 一、ViT 是什么？

**ViT（Vision Transformer，视觉Transformer）**是由 **Google Research** 在 **2020年** 提出的深度学习模型，首次将 **Transformer 架构（原本用于自然语言处理，NLP）成功迁移到计算机视觉（CV）任务中**，尤其是在 **图像分类（Image Classification）** 任务上取得了与 CNN（如 ResNet）相媲美甚至更好的效果。

> 📌 **核心思想：用 Transformer 的编码器（Encoder）来直接处理图像，代替传统的卷积神经网络（CNN）。**

------

## 二、ViT 的诞生背景

在 ViT 出现之前，几乎所有的 **主流视觉模型（如 AlexNet、VGG、ResNet、EfficientNet）都基于 CNN（卷积神经网络）**，因为 CNN 擅长提取局部空间特征（比如边缘、纹理）。

而 **Transformer 最初是为 NLP 设计的（如 BERT、GPT），它的核心是 Self-Attention（自注意力机制），擅长捕捉全局依赖关系**。

ViT 的核心贡献就是：

> **“既然 Transformer 在 NLP 上这么强，那我们能不能直接拿它来做图像？”**

------

## 三、ViT 的核心思想（与 CNN 对比）

| 特性             | **CNN（如 ResNet）**     | **ViT（Vision Transformer）**           |
| ---------------- | ------------------------ | --------------------------------------- |
| **基本单元**     | 卷积核（局部感受野）     | Self-Attention（全局感受野）            |
| **信息捕捉方式** | 局部感知，逐步扩大感受野 | 全局建模，直接关注所有位置关系          |
| **输入结构**     | 直接输入图像像素         | 将图像切分为多个 Patch，线性嵌入        |
| **位置信息**     | 卷积自带位置信息         | 需要显式添加 **Position Embedding**     |
| **模型架构**     | 多层卷积 + 池化          | **Transformer Encoder（只有 Encoder）** |
| **适用场景**     | 传统 CV 任务             | 图像分类、可扩展至检测/分割等           |

------

## 四、ViT 的工作原理（详细流程）

ViT 的整体流程可以概括为以下几个步骤：

------

### **1️⃣ 输入图像 → 切割成多个小 Patch（图像分块）**

- 假设输入一张 **224×224 的 RGB 图像**
- 将其切割成多个固定大小的 **Patch（比如 16×16 像素的小块）**
  - 每个 Patch 大小：16×16×3 = **768 维向量**
  - 224 / 16 = 14，所以总共有 **14×14 = 196 个 Patch**

🔧 **数学表示：**

```
Image∈RH×W×C切块Patches∈RN×(P2⋅C)
```

- N=14×14=196（Patch 数量）
- P=16（Patch 大小，如 16×16）
- C=3（通道数，RGB）

------

### **2️⃣ 每个 Patch → 线性投影（Linear Projection）→ Patch Embedding**

- 每个 16×16×3 的 Patch（768 维）通过一个 **可学习的线性层（全连接）** 映射到一个更低维（或相同维）的向量，比如 **D=768**
- 这样每个 Patch 就变成了一个 **D 维向量（如 768 维）**
- 这一步类似于 CNN 中的 **卷积核提取特征**，但在 ViT 里是用全连接层

🔧 **数学表示：**

```
zi=E⋅xi+Epos
```

- xi：第 i 个 Patch 的像素向量
- E：可学习的投影矩阵（将 Patch 映射为 D 维）
- Epos：位置编码（见下一步）

------

### **3️⃣ 加入位置信息（Position Embedding）**

- **Transformer 本身没有“位置感”**，不像 CNN 的卷积核天然具有局部位置信息
- 所以 ViT 需要 **显式地为每个 Patch 添加位置编码（Position Embedding）**
  - 通常是可学习的向量，和 Patch Embedding 相加
- 位置编码的维度也是 D（比如 768 维）

🔧 类似于 NLP 中的 Token 位置编码，但这里是 **图像 Patch 的位置**

------

### **4️⃣ 加入 [CLS] Token（分类专用 Token）**

- 在 NLP 中（如 BERT），我们会加入一个特殊的 **[CLS] Token** 用于分类任务

- ViT 也借鉴了这个思想，在所有 Patch Tokens **前面额外加一个可学习的 [CLS] Token**

- 最终输入 Transformer 的序列是：

  ```
  [[CLS],z1,z2,...,zN]
  ```

- 其中 zi是第 i 个 Patch 的 Embedding
- **[CLS] Token 的最终输出会被用作整个图像的分类表示**

------

### **5️⃣ 输入 Transformer Encoder（核心！）**

- 把上述序列（包含 [CLS] + 所有 Patch Embeddings + Position Embeddings）输入到 **标准的 Transformer Encoder**
- Transformer Encoder 包括：
  - **Multi-Head Self-Attention（多头自注意力）**
  - **Feed-Forward Network（前馈神经网络，FFN）**
  - **Layer Normalization & 残差连接（Add & Norm）**

🔧 ViT 通常使用 **多个 Transformer Encoder 层堆叠（比如 12 层或 24 层）**

------

### **6️⃣ 输出 [CLS] Token → 分类头（Classification Head）**

- 经过多层 Transformer Encoder 后，**取 [CLS] Token 的最终输出向量**
- 这个向量被认为包含了整个图像的全局信息
- 然后接一个 **全连接层（MLP）**，输出类别概率（比如 1000 类 ImageNet）

🔧 **分类公式：**

```
Class=Softmax(W⋅[CLS]L+b)
```

- [CLS]L：经过 L 层 Encoder 后的 [CLS] Token 表示

------

## 五、ViT 的整体架构图（简化版）

```
原始图像 (224x224x3)
       ↓
切分成 16x16 的 Patches (196个)
       ↓
每个 Patch 线性投影 → Patch Embedding (D=768)
       ↓
加入可学习的位置编码 (Position Embedding)
       ↓
在最前面加入 [CLS] Token
       ↓
输入 Transformer Encoder（多层 Self-Attention + FFN）
       ↓
取出 [CLS] Token 的输出
       ↓
通过分类头（MLP）→ 输出类别概率
```

------

## 六、ViT 的训练方式（与 CNN 不同之处）

ViT 在训练时通常：

1. **在大规模数据集上预训练（如 JFT-300M、ImageNet-21k）**
   - 因为纯 ViT 在小数据上表现不如 CNN（需要大量数据才能学好 Self-Attention）
2. **使用强大的数据增强（如 Mixup、CutMix、Random Erasing）**
3. **使用 AdamW 优化器 + 学习率 Warmup + Dropout 等正则手段**
4. **预训练后可以微调（Fine-tuning）到小数据集（如 ImageNet-1k、CIFAR）**

------

## 七、ViT 的优缺点分析

### ✅ 优点：

1. **全局建模能力强**：Self-Attention 可以捕捉图像中任意两个区域的关系，不像 CNN 只能逐步扩大感受野
2. **架构统一**：和 NLP 的 Transformer 完全一致，便于跨模态融合（如 Vision + Language）
3. **可扩展性强**：可以通过增加层数、头数、Patch 数量来提升性能
4. **在大数据下超越 CNN**：当训练数据足够大时（如 JFT-300M），ViT 性能超过 ResNet

### ❌ 缺点：

1. **对数据量要求高**：在小数据集（如 CIFAR-10）上，ViT 效果通常不如 CNN
2. **计算复杂度高**：Self-Attention 的计算复杂度是 O(N2)（N 是 Patch 数量），图像越大越慢
3. **缺乏归纳偏置（Inductive Bias）**：CNN 有“局部性”和“平移不变性”的天然假设，ViT 没有，需要更多数据学习

------

## 八、ViT 的变种与改进模型

由于原始 ViT 存在一些缺陷，后续研究者提出了很多改进版本，包括：

| 模型                                              | 核心改进                            | 说明                                           |
| ------------------------------------------------- | ----------------------------------- | ---------------------------------------------- |
| **DeiT（Data-efficient Image Transformer）**      | 更高效训练，小数据也能用好 ViT      | 引入蒸馏（Distillation Token）、更快的训练策略 |
| **Swin Transformer**                              | 引入局部窗口 Attention + 层次化结构 | 更适合高分辨率图像，成为 CV 主流模型           |
| **T2T-ViT（Tokens-to-Token ViT）**                | 逐步聚合 Patch，减少冗余计算        | 缓解 ViT 过平滑问题                            |
| **LeViT（Lightweight ViT）**                      | 更轻量，适合移动端/边缘设备         | 结合 CNN 的卷积先验                            |
| **Patch Refinement / Dynamic Vision Transformer** | 动态调整 Patch 或 Attention         | 提升效率与精度                                 |

------

## 九、总结：ViT 是什么？为什么要学它？

| 问题                    | 答案                                                         |
| ----------------------- | ------------------------------------------------------------ |
| **ViT 是什么？**        | 将 Transformer 架构应用于图像任务的模型，用 Self-Attention 替代卷积，用于图像分类等视觉任务 |
| **核心思想是什么？**    | 把图像切成 Patch，线性嵌入后输入 Transformer Encoder，用 [CLS] Token 做分类 |
| **和 CNN 的最大区别？** | CNN 是局部感知，ViT 是全局建模（Self-Attention）             |
| **优点是什么？**        | 全局关系建模强，架构统一，适合多模态                         |
| **缺点是什么？**        | 对数据量要求高，计算复杂，缺乏 CNN 的归纳偏置                |
| **应用场景？**          | 图像分类（主任务）、可扩展到目标检测、分割、视频理解等       |
| **未来趋势？**          | 与 CNN 结合（如 ConvNeXt、Hybrid Models），或更高效的 Transformer（如 Swin、MobileViT） |

------

## ✅ 如果你想要：

- **了解 ViT 的代码实现（PyTorch / HuggingFace）**
- **想了解 ViT 在目标检测 / 分割上的应用（如 DETR）**
- **想知道 ViT 和 CNN 的对比实验**
- **想了解 DeiT / Swin Transformer 等改进模型**

欢迎继续提问！我可以为你详细展开！ 😊