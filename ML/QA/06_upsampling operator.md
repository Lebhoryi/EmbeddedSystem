### 核心定义
- **上采样算子（Upsampling operator）**：在深度学习中，把特征图的空间分辨率从 \((H,W)\) 提升到更大的 \((H',W')\) 的算子。常用于解码器、分割、超分辨率、生成式模型等。可分为“非学习型插值”和“可学习型上采样”。

### 常见类型
- **非学习型（固定权重）**
  - 最近邻插值、双线性/双三次插值
  - 反池化（unpooling with indices）：用最大池化的索引把值放回原位置
- **可学习型**
  - 转置卷积（Transposed Conv，俗称“deconv”）
    - 输出尺寸：`out = (in - 1) * stride - 2*padding + kernel + output_padding`
    - 易出现棋盘格伪影
  - 子像素卷积（Pixel Shuffle）
    - 先卷积得到 `C*r^2` 通道，再重排为 `(H*r, W*r, C)`
  - 上采样后卷积（Resize-then-Conv）
    - 先插值放大，再用普通卷积，常比转置卷积更稳

### 选型建议
- **速度/简单**：最近邻或双线性插值 + 卷积
- **需要可学习放大**：转置卷积或子像素卷积（超分辨率常用）
- **避免棋盘格**：优先“上采样+卷积”或合理的 stride/kernel
- **对齐问题**：双线性插值注意 `align_corners` 设置，影响边界对齐

### PyTorch 常用写法
```python
# 1) 非学习：插值
nn.Upsample(scale_factor=2, mode='nearest')             # 或 'bilinear', align_corners=False

# 2) 可学习：转置卷积（上采样倍数=stride）
nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)

# 3) 可学习：子像素卷积（像素重排，放大倍数=r）
nn.Sequential(
    nn.Conv2d(in_c, out_c * (r*r), kernel_size=3, padding=1),
    nn.PixelShuffle(r)
)
```

### 典型应用
- 语义分割解码器（FPN/UNet）
- GAN 生成器上采样
- 超分辨率（SR，常用 Pixel Shuffle）
- 自动编码器解码阶段