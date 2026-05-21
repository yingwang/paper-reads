# LiBrA-Net：用李代数双边仿射场做实时 4K 视频去雾

> **原题**：LiBrA-Net: Lie-Algebraic Bilateral Affine Fields for Real-Time 4K Video Dehazing
> **作者**：Yongcong Wang, Chengchao Shen, Guangwei Gao, Wei Wang, Pengwen Dai, Dianjie Lu, Guijuan Zhang, Zhuoran Zheng
> **机构**：中南大学，南京理工大学，中山大学，山东师范大学，齐鲁工业大学
> **年份**：2026（arXiv 2605.11508，提交于 5 月 12 日）
> **分类**：cs.CV（Computer Vision and Pattern Recognition）
> **链接**：https://arxiv.org/abs/2605.11508
> **代码与数据**：https://anonymous.4open.science/r/LiBrA-Net-42B8
> **精读日期**：2026-05-21

## 阅读须知

**这篇在领域里的位置**。图像去雾已经有两条很成熟的路线：一条是单图像去雾，近几年已经能靠降采样特征、引导上采样、频域分解等办法处理 4K 图像；另一条是视频去雾，靠光流、可变形卷积、时序注意力或记忆模块稳定连续帧。但这两条线交叉处一直缺一块：原生 4K 视频去雾。单图像方法没有时间约束，逐帧跑会闪；视频方法的时序模块大多绑定在密集特征图上，升到 3840x2160 后计算和显存都爆。LiBrA-Net 的位置就在这个缺口：它不是把一个 720p 视频模型硬拉到 4K，而是把雾霾退化写成低频的逐像素仿射变换，再用小尺寸双边网格预测这个变换，最后切回 4K。

**读完能回答什么**。读完这份笔记之后，读者应当能回答如下几个问题。第一，为什么大气散射模型可以被改写成逐像素仿射颜色变换。第二，为什么这个仿射场适合放进双边网格，而不是直接在 4K 特征图上密集预测。第三，LiBrA-Net 为什么要把完整的时空颜色网格拆成 spatial-color 和 temporal 两个子网格。第四，所谓 Lie-algebraic fusion 与 Cayley map 究竟解决了什么工程问题。第五，UHV-4K 这个新 benchmark 补上了什么数据空白，以及为什么论文里 25 FPS、24.28 dB PSNR 这组数值是核心卖点。

**阅读前置**。本文默认读者知道大气散射模型（ASM）的基本形式，理解 PSNR、SSIM、LPIPS 这类恢复指标，也大致知道双边滤波和双边网格的思想：在空间坐标之外加一个颜色或强度维度，把边缘感知的局部变换压缩在一个低分辨率网格里。读者不需要熟悉李群李代数，但需要接受一个直觉：在靠近单位矩阵的位置，矩阵变换可以先在李代数里相加，再用 Cayley 或指数映射回可逆矩阵群。

**首次出现的缩写表**：

- **ASM**：Atmospheric Scattering Model，大气散射模型
- **UHD / 4K**：Ultra-High Definition，本文主要指 3840x2160
- **LiBrA-Net**：Lie-algebraic Bilateral-grid Affine Network
- **CAF**：Chromatic Affine Field，空间-颜色仿射场分支
- **TAF**：Temporal Affine Field，时序仿射场分支
- **HF-Refiner**：High-Frequency Refiner，高频细节恢复分支
- **UHV-4K**：论文发布的原生 4K 配对视频去雾 benchmark
- **tOF**：基于 RAFT warp 的相邻帧时间一致性误差

## 为什么这个问题值得做

4K 视频去雾难的地方，不只是像素多。雾的物理退化是全局、低频、与深度相关的：远处建筑、路牌、线缆、行人边界会先被雾抹掉，而这些正是 4K 传感器比 720p 更有价值的部分。如果把 4K 视频切成 patch，局部块看不到全局 atmospheric light 和大尺度深度变化；如果先降采样到 720p，再做去雾，很多近 4K Nyquist 频率的细节已经没了；如果直接把视频去雾网络放大到 4K，光流、注意力、对齐模块的成本又会随像素数线性甚至二次增长。

论文的核心判断是：去雾需要在两个频段里分工。低频部分决定颜色校正和雾浓度，是一个由深度场控制的仿射变换；高频部分主要是被 transmission 衰减但仍然残留在输入里的边缘和纹理。前者不应该在 4K 特征图上密集预测，而应该用一个很小的双边网格描述；后者不应该指望低分辨率网格凭空生成，而应该从原始 4K 输入里抽取并增强。LiBrA-Net 就是围绕这个分工搭出来的。

另一个现实问题是数据。REVIDE 是真实室内视频，但不是 4K；HazeWorld 规模大，但最高 720p；GoProHazy 和 DrivingHazy 是 1080p 且存在未对齐或无参考的问题。之前没有一个配对的原生 4K 视频去雾数据集，同时带 depth、transmission 和 optical flow。没有这样的 benchmark，就很难系统回答“4K 视频去雾到底需要什么结构”。

## 一、问题

LiBrA-Net 要解决的问题可以这样写：给定连续 T 帧有雾视频，直接恢复中心帧的干净 4K 图像，同时保持时间一致性，并且推理速度要接近实时。论文默认 T=5，也就是每次看一个很短的时间窗口。

从物理模型看，ASM 写成：

```text
I(x) = J(x) t(x) + A_inf (1 - t(x))
t(x) = exp(-beta d(x))
```

其中 I 是有雾图像，J 是干净图像，d 是深度，beta 是散射系数，A_inf 是大气光。反过来解 J，就得到：

```text
J(x) = a(x) I(x) + b(x)
a(x) = 1 / t(x)
b(x) = A_inf (1 - 1 / t(x))
```

这说明去雾的低频主干可以被看成逐像素仿射颜色变换。理想单通道模型里 a(x) 只是一个标量；真实 RGB 图像里，波长差异、相机 ISP、白平衡和颜色校正会让三个通道互相耦合。论文因此把标量推广成 3x3 矩阵 A(x) 和 3 维偏置 b(x)，每个像素需要 12 个仿射系数。

真正的难点是如何预测这 12 个系数。直接预测 HxWx12 的 4K 仿射场太贵，而且会鼓励网络学习不必要的高频自由度，造成闪烁和颜色漂移。可是如果只做低分辨率预测，又必须保留边缘感知能力，否则天空、树、建筑边界附近的 transmission 会被抹平。双边网格正好适合这件事：空间维度很小，颜色或引导维度保留边缘结构，最后通过 trilinear slicing 切回目标分辨率。

## 二、方法

LiBrA-Net 的整体设计可以概括成一句话：在固定 256x256 输入上预测两个小双边仿射网格，在李代数里融合后用 Cayley map 变成可逆 3x3 颜色变换，再用原始 4K 输入补回高频。

```mermaid
graph TD
  A[5 帧有雾视频] --> B[固定 256x256 编码]
  B --> C[Chromatic Affine Field<br/>空间-颜色网格]
  B --> D[Temporal Affine Field<br/>时序网格]
  C --> E[Trilinear slicing]
  D --> E
  E --> F[Lie algebra 中相加]
  F --> G[Cayley map 到 GL(3)]
  G --> H[低频去雾结果]
  A --> I[原始 4K 中心帧]
  I --> J[HF-Refiner]
  H --> J
  J --> K[4K 干净中心帧]
```

### 1. 从完整 6D 网格拆成两个 4D 子网格

如果把视频去雾的仿射场完整写成双边网格，它会同时依赖时间、空间 x/y、颜色 r/g/b，也就是一个 6D 网格：

```text
G[t, x, y, r, g, b] -> R^12
```

这个对象太大，也太难学。LiBrA-Net 把它近似拆成两个子网格：

- spatial-color grid：形状为 12 x Gc x Gh x Gw，用来建模中心帧中颜色与空间位置相关的仿射系数。
- temporal grid：形状为 12 x T x Gh x Gw，用来建模连续帧之间的低频变化。

论文固定 Gc=8、Gh=Gw=16、T=5。也就是说，不管输入是 720p、4K 还是 8K，真正要预测的核心网格都很小。输出分辨率只影响最后 slicing 的像素数，而不影响网格预测的主干成本。

### 2. Chromatic Affine Field

Chromatic Affine Field 只看中心帧。中心帧先被双线性缩放到 256x256，经过一个 stride-16 encoder，得到 192 通道、16x16 的特征图。网格头输出两套 spatial-color grid，分别用 R 通道和 G 通道作为 guide。这样做的理由是 R/G 对雾和景深的敏感性不同，两个 guide 可以提供互补的边缘分区。切片后，两个预测取平均，同时用 guide-consistency regularization 防止二者漂移。

这个分支承担的是空间与颜色结构：哪些区域远、哪些区域近，天空和建筑边缘怎样过渡，某个颜色 bin 里的仿射校正应该更接近哪个局部深度模式。

### 3. Temporal Affine Field

Temporal Affine Field 看 T=5 帧。每帧同样缩到 256x256 并经过独立 encoder，然后在每个 16x16 空间位置上沿时间维做 multi-head self-attention。注意力只在 5 个时间 token 上做，所以成本很小；但它比固定的 1x1 temporal convolution 更灵活，可以根据运动、遮挡和雾浓度变化调整跨帧融合权重。

这个分支不直接做光流 warp，也不显式预测运动。它学的是低频仿射系数随时间如何变化。论文后面的时间一致性实验也支持这一点：去掉 Temporal Field 后，tOF 变化不大，但每段视频内部的 PSNR 波动明显变大，说明它主要稳定的是逐帧恢复质量，而不是靠模糊换来低 tOF。

### 4. Lie-algebraic fusion 与 Cayley map

两个分支都输出 12 维仿射系数。直觉上可以把它们相加，但直接相加 3x3 变换矩阵有一个问题：两个可逆矩阵的和不一定可逆。论文的做法是先把矩阵部分放在李代数 gl(3) 里相加：

```text
c_fused = c_chromatic + c_temporal
```

前 9 个系数组成 M in gl(3)，后 3 个系数是平移向量。然后用 Cayley map 把 M 映射回可逆矩阵：

```text
A = Cay(M) = (I - M / 2)^(-1) (I + M / 2)
```

这个设计有两个好处。第一，M=0 时 A=I，所以所有网格头可以零初始化，模型一开始就是近似 identity mapping，训练更稳。第二，只要 M 不靠近让 I - M/2 奇异的位置，A 就是可逆的。论文用 identity prior、空间平滑、时间平滑、R/G guide consistency 四个无参数正则，把网格限制在靠近单位变换的小范围里。

这里的数学主张是：在靠近单位矩阵时，Cay(M_chi + M_tau) 近似等于 Cay(M_chi) Cay(M_tau)，误差是二阶量，主要由 commutator 控制。换句话说，加法融合在李代数里是合理的一阶近似，而不是随意拼接。

### 5. HF-Refiner

双边网格擅长低频颜色和雾浓度校正，但它不负责 4K 纹理。论文把高频恢复单独交给 HF-Refiner。根据 ASM 的梯度近似：

```text
grad I(x) ~= t(x) grad J(x)
```

有雾图像里仍然保留了一份被 t(x) 缩小的干净梯度。因此 HF-Refiner 不需要超分辨率式地发明纹理，只要从原始中心帧中提取高频载体，并根据低频去雾结果进行增强。实现上，它在 H/4 x W/4 上工作：原始 4K 中心帧经过 PixelUnshuffle，低频去雾图也在同一尺度，二者融合后再 PixelShuffle 回 4K。

最终输出可以理解为三项相加：原始输入作为高频载体、双边路径提供低频颜色校正、HF-Refiner 提供学习到的高频残差。

## 三、实验

### 数据集

论文评估三个配对视频去雾 benchmark：

| 数据集 | 类型 | 分辨率 | 规模 | 备注 |
| ---- | ---- | ---- | ---- | ---- |
| UHV-4K | 合成视频 | 3840x2160 | 100 段视频，2500 帧 | 本文新建，带 depth、transmission、flow |
| REVIDE | 真实室内视频 | 2708x1800 | 42 train / 6 test | 配对真实数据 |
| HazeWorld | 合成户外视频 | <=720p | 1016 train / 584 test | 取 DAVIS、DDAD、UA-DETRAC 子集 |

UHV-4K 来自 Inter4K 和 UVG 的干净 4K 视频。论文先用 Depth Anything V2 估计深度、用 RAFT 估计光流，再按 ASM 合成有雾帧，并输出五种对齐模态：hazy input、clean ground truth、depth、transmission、optical flow。每段视频固定 beta 和 A_inf，以保持时间一致性。

### 主结果

LiBrA-Net 和 16 个 baseline 比较，包括 8 个单图像去雾方法和 8 个视频去雾方法。核心结果如下：

| 方法 | UHV-4K PSNR | UHV-4K SSIM | UHV-4K LPIPS | 4K FPS | GFLOPs | 参数量 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| DehazeFormer | 23.74 | 0.9313 | 0.0468 | 0.18 | 6.00K | 2.54M |
| UHDformer | 23.16 | 0.9381 | 0.0746 | 0.13 | 33.30K | 20.33M |
| ViWS-Net | 22.06 | 0.7742 | 0.3586 | 4.21 | 1.20K | 57.67M |
| 4KDehazing | 18.56 | 0.8608 | 0.1011 | 3.11 | 10.20K | 17.26M |
| LiBrA-Net | **24.28** | **0.9437** | **0.0401** | **25.27** | **0.28K** | 6.12M |

最重要的不是单个 PSNR 数字，而是质量和速度同时成立。LiBrA-Net 是表中唯一一个在 UHV-4K 上超过 23 dB PSNR、同时达到实时 4K 的方法。相对于最接近的双边网格路线 4KDehazing，它在 4K 上快约 8 倍，GFLOPs 少约 36 倍。

在 REVIDE 上，LiBrA-Net 得到 21.43 dB / 0.8678 / 0.2397，排在视频去雾方法第一，并接近最强单图像 baseline。HazeWorld 只有 720p，分辨率解耦的优势变小，LiBrA-Net 的 PSNR 是 27.46，略低于 ViWS-Net 的 27.66，但 LPIPS 最好。

### 时间一致性

论文用 tOF 和视频内 PSNR 标准差衡量时间稳定性。LiBrA-Net 的 tOF 是 0.004769，不是绝对最低；DCL 和 ViWS-Net 更低。但这两个方法的 PSNR 分别只有 20.81 和 22.06，比 LiBrA-Net 低 3.47 dB 和 2.22 dB。论文的解释是：tOF 会奖励任何相邻帧相似，包括过度平滑；因此必须和 PSNR 一起看。LiBrA-Net 的位置是“比所有单图像方法更稳，同时保持最高恢复质量”。

### 消融

几个消融很关键：

| 变体 | UHV-4K PSNR | 降幅 |
| ---- | ---- | ---- |
| Full model | 24.28 | -- |
| 去掉 Chromatic Field | 22.02 | -2.26 |
| 去掉 Temporal Field | 22.09 | -2.19 |
| 去掉 HF-Refiner | 20.87 | -3.41 |
| 去掉 Cayley map | 21.69 | -2.59 |
| 去掉 resolution-adaptive slicing | 21.01 | -3.27 |
| 去掉 Lie regularization | 21.87 | -2.41 |

这组结果说明三件事。第一，Chromatic Field 和 Temporal Field 贡献接近，两个子网格拆分不是装饰。第二，HF-Refiner 在 4K 上最重要，去掉后掉 3.41 dB，说明高频纹理确实不能靠低分辨率网格解决。第三，Cayley map 和 Lie regularization 对数值稳定和时间一致性都有实质影响，不只是漂亮的数学包装。

## 四、局限

**论文自己承认的**。第一，当前时序窗口只有约 1 秒，无法保证分钟级长视频的一致性。第二，UHV-4K 的合成过程假设每段视频有单一 beta 和 A_inf，因此混合天气、局部雾团、烟尘与雨雾混合不在覆盖范围内。第三，真实配对 4K 有雾/无雾视频仍然不存在，所以 UHV-4K 虽然补了 benchmark 空白，但仍然不能替代真实采集数据。

**读完能看出来的**。还有几个潜在风险。第一，论文把去雾低频结构高度绑定到 ASM 和深度平滑性上；这对大气雾很合理，但对非均匀烟雾、车窗污渍、镜头眩光这类退化未必成立。第二，UHV-4K 的深度和光流本身来自 Depth Anything V2 与 RAFT，合成质量受这些上游模型影响；benchmark 的 annotation 很丰富，但不是物理采集真值。第三，LiBrA-Net 的强项在 4K 及以上，HazeWorld 的 720p 结果显示，当分辨率不高时，它的优势会缩小。第四，Cayley map 每个像素都要做 3x3 solve，虽然整体 GFLOPs 很低，但具体部署时仍需要高效 kernel，否则理论上的分辨率解耦可能被实现细节吃掉。

## 一句话

LiBrA-Net 把 4K 视频去雾拆成“低频仿射场 + 高频残差”两件事：前者用两个固定小尺寸双边网格在李代数里融合，后者从原始 4K 输入中补回细节，因此在 UHV-4K 上同时做到 24.28 dB 和 25 FPS。
