# LiBrA-Net: Lie-Algebraic Bilateral Affine Fields for Real-Time 4K Video Dehazing

> **Original title**: LiBrA-Net: Lie-Algebraic Bilateral Affine Fields for Real-Time 4K Video Dehazing
> **Authors**: Yongcong Wang, Chengchao Shen, Guangwei Gao, Wei Wang, Pengwen Dai, Dianjie Lu, Guijuan Zhang, Zhuoran Zheng
> **Institutions**: Central South University, Nanjing University of Science and Technology, Sun Yat-sen University, Shandong Normal University, Qilu University of Technology
> **Year**: 2026 (arXiv 2605.11508, submitted May 12)
> **Subject**: cs.CV (Computer Vision and Pattern Recognition)
> **Link**: https://arxiv.org/abs/2605.11508
> **Code and data**: https://anonymous.4open.science/r/LiBrA-Net-42B8
> **Reading date**: 2026-05-21

## Reading Notes

**Where this paper sits in the field.** Image dehazing has two mature tracks. Single-image dehazing has learned to handle 4K images through reduced-resolution features, guided upsampling, and frequency decomposition. Video dehazing has learned to stabilize consecutive frames through optical flow, deformable alignment, temporal attention, or memory. The intersection of these tracks is still sparse: native 4K video dehazing. Single-image methods flicker when applied frame by frame, while video methods usually bind temporal modeling to dense feature maps and become too expensive at 3840x2160. LiBrA-Net targets exactly this gap. It does not scale a 720p video model upward; it rewrites haze removal as a low-frequency per-pixel affine transform, predicts that transform with compact bilateral grids, and slices it back to 4K.

**What you should be able to answer afterwards.** After reading these notes, the reader should be able to answer five questions. First, why can the atmospheric scattering model be rewritten as a per-pixel affine color transform. Second, why is this affine field better represented by a bilateral grid than by dense 4K prediction. Third, why does LiBrA-Net factor a full spatiotemporal color grid into spatial-color and temporal sub-grids. Fourth, what problem Lie-algebraic fusion and the Cayley map solve. Fifth, what gap UHV-4K fills, and why the paper's 25 FPS, 24.28 dB PSNR, 6.12M-parameter result is the key claim.

**Prerequisites.** This note assumes familiarity with the atmospheric scattering model, restoration metrics such as PSNR, SSIM, and LPIPS, and the broad idea behind bilateral filtering or bilateral grids: add a guidance dimension to spatial coordinates so that a low-resolution grid can represent edge-aware local transforms. No deep background in Lie groups is required. The only needed intuition is that near the identity, transformations can be added in the Lie algebra and then mapped back to an invertible matrix group.

**Glossary of abbreviations**:

- **ASM**: Atmospheric Scattering Model
- **UHD / 4K**: Ultra-High Definition, here mainly 3840x2160
- **LiBrA-Net**: Lie-algebraic Bilateral-grid Affine Network
- **CAF**: Chromatic Affine Field
- **TAF**: Temporal Affine Field
- **HF-Refiner**: High-Frequency Refiner
- **UHV-4K**: the paper's native-4K paired video dehazing benchmark
- **tOF**: temporal optical-flow error measured with RAFT warping

## Why This Problem Is Worth Solving

4K video dehazing is not hard only because there are many pixels. Haze is a global, low-frequency, depth-governed degradation. Distant signs, cables, railings, and pedestrian boundaries lose contrast first, and these are exactly the details that justify using a 4K sensor. If a 4K video is processed patch by patch, each patch loses the global atmospheric context. If it is downsampled to 720p first, many near-Nyquist details disappear before the dehazer can see them. If a conventional video dehazing network is run natively at 4K, its flow, attention, or alignment modules scale linearly or quadratically with pixel count.

The paper's central judgment is that dehazing should be split by frequency. The low-frequency part is the color correction and haze density field, which can be represented as a depth-governed affine transform. The high-frequency part is mostly edge and texture information that remains in the hazy input but is attenuated by transmission. The former should not be densely predicted on a 4K feature map; it should be encoded in a compact bilateral grid. The latter should not be hallucinated by a low-resolution grid; it should be extracted from the original 4K input and strengthened.

There is also a data issue. REVIDE is real indoor video but not 4K. HazeWorld is large but capped at 720p. GoProHazy and DrivingHazy are 1080p and either unaligned or no-reference. Before this paper, there was no paired native-4K video dehazing benchmark with depth, transmission, and optical-flow annotations on every frame. Without such a benchmark, it is difficult to evaluate what structure 4K video dehazing actually needs.

## I. The Problem

LiBrA-Net solves the following problem: given T consecutive hazy frames, restore the clean center frame at native 4K resolution, preserve temporal consistency, and run near real time. The paper uses T=5, so each prediction observes a short temporal window.

The atmospheric scattering model is:

```text
I(x) = J(x) t(x) + A_inf (1 - t(x))
t(x) = exp(-beta d(x))
```

Here I is the hazy image, J is the clean image, d is depth, beta is the scattering coefficient, and A_inf is global atmospheric light. Solving for J gives:

```text
J(x) = a(x) I(x) + b(x)
a(x) = 1 / t(x)
b(x) = A_inf (1 - 1 / t(x))
```

So the low-frequency backbone of dehazing can be interpreted as a per-pixel affine color transform. In the ideal scalar model, a(x) is a scalar gain. In real RGB imaging, wavelength dependence, ISP processing, white balance, and color correction couple channels together. The paper therefore promotes the scalar gain to a 3x3 matrix A(x) plus a 3D bias b(x), giving 12 affine coefficients per pixel.

The real question is how to predict those coefficients. Dense HxWx12 prediction at 4K is expensive and gives the network too much high-frequency freedom, often causing flicker and color drift. But a purely low-resolution prediction also needs to preserve edge-aware transitions, otherwise transmission changes around sky, buildings, trees, and object boundaries become blurry. Bilateral grids are a natural fit: the spatial grid is small, a guidance dimension preserves edge awareness, and trilinear slicing returns coefficients at the target resolution.

## II. Method

LiBrA-Net can be summarized in one sentence: predict two compact bilateral affine grids from fixed 256x256 inputs, fuse them in the Lie algebra, map the result into invertible 3x3 color transforms with the Cayley map, and restore high-frequency residuals from the original 4K input.

```mermaid
graph TD
  A[5 hazy frames] --> B[fixed 256x256 encoding]
  B --> C[Chromatic Affine Field<br/>spatial-color grid]
  B --> D[Temporal Affine Field<br/>temporal grid]
  C --> E[Trilinear slicing]
  D --> E
  E --> F[addition in Lie algebra]
  F --> G[Cayley map to GL(3)]
  G --> H[low-frequency dehazed result]
  A --> I[original 4K center frame]
  I --> J[HF-Refiner]
  H --> J
  J --> K[clean 4K center frame]
```

### 1. Factorizing One 6D Grid into Two 4D Grids

A complete spatiotemporal bilateral grid for video dehazing would depend on time, spatial coordinates, and full RGB color:

```text
G[t, x, y, r, g, b] -> R^12
```

This 6D object is too large to store and too hard to learn. LiBrA-Net approximates it with two sub-grids:

- A spatial-color grid with shape 12 x Gc x Gh x Gw, modeling coefficients that depend on image location and color guidance.
- A temporal grid with shape 12 x T x Gh x Gw, modeling low-frequency changes across frames.

The paper fixes Gc=8, Gh=Gw=16, and T=5. The key point is that the learned grids stay tiny whether the output is 720p, 4K, or 8K. Output resolution affects only the slicing step, not the core grid prediction cost.

### 2. Chromatic Affine Field

The Chromatic Affine Field sees only the center frame. The frame is bilinearly resized to 256x256, processed by a stride-16 encoder, and reduced to a 192-channel 16x16 feature map. The grid head predicts two spatial-color grids, one guided by the R channel and one guided by the G channel. The reason is that R and G have different spectral responses under haze, so they provide complementary edge-aware partitions. After slicing, the two predictions are averaged, and a guide-consistency regularizer keeps them aligned.

This branch carries spatial and chromatic structure: which regions are far or near, how sky and building edges transition, and how a local color bin should map to an affine correction.

### 3. Temporal Affine Field

The Temporal Affine Field sees T=5 frames. Each frame is resized to 256x256 and encoded independently. At each of the 16x16 spatial positions, a multi-head self-attention block runs along the time axis. Since the attention sequence length is only five, the cost is tiny, but it is more adaptive than a fixed 1x1 temporal convolution.

This branch does not explicitly warp with optical flow. It learns how low-frequency affine coefficients evolve through the short frame window. The temporal consistency experiments support this reading: removing the Temporal Field barely changes tOF, but it greatly increases the within-video PSNR variation, meaning the branch stabilizes restoration quality rather than simply blurring frames into each other.

### 4. Lie-Algebraic Fusion and the Cayley Map

Both branches output 12 affine coefficients. It is tempting to add the resulting matrices directly, but the sum of two invertible matrices is not necessarily invertible. LiBrA-Net instead adds the matrix part in the Lie algebra gl(3):

```text
c_fused = c_chromatic + c_temporal
```

The first nine coefficients form M in gl(3), and the last three are the translation vector. The matrix is then mapped back into an invertible transform with the Cayley map:

```text
A = Cay(M) = (I - M / 2)^(-1) (I + M / 2)
```

This has two practical benefits. First, M=0 gives A=I, so the grid heads can be zero-initialized and the network starts as an identity mapping. Second, A is invertible as long as M does not hit the singularity of I - M/2. The paper uses four parameter-free regularizers to keep the grids in a small, smooth, near-identity region: identity prior, spatial smoothness, temporal smoothness, and R/G guide consistency.

The mathematical claim is local: near the identity, Cay(M_chi + M_tau) approximates Cay(M_chi) Cay(M_tau), with second-order error dominated by the commutator. Thus addition in the Lie algebra is a principled first-order composition, not just a convenient concatenation trick.

### 5. HF-Refiner

Bilateral grids are good at low-frequency color and haze correction, but not at 4K texture recovery. The paper therefore gives high-frequency restoration to a separate HF-Refiner. From the gradient form of the ASM:

```text
grad I(x) ~= t(x) grad J(x)
```

the hazy frame still contains an attenuated copy of the clean gradients. The HF-Refiner does not need to invent super-resolution detail; it needs to extract and amplify high-frequency carriers from the original center frame. Implementation-wise, it works at H/4 x W/4: the original frame is PixelUnshuffled, the low-frequency dehazed result lives at the same scale, and the fused residual is PixelShuffled back to 4K.

The final output is the sum of three pieces: the original input as high-frequency carrier, the bilateral path's low-frequency color correction, and the learned high-frequency residual.

## III. Experiments

### Datasets

The paper evaluates on three paired video dehazing benchmarks:

| Dataset | Type | Resolution | Scale | Notes |
| ------- | ---- | ---------- | ----- | ----- |
| UHV-4K | synthetic video | 3840x2160 | 100 videos, 2500 frames | new benchmark with depth, transmission, and flow |
| REVIDE | real indoor video | 2708x1800 | 42 train / 6 test | paired real data |
| HazeWorld | synthetic outdoor video | <=720p | 1016 train / 584 test | DAVIS, DDAD, and UA-DETRAC subsets |

UHV-4K is built from clean 4K videos in Inter4K and UVG. The authors estimate depth with Depth Anything V2, estimate flow with RAFT, synthesize haze with the ASM, and export five aligned modalities per frame: hazy input, clean ground truth, depth, transmission, and optical flow. Each video uses fixed beta and A_inf values for temporal coherence.

### Main Results

LiBrA-Net is compared with 16 baselines: eight single-image dehazers and eight video dehazers. The key UHV-4K numbers are:

| Method | UHV-4K PSNR | UHV-4K SSIM | UHV-4K LPIPS | 4K FPS | GFLOPs | Params |
| ------ | ----------- | ----------- | ------------ | ------ | ------ | ------ |
| DehazeFormer | 23.74 | 0.9313 | 0.0468 | 0.18 | 6.00K | 2.54M |
| UHDformer | 23.16 | 0.9381 | 0.0746 | 0.13 | 33.30K | 20.33M |
| ViWS-Net | 22.06 | 0.7742 | 0.3586 | 4.21 | 1.20K | 57.67M |
| 4KDehazing | 18.56 | 0.8608 | 0.1011 | 3.11 | 10.20K | 17.26M |
| LiBrA-Net | **24.28** | **0.9437** | **0.0401** | **25.27** | **0.28K** | 6.12M |

The important result is not only the PSNR. It is the fact that quality and speed hold together. LiBrA-Net is the only method in the table that exceeds 23 dB PSNR on UHV-4K while sustaining real-time native 4K throughput. Compared with the closest bilateral-grid route, 4KDehazing, it is about 8x faster at 4K and uses about 36x fewer GFLOPs.

On REVIDE, LiBrA-Net reaches 21.43 dB / 0.8678 / 0.2397, ranking first among video dehazing methods and approaching the strongest single-image baseline. On HazeWorld, where the resolution is only 720p and the resolution-decoupling advantage is weaker, LiBrA-Net scores 27.46 PSNR, slightly below ViWS-Net's 27.66, while achieving the best LPIPS.

### Temporal Consistency

The paper measures temporal stability with tOF and within-video PSNR standard deviation. LiBrA-Net's tOF is 0.004769, not the lowest. DCL and ViWS-Net are lower. But those methods score only 20.81 and 22.06 PSNR, trailing LiBrA-Net by 3.47 dB and 2.22 dB. The paper's point is that tOF rewards any form of frame similarity, including over-smoothing. It must be read jointly with fidelity. LiBrA-Net's position is: steadier than every single-image dehazer while retaining the highest restoration quality.

### Ablations

The ablations are especially informative:

| Variant | UHV-4K PSNR | Drop |
| ------- | ----------- | ---- |
| Full model | 24.28 | -- |
| w/o Chromatic Field | 22.02 | -2.26 |
| w/o Temporal Field | 22.09 | -2.19 |
| w/o HF-Refiner | 20.87 | -3.41 |
| w/o Cayley map | 21.69 | -2.59 |
| w/o resolution-adaptive slicing | 21.01 | -3.27 |
| w/o Lie regularization | 21.87 | -2.41 |

Three lessons stand out. First, the Chromatic and Temporal Fields contribute almost equally, so the grid factorization is not decorative. Second, the HF-Refiner matters most at 4K; removing it costs 3.41 dB, confirming that low-resolution grids cannot recover fine texture alone. Third, the Cayley map and Lie regularization materially improve stability and quality; they are not just elegant mathematical dressing.

## IV. Limitations

**Acknowledged by the paper.** First, the temporal window covers only about one second, so minute-scale coherence remains open. Second, UHV-4K assumes one beta and one A_inf per video, so mixed weather, local fog banks, smoke, and rain-fog mixtures are outside its scope. Third, paired real 4K hazy/clean video data still does not exist; UHV-4K fills an evaluation gap but cannot replace real capture.

**Observable on close reading.** Several additional risks are worth noting. First, the method strongly relies on the ASM and smooth depth-governed haze, which is reasonable for atmospheric haze but may not hold for non-uniform smoke, dirty glass, or lens glare. Second, UHV-4K annotations are generated by upstream models such as Depth Anything V2 and RAFT; they are rich, but not physical ground truth. Third, LiBrA-Net's advantage is clearest at 4K and above. On 720p HazeWorld, its edge over dense methods shrinks. Fourth, the Cayley map requires a 3x3 solve per pixel. The reported GFLOPs are low, but deployment still needs efficient kernels, or implementation overhead may eat into the theoretical resolution decoupling.

## One Sentence

LiBrA-Net splits 4K video dehazing into a low-frequency affine field and a high-frequency residual: the former is fused from two compact bilateral grids in the Lie algebra, and the latter is recovered from the original 4K input, yielding 24.28 dB and 25 FPS on UHV-4K.
