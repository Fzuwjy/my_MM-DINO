# ViT-S CoLA 首轮实现（2026-09-21）

这是独立研究实现，基于集群 ViT-S 控制基础设施 `6b613607429186d4e80e0e88374f1e0aea14b43d`，
分支 `exp/whu-vits-cola`。作者模型、训练器、数据集和 metrics 文件未修改。
本地工作树 `D:\MM-DINO\work\my_MM-DINO-whu-vits-cola`；集群目录
`/mnt/csip-113/wjy/MM-DINO/repo-vits-cola`。首轮先用 ViT-S；ViT-L 尚未纳入。

## 模型合同

同一个冻结 DINOv3-S/16 LVD1689M，两个流（Optical、SAR）。SAR 重复成三个通道进入同一个
patch embedding。每 block、每方向、每个 Q/K/V/O/FFN-up/FFN-down 投影有独立增量：

`W0(x) + (8/16) B_L A_L x + lambda B_C Phi(pool(other)) A_C x`

`lambda=0.1` 可学习、不约束；两个 A 为 Kaiming，两个 B 为零。没有额外零门控。
Phi 为 `384→24→GELU→LN(24)→256→reshape(16,16)→LN(16,16)`，LN eps=1e-6。
池化仅对空间 patch，排除 CLS 和 4 个 storage token；增量作用于完整 token 序列。

同步三阶段：QKV 的跨模态上下文来自对方 block 输入；O 来自对方 attention 的 pre-proj
输出；两个 FFN 投影都来自对方 attention residual。上下文不 detach。
原 qkv masked-bias 层和 compute_attention、RoPE、LayerScale、MLP 顺序保留。
训练态每模态每 block 独立抽取 RoPE（rescale=2），在 checkpoint 外生成并传入。
非重入 checkpoint 仅覆盖 paired transformer block，闭包绑定当前 block。
Adapter 和 Decoder 直接使用原类，保留两次 FRM 和相应 BN 更新。
不 merge 任何增量，不启用 NAF、光学 stem、raw_logits 或缺失模态路径。

`--variant intra` 是同位置同 rank 的普通 LoRA 主对照，无 cross 参数；
`--variant cola` 有两套低秩分支和超网络。模态内增量、原 Adapter、Decoder 分组件独立种子，
公共初始化哈希相同；不是仅设置一个全局 seed。

| 版本 | 新增 backbone 可训练参数 | 全模型可训练参数 |
|---|---:|---:|
| intra | 2,654,208 | 32,051,600 |
| cola | 7,641,360 | 37,038,752 |

CoLA 比 intra 参数更多，本轮差值不能单独证明“跨模态条件”本身有效；若保留候选，
后续再加入 self 条件和多 seed。它也不同于历史作者 q/v rank=3 对照。

## 训练合同与诊断

封存 80/20 文件名单逐字节核对；原 WHU_Dataset 的训练裁剪、resize、flip 和 padding
行为保留。Optical common/ImageNet，SAR `/255` 后 float32、不另做 normalize。
50 epoch，seed42，FP32，实 batch8、accum1；AdamW 1e-4、wd0.01，cosine T50、eta_min1e-7；
原 soft-CE + Dice，各1，smooth0.05，ignore7。每5 epoch 原滑窗512/stride341，默认 crop batch32。
按图累计 int64 混淆矩阵与原 sklearn 计算定义一致，无全数据 GPU gather。
best_test、latest、e50 分别保存，完整曲线写 JSONL。

数据 loader 使用独立 generator，4 workers、每 worker 原 LRU 容量32；不按宿主机 MemTotal
擅自增大。两条研究运行共用这一采样协议；不宣称与历史脚本逐裁剪完全相同。
单卡，不启用 AMP、梯度裁剪、额外 dropout、rank8、缩短日程或更强正则化。

每5 epoch 增加固定诊断：train_source 和 test 各8幅图、每幅2个512裁剪，共32裁剪；
用私有 seed20260921 按文件名和几何选择，不看标签、类别、指标。
`fixed_crops.json` 冻结文件名、坐标、原图尺寸、split SHA256。两次运行应完全相同。
固定 batch2、无增强、eval + inference_mode，报告原联合 loss、逐类 IoU、mIoU、train-test gap。
联合 Dice loss 有 batch 依赖，因此固定诊断 batch 不随滑窗推理 batch 调整。
这组 train_source 来自训练图像，但不能声称这些精确窗口每个都被随机训练采样过。
固定 crop 对固定 crop 是主要诊断比较；完整 test 大图滑窗结果另列，不能混成同一个 gap。

诊断保存并恢复 Python/NumPy/Torch CPU/CUDA RNG 和每个模块的 train/eval 状态，不更新 BN。
不依据曲线自动早停、改 rank 或改变 best 选择。best 仍仅依据原完整 test mIoU。
本轮 best-test 属探索协议，不是独立泛化证据；后续结构调参应封存训练内 validation。

`steps.jsonl`：每个训练 step 的 loss；`train.jsonl`：epoch loss/LR/耗时/资源；
`branches.jsonl`：每 epoch 第1、2和末 step 的各层 lambda、intra/cross/base RMS、cross/base
比值和 A/B/超网络梯度范数，以及双模态输入均值、标准差、内容哈希；
`eval.jsonl`：固定裁剪两侧指标及完整 test 指标。第一步 A/Phi/lambda 梯度为零是 B=0
初始化的正常现象；第二步测试必须能启动。lambda 本身不等于实际贡献。

判读：train_source 持续改善而 test 连续退化才支持考虑容量问题；前期震荡和分支尺度增长
先查优化；两种方法均没有拟合收益或 cross 长期近零先查是否学起来。任何单条曲线都不是定论。

## 验收与当前限制

CPU 自动化验收使用仓库实际 DINO block，覆盖零增量训练/推理严格一致、完整后端输出与 BN
一致、原两次 FRM、跨模态双向依赖、梯度启动、checkpoint 输出/RNG/梯度一致、冻结权重
保存重载、公共初始化、精确参数量、固定裁剪预处理与原 loader 一致、RNG/模式/BN 隔离及
原指标一致性。故意缺失类别的 metric 测试会触发作者函数的 NaN warning，结果按其 nanmean 对齐。

```bash
cd /mnt/csip-113/wjy/MM-DINO/repo-vits-cola
source /opt/conda/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=/mnt/csip-113/wjy/MM-DINO/runtime/python-packages${PYTHONPATH:+:$PYTHONPATH}
python -m unittest discover -s tests/cola -v
```

默认是小尺寸真实 TIFF 前向部署检查，不更新参数：

```bash
cd /mnt/csip-113/wjy/MM-DINO/repo-vits-cola
source /opt/conda/etc/profile.d/conda.sh
conda activate base
bash scripts/run_whu_vits_cola.sh --variant cola
bash scripts/run_whu_vits_cola.sh --variant intra
```

Titan Xp 仅部署；新增4090节点用于真实512 batch8 的显存、吞吐与 batch32 推理短测，不能从
64尺寸部署检查推断通过。4090或3090的前台短测命令：

```bash
cd /mnt/csip-113/wjy/MM-DINO/repo-vits-cola
source /opt/conda/etc/profile.d/conda.sh
conda activate base
bash scripts/run_whu_vits_cola.sh --variant intra --mode smoke
bash scripts/run_whu_vits_cola.sh --variant cola --mode smoke
```

短测使用原数据增强，2次 batch8 更新（包含 Adam 状态），并核对 eval batch32 与 batch1。
两步后另保存 train_smoke.json，后续检查即使失败也保留显存信息。固定32裁剪也在短测中
真实运行，核对 BN buffer 不变。短测指标仅用于执行验收，不是拟合或泛化结果。
批大小对比使用 rtol1e-3/atol1e-4，并要求 argmax 一致率至少99.9%，保存实际误差和一致率。
依据：原 MM-DINO 后端在4090、原 cuDNN TF32 设置下 B1/B32 差7.0523e-5，RMS6.0683e-6，
argmax一致99.9981%。不为此改变原数值配置。同 batch 零增量和完整后端验收仍要求逐位相同。
检查 smoke.json 的 train/eval reserved、耗时，保留2–4 GiB余量。测试 OOM 则保留现场，先
诊断再统一决定两组 microbatch；当前实现不会自动降 batch 或增加梯度累积。
允许单独指定 `--eval-batch` 后重测；正式命令也应传入验收后的相同值。
输出目录若存在会拒绝覆盖，重测用 `--output` 指定新的项目 NFS 子目录。

仅在两组短测通过且用户拿到3090后，用户本人前台执行正式运行：

```bash
cd /mnt/csip-113/wjy/MM-DINO/repo-vits-cola
source /opt/conda/etc/profile.d/conda.sh
conda activate base
bash scripts/run_whu_vits_cola.sh --variant intra --mode train
bash scripts/run_whu_vits_cola.sh --variant cola --mode train
```

运行入口拒绝 Titan Xp/4090 正式训练，允许4090有限步短测；记录 Git SHA、权重 SHA、初始化哈希和协议。
首次实现没有恢复训练入口；latest 保留模型、优化器、scheduler、epoch及主进程 RNG，
但未保存 persistent-worker RNG，不可声称中断恢复逐裁剪等价。

## 文献和代码依据

- CoLA: https://github.com/peterwisu/CoLA ，参考提交 `d26d99d07a8dcdaf541db1ca1b345d0b1b5a4644`。
- MM-DINO: https://github.com/KimotaQY/MM-DINO 。
- 本地总设计：`D:\MM-DINO\idea\CoLA_ViTS首轮统一设计_20260921.md`。
- 本实现是 CoLA 向共享 DINOv3 的三阶段移植，不声称是 CoLA 官方直接实现。
