# 2026-09-21 ViT-S CoLA 实现验收

已实现、推送 origin、部署并通过4090有限步验收；未启动正式训练。
分支 `exp/whu-vits-cola`，实测代码提交 `76c6f419483a685a6a8bbf28d287cc8eb6c51d2c`。
初版模型和测试提交 `d3225c3`；后续变更只改进短测证据保存和诊断验收。

## 环境与位置

- 部署：`/mnt/csip-113/wjy/MM-DINO/repo-vits-cola`，origin 为用户仓库，upstream push 禁用。
- 4090节点：`root@172.17.171.180:39210`，主机 csip-180，PyTorch2.7.1+cu118。
- PyTorch 可见总显存23.5135 GiB，容器内存上限66,571,993,088 bytes（62 GiB）。
- Titan Xp节点 `root@172.17.175.101:25897` 的同一 NFS 可见此部署；没有在那里运行训练。
- NFS证据：`/mnt/csip-113/wjy/MM-DINO/outputs/cola-acceptance-20260921/`。
- 有效目录 `intra-76c6f41`、`cola-76c6f41`。各含 config、steps、train_smoke、
  eval_batch_comparison、fixed_crops、smoke；CoLA 另含 full_image.json。
- 本地证据复制在工作树 `outputs/cola-acceptance-20260921/`，受 .gitignore 排除。

## 验收结果

本地 CPU：12通过、1项CUDA测试跳过；集群CPU/CUDA：13项全部通过。
包含同批零增量输出逐位一致、原后端两次FRM与BN buffer一致、双向跨模态依赖、
checkpoint输出/RNG/梯度一致、冻结参数不变与保存重载、相同公共初始化、配对样本独立性、
确定性裁剪与原预处理一致、评估隔离及原指标定义一致。

两种模型都严格加载了官方S权重，SHA256：
`08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d`。
真实TIFF的64尺寸部署前向均通过。下表使用512、FP32、实batch8、无累积、两步AdamW，
推理batch32；单位GiB，allocated/reserved为PyTorch峰值。

| 验收 | intra | CoLA |
|---|---:|---:|
| 训练 peak allocated | 9.4454 | 9.5020 |
| 训练 peak reserved | 9.9824 | 10.0449 |
| 两步耗时，含取数和遥测（秒） | 16.70 | 18.85 |
| 推理 peak allocated | 11.3458 | 11.4023 |
| 推理 peak reserved | 12.4063 | 12.4199 |
| batch1/32 最大输出绝对差 | 8.4497e-5 | 9.3719e-5 |
| batch1/32 argmax一致率 | 99.9966% | 99.9943% |
| 固定训练来源裁剪 / test裁剪 | 16 / 16 | 16 / 16 |
| 固定裁剪评估不更新BN | 通过 | 通过 |
| 冻结基础权重哈希不变 | 通过 | 通过 |

CoLA 单幅完整 WHU 大图滑窗也通过：`NH49E001014.tif`，有效像素20,579,408，
512/stride341/batch32，14.57秒（含取数和CPU指标累计），peak allocated12.1884、
reserved13.7246 GiB。单图检查覆盖完整输出和覆盖次数缓冲区，不是完整20图评测。

## 分支是否启动

在真实数据的144个跨模态分支中：

- 第1步：144个 cross-B 有非零梯度；cross-A、Phi、lambda 为零，符合 B=0 初始条件。
- 第2步：144个分支的 cross-B、cross-A、Phi 和 lambda 均有非零梯度。
- 两条运行 base、Adapter、Decoder、intra 初始化哈希相同，固定裁剪清单逐字节相同。

这些只能证明实现能启动学习，不能证明完整训练后跨模态分支有效；两步后的低mIoU不作为研究结果。

## 保留的失败与解释

`intra-d3225c3` 首轮短测完成两步和batch32前向，但被过严的batch1/32数值门槛拦下，
没有OOM；现场保留。原MM-DINO后端对比也出现 max7.0523e-5、RMS6.0683e-6、
argmax99.9981%的跨batch差异。原 cuDNN TF32=True、matmul TF32=False 设置没有改变。
跨batch验收改用 atol1e-4/rtol1e-3，同时记录误差并要求argmax≥99.9%；
同batch零增量等价验收仍保持逐位一致。

## 后续

首轮维持rank16、原AdamW/学习率/wd/50epoch；不因这些短测指标改超参数。
每5epoch固定诊断比较保持相同batch2，其类别组成固定但不保证全面；应看同组随时间的变化，
并结合逐类IoU和完整test曲线，不把单次train-test差距直接解释成过拟合。
正式前台命令见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。未运行50epoch，未检验多seed。
4090测试已完成，可以释放测试节点；正式训练等待3090。4090的速度不能外推为3090吞吐。
当前短测直接取训练样本；4-worker/cache32的整轮主机内存与长期稳定性仍需在正式环境观察。
