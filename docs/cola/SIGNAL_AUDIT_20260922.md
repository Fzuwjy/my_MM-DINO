# CoLA 最终结果与无训练信号审计（2026-09-22）

结论：当前迁移版本没有展示明显额外收益；但 cross 分支不是未启动，Phi 也不是普遍静态，
信号没有被 SampleAdapter 普遍消掉。它改变了最终预测，而这种已训练模型内的依赖性，
不能等同于另一传感器 conditioning 相对普通 LoRA 的边际价值。

## 1. 最终完整 test 结果

| 方法 | best-test mIoU | best epoch | e50 mIoU | 可训练参数 |
|---|---:|---:|---:|---:|
| S-intra16 | 55.4863% | 45 | 55.1943% | 32,051,600 |
| S-cola16 | 55.6536% | 45 | 55.3981% | 37,038,752 |
| 差值 | +0.1673 pp | — | +0.2038 pp | +4,987,152 |

两组均完成50epoch，complete.json确认冻结基础权重哈希未变。单seed、额外参数约15.56%，
不能据此宣称稳定的跨模态收益，也不能外推为 CoLA 方法整体无效。

## 2. 审计范围和复现

- 实测诊断代码：`b72025e`（模型训练代码保持原样），纯eval、无优化器步骤。
- 节点 `172.17.175.90:28136`，RTX2080Ti，PyTorch2.7.1+cu118。
- CoLA best e45和e50分别检查：原固定清单32个512裁剪，训练来源/test各16，合计16幅不同原图。
- 推理batch1，跨样本Phi逐张累积；32裁剪不是32幅独立图像。
- block索引 `[2,5,8,11]`，两个目标模态，Q/K/V/O/up/down，共48个投影。
- cross/intra统计同时保存patch-only和all-token口径；正文使用patch-only。
- on/off关闭整个编码器的全部cross contribution，保留全部intra与原后端。
  因此下游差值是累积效应，不是某一单层的局部因果贡献。
- 实际融合权重约 Optical=0.52869，SAR=0.47131；调用原Adapter并校验两个融合输出slot相同。
- 审计前后整个state_dict哈希一致，参数与BN不变。
- best32裁剪含完整特征保存约245秒；e50统计复核约141秒；峰值reserved均约0.633GiB。
- 新增3项测试通过：并行/正交/反向融合、静态/动态Phi及跨图配对、采集hook不改变输出。

命令（审计输出目录必须不存在）：

```bash
cd /mnt/csip-113/wjy/MM-DINO/repo-vits-cola
source /opt/conda/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=/mnt/csip-113/wjy/MM-DINO/runtime/python-packages
python -m research.cola.signal_audit \
  --checkpoint /mnt/csip-113/wjy/MM-DINO/outputs/whu-vits-cola-train/best_test.pt \
  --run /mnt/csip-113/wjy/MM-DINO/outputs/whu-vits-cola-train \
  --output /mnt/csip-113/wjy/MM-DINO/outputs/cola-signal-audit-20260922/best45 \
  --save-features
```

e50使用e50.pt及独立目录e50，不重复保存大特征。原训练指标仍以3090/batch32大图评测为准。
2080Ti/batch1的cross-on固定crop mIoU与原固定诊断相差不到0.001个百分点，但不能据此
保证所有跨硬件/批大小计算逐位相同。

## 3. Cross强度、方向和Phi动态性

以下best统计中，C/L与方向是该block的12投影×32裁剪的中位数；Phi统计先对每投影的
32样本计算，再对12投影取中位数。完整投影、样本和split细分保存在原始JSON。

Phi相对变化定义为 `RMS(Phi_i - mean_i Phi_i) / RMS(Phi_i)`，是样本间变化相对总幅值，
同时另存逐元素总体方差及所有样本对/不同原图样本对的余弦。高余弦应与此比例一起读。

| Block | RMS(C)/RMS(L) | cos(C,L) | Phi相对变化 | 不同原图Phi余弦均值的中位数 |
|---|---:|---:|---:|---:|
| 2 | 1.969 | 0.778 | 16.77% | 0.970 |
| 5 | 2.692 | 0.475 | 36.11% | 0.862 |
| 8 | 4.058 | 0.571 | 48.97% | 0.742 |
| 11 | 2.476 | 0.389 | 45.65% | 0.777 |

所以没有普遍的 cross << intra。训练末次遥测中144个cross分支的A/B/hyper梯度均非零。
有局部冗余：例如block2 Optical Q/V的C/L方向余弦约0.93/0.94、SAR Q约0.98；
但其它投影包括低相关甚至负相关，不能概括为全分支复制LoRA。
Phi浅层若干投影变化较小；中后层总体变化明显，不支持普遍退化成常量矩阵。
这仍不证明Phi变化都被下游有效利用，也不证明这些变化必须来自另一模态。

## 4. Adapter是否抵消了修正

P定义为原Adapter的共享project+resize之后、weighted sum之前的每模态特征；
原backbone四层输出也单独保存。G为实际融合输出。

为了避免把0.5权重的自然缩小误判成抵消，定义：

`retention = RMS(deltaG) / (RMS(wO*deltaO) + RMS(wS*deltaS))`。

同时保存相对于正交合成参考的比例，分母为 `sqrt(RMS(wO*deltaO)^2 + RMS(wS*deltaS)^2)`。
两个等幅、正交信号在第一种定义下约0.707，不能把这个值直接说成严重丢失信息。

| Block | cos(deltaO,deltaS) | retention | 相对正交参考 | RMS(deltaG)/RMS(G_on) |
|---|---:|---:|---:|---:|
| 2 | 0.635 | 0.907 | 1.255 | 18.19% |
| 5 | 0.253 | 0.797 | 1.117 | 16.01% |
| 8 | 0.356 | 0.824 | 1.162 | 87.62% |
| 11 | -0.138 | 0.712 | 0.950 | 50.91% |

表为32裁剪的中位数，归一化比值是信号量，不是信息量或性能贡献。
最后一尺度确有部分反向抵消，但融合变化仍然明显，不符合“deltaG几乎没了”的描述。
其余三尺度的两模态修正通常同向。加权恒等式中位数最大绝对残差在1e-7～8e-7量级，
说明采集对齐与原实现的线性融合一致。

## 5. 到达最终分类输出了吗

原Decoder末端行为保持不变；这里称output scores，避免把其已有激活后的输出误叫raw logits。

- best：输出相对变化RMS中位数31.77%；像素预测改变比例中位数7.25%、均值11.18%。
- e50：对应34.59%、8.64%、12.72%。
- best固定test裁剪mIoU：cross-on47.505%、off38.479%；固定训练来源50.855%→46.464%。
- e50固定test裁剪：47.209%→37.891%；固定训练来源50.868%→45.504%。

这些是16个test裁剪的诊断结果，不是20幅完整test结果。off去掉了已经共同适应的整条
分支，既删除额外容量也删除动态修正，不能用约9个百分点的下降声称cross conditioning
给普通LoRA提升了9个百分点。它说明已训练CoLA依赖该分支，且影响确实到达分类结果。
因此“到达logits却对该模型预测完全没作用”也不符合本次on/off观察。

## 6. e50复核与下一步

e50的四层C/L中位数为1.962/2.692/4.051/2.503，Phi相对变化16.79%/36.19%/48.93%/45.71%，
融合retention为0.907/0.796/0.824/0.711。与best的结构性判断一致。

目前更准确的问题是：分支强、Phi会变、信号能传下去，为什么完整模型相对普通LoRA
只多约0.2个百分点？审计尚未区分额外动态容量、跨模态条件来源、优化分工和任务监督限制。

下一条训练优先S-self16：保持两条低秩路径、r16、alpha8、lambda初值0.1、144个hyper、
参数量37,038,752、Adapter/Decoder、公共初始化及整个训练/评测协议一致；
仅把QKV的other block input换成own block input、O的other pre-proj attention换成own，
两个FFN的other attention residual换成own。不要仅在一个阶段改成self，也不要删掉cross路径。
这次审计没有实现或启动新训练；2080Ti只执行诊断。

Self与cross差距仍需结合多seed评估，不以某个小阈值宣布有效。当前没有足够证据优先
改Adapter，也没有证据证明global pooling就是主要瓶颈；空间条件化可作为后续假设，
不应当作本次审计已经证实的结论。

## 7. 证据位置

NFS：`/mnt/csip-113/wjy/MM-DINO/outputs/cola-signal-audit-20260922/`。
`best45/features_00.pt`至`features_31.pt`保存每裁剪四尺度的raw、P_O/P_S、G及输出on/off；
summary.json含checkpoint SHA256、epoch、权重、Phi统计和crop指标；samples.jsonl保存逐样本六类诊断；
phi.pt保存所有样本的小矩阵。e50目录保留统计和Phi，未重复存大特征。

本地小证据和热图：`outputs/signal-audit-20260922/{best45,e50}/`，Git忽略；
完整训练最终指标：同目录的`intra-final`、`cola-final`。
生成汇总和热图：`python -m research.cola.summarize_signal_audit <audit-dir> --plot`。
