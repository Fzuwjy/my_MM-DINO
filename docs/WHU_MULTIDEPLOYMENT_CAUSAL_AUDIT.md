# WHU 多部署缺失模态：基座与因果审计协议

## 1. 研究目标修订

后续方法不再只优化 `Full train -> SAR-only deploy`。对于 WHU 的两个规范槽位
`Optical=NIR-R-G` 与 `SAR=single-channel`，同一个模型至少必须支持：

- `Full = Optical + SAR`；
- `Optical-only`；
- `SAR-only`。

未来候选必须报告三端点，不能靠牺牲一个缺失状态换另一个状态的增益。最终方法
仍使用一套共享 MM-DINO，而不是为三个端点训练三套 decoder。

## 2. 现有 WHU Run C 能回答什么

历史 Run C E50（seed 42，SHA256 `e8291383...7824`）训练时只采样
`Full/SAR`，没有训练 `Optical-only`。因此：

- 它仍是已完成的 S+O->S 历史锚点（SAR Test `46.738`）；
- 可以对三个端点做只读机制审计；
- 不能作为通用三端点方法的充分 baseline；
- Optical-only 结果必须明确标注为 unseen-state 描述值。

## 3. WHU 没有官方 Val

WHU 仓库只有 80 张官方 Train 与 20 张官方 Test。历史 A/C 用满 80 张 Train，
E50 只在 Test 报一次。因此官方 Test 上的只读审计可以解释机制，但不得反复用来
选择模块、尺度、loss 或超参数。

若 WHU 成为主开发集，必须从官方 80 张 Train 中建立新的 group-disjoint
development Train/Val，并在该新协议上重训最简单的三状态 baseline。官方 20 张
Test 保留到方法冻结后一次性运行。历史 A/C 继续保留，但不与新 split 的数值做
单变量比较。

建议 split 合同：

- 以地图分幅前缀（例如 `NH49E006`）为 group；
- 同一 group 不得跨 development Train/Val；
- 目标约 64/16 scenes；
- 只用 GT 类别像素直方图做确定性分层，不读取模型预测；
- 输出场景清单、group 清单、全类支持及 SHA256，并永久冻结。

## 4. 已准备的历史 Run C 三端点审计

### 4.1 Feature intervention oracle

`diagnose_whu_multideployment_oracle.py` 对 Full、Optical-only、SAR-only 使用同一
cached DINO backbone。对两个缺失端点分别执行：

`target_stage <- (1-alpha)*target_stage + alpha*paired_full_stage`

首轮固定 12 个单尺度节点（Adapter/FRM/SE 的 P2-P5）和 `alpha=1`。为避免 WHU
大图同时重建 27 份整图 logits，审计使用确定性的、不重叠的 512 crop，边缘余数
显式排除并报告 coverage。每个 crop 的混淆矩阵在线累积，不保存内部特征。

该结果只回答 Full tensor 在下游是否具有因果充分性，不回答它是否能由缺失端点
预测，也不允许选择最终方法。

### 4.2 Pairwise gradient audit

`diagnose_whu_multideployment_gradients.py` 在官方 Train crop 上计算：

- Full vs Optical；
- Full vs SAR；
- Optical vs SAR。

每对都在 train-BN/eval-BN 下报告 Adapter、FRM、SEFusion、PRN、head 的 cosine、
norm ratio 与负梯度比例。脚本不构造 optimizer，逐端点恢复 RNG 与 persistent BN，
所有 `.grad` 必须保持 `None`。

### 4.3 Endpoint recoverability

`diagnose_whu_multideployment_recoverability.py` 分别以 Optical-only/SAR-only 的
Adapter-P5 与 post-FRM P2 为输入，预测“该端点错误时，paired Full 是否正确”。
这两个位置来自先行 Oracle 的因果资格，而不是按表征 gap 大小选择。线性 probe 使用
官方 Train 的固定 disjoint crops 拟合，在官方 Test 描述性评估；控制包括 GT 类先验
和类内 shuffled target。它不会训练 MM-DINO，但因为评估读取 Test，仍不能用于方法
选择。

## 5. 下一阶段真正需要的新 baseline

新的通用 baseline 应统一采样 `Full / Optical / SAR` 三种 homogeneous batch state，
同时评估三个端点。第一版不加蒸馏、重建、router、专家或新模块，只回答最简单的
三状态 modality dropout 能做到什么。

后续候选至少满足：

1. 三端点平均 mIoU 高于 matched 三状态 baseline；
2. 两个 missing endpoint 都不能出现预注册的明显退化；
3. Full endpoint 保持 guard；
4. 报告 worst-endpoint mIoU，而不只报平均值；
5. 在 development Val 冻结结构后，才运行官方 Test。

在新 development split 与 matched baseline 完成前，不应根据历史 Test 审计直接
写新模块。
