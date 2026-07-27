# MM-DINO 对话迁移与当前状态

最后更新：2026-07-27。

## 当前目标

忠实复现 MM-DINO 在 WHU-OPT-SAR 上的 DINOv3 ViT-S/16 双模态结果。第一阶段
服从作者公开实现，包括每 5 epoch 在 test 上评估并按 test mIoU 选择 best；不因
方法学偏好擅自改模型、数据、增强、训练日程、推理或指标。

官方发布权重已经评估成功：mIoU `54.1164%`，对应权重名中的 `54.12`。结果位于：

```text
/root/autodl-tmp/mm-dino/outputs/evaluation/whu_vits16_multi_official_fp32
```

## 仓库与 Git 线

```text
本地开发仓库  D:\MM-DINO\work\my_MM-DINO
只读官方镜像  D:\MM-DINO\official\MM-DINO
服务器仓库    /root/my_MM-DINO
origin         https://github.com/Fzuwjy/my_MM-DINO.git
upstream       https://github.com/KimotaQY/MM-DINO.git
```

- `official`：严格对齐 `upstream/main`，不得加入项目提交。
- `exp/local-adaptation`：早期可移植工程与官方权重评估线，不用于忠实训练。
- `exp/faithful-whu-reproduction`：当前工作线；服务器也应使用这条线。

作者的 `tasks/segmentation/` 保持与 `official` 零差异。复现兼容工作放在外部：

```text
scripts/prepare_faithful_whu.py
scripts/run_seeded_official.py
scripts/probe_official_whu_memory.py
scripts/whu_label_dtype_compat.py
scripts/whu_cache_compat.py
splits/whu/official_train.txt
splits/whu/official_test.txt
docs/FAITHFUL_WHU_REPRODUCTION.md
```

## 服务器环境与运行约束

```text
SSH       root@connect.bjb2.seetacloud.com:32123
Conda     mm-dino
数据      /root/autodl-tmp/mm-dino/datasets/whu-opt-sar
权重      /root/autodl-tmp/mm-dino/weights
GPU       RTX 5090 32 GB
PyTorch   2.7.1+cu128
```

新服务器终端访问 GitHub 前必须先 `source /etc/network_turbo`。正式训练及约 5 分钟
以上的评估由用户本人以前台方式运行；助手只准备命令并只读检查进度。不得默认已经
进入仓库或激活 Conda。

## 忠实协议

- 完整 80 幅 `train_list.txt`，不拆 val；20 幅 `test_list.txt`。
- 50 epoch；每 5 epoch 在 test 上评估并按 test mIoU 选 best。
- FP32，batch size 8，单卡 `torchrun` DDP。
- `find_unused_parameters=True`；不使用 BF16、梯度累积或 resume。
- AdamW，单卡 LR `1e-4`，weight decay `0.01`。
- Cosine scheduler：`T_max=50`，`eta_min=1e-7`。
- 滑窗推理 batch 32；seed 42。

显存探测已通过：训练 batch 8 峰值保留 `11.08 GiB`；滑窗 batch 32 峰值保留
`15.66 GiB`。

## 已确认的两个公开代码运行问题

### 1. WHU 标签类型

公开数据集返回 `int32` 标签，但 soft cross-entropy 中的 `torch.gather` 要求 `int64`
索引，原样运行会立即报错。经用户明确同意，外部兼容层只执行 `int32 -> int64`；
标签值不变，作者文件不改。

### 2. 每 worker 容量 100 的全图缓存造成主机内存压力

第一次正式运行完成 epoch 1-5 和第 5 轮全部 20 幅 test 推理，随后在指标汇总阶段
收到 `SIGKILL (-9)`。失败现场：

```text
/root/autodl-tmp/mm-dino/outputs/faithful-whu-author-protocol/DINOv3/WHU_20260727_214505
/root/autodl-tmp/mm-dino/outputs/faithful-whu-author-protocol/formal_train_seed42.log
```

平均训练 loss：

```text
epoch 1  2.0483856
epoch 2  1.9149497
epoch 3  1.8682649
epoch 4  1.8173699
epoch 5  1.7878761
```

第 5 轮 test 推理 20/20 完成，约 71 秒；`test_metrics.json` 仍为空，说明终止发生在
全量预测/标签拼接和指标阶段。容器 `memory.max=96636764160`（约 90 GiB）、
`memory.high=92341796864`（约 86 GiB）、无 swap，失败后记录到 348,095 次
`memory.high` 事件。没有 CUDA OOM；GPU显存不是原因。

作者默认每个持久化 worker 各自缓存最多 100 幅 optical、SAR 和转换后的 label。
四个 worker 在五轮后占据大量主机内存，再叠加官方评估保留和拼接全部全分辨率预测，
形成持续内存压力。cgroup 没记录 `oom_kill`，因此更精确地说是内存压力下的外部
`SIGKILL`，而非已被内核计数确认的 cgroup OOM。

当前最小修复是在外部 `scripts/whu_cache_compat.py` 中把每 worker 缓存容量从 100
限制为 64。训练集只有 80 幅，因此原容量 100 实际最多保留 80 幅；容量 64 仍保留
约 80% 的工作集。根据失败前四个 worker 合计约 56-58 GiB 的RSS估算，这将释放
约 11-12 GiB，使原先 86-90 GiB 的压力降至约 75-79 GiB，同时尽量保持缓存命中率
和训练速度。它只改变磁盘重读频率，不改变文件、随机数、裁剪、增强、tensor、评估
或指标。若第 5 轮仍出现内存压力，下一档才降到 50，而不是直接降到 2。暂不替换
作者的指标实现；下一次第 5 轮评估将用于验证原始指标路径能否完成。

## 继续工作前的注意事项

1. 失败目录与 `formal_train_seed42.log` 必须同时保留。作者 `clean_logs()` 会删除
   checkpoint 数量不超过 2 的旧目录，且下一次 `tee` 会覆盖同名总日志。因此重新
   训练前必须把两者移入独立归档目录。
2. 当前 checkpoint 是 epoch 4 保存的 `WHU_checkpoint.pth`；epoch 5 因评估后被杀，
   尚未执行该轮 checkpoint 保存。忠实基线不使用 resume，修复后从 epoch 1 重跑。
3. 拉取包含 cache 兼容层的最新 `exp/faithful-whu-reproduction` 后，先重跑 batch 8
   探测并确认打印的三个 cache capacity 均为 64。
4. 然后以前台方式从头启动正式训练；第 5 轮 test 指标成功写入后再判断修复完成。
5. 若容量 2 后仍在指标阶段出现内存问题，才考虑外部增量 7x7 混淆矩阵；实施前必须
   用同一官方 checkpoint 验证 confusion matrix 和所有指标完全一致。
