# RankUp 适配忠实性审计与一次受控官方配方复核（2026-11-05）

## 0. 结论

**UVR + RankUp 仍为 NO-GO。** 官方配方适配后崩塌消失，但并没有形成稳定、可晋级的收益：

- fold0 上 B3 相对 B2 的 MAE 仅改善 0.0012（未达 0.002），Pearson/CCC/R² 有大幅改善；
- 但 B3 fold1 反而比 B2 差 0.0062，方向不稳定；
- B2 官方配方五折 MAE 已到 **0.18087**，明显差于冻结 UVR 的 seed0 0.16762 / 三种子均值 0.17106；该训练配方本身欠拟合，无法据此声称超过 UVR。

因此不把 UVR 强行作为 RankUp 的组件一；RankUp 框架保留，下一候选必须改变无标签信息利用方式。

## 1. 实现审计（对照 RankUp 2024 官方源码）

| 项 | 审计结果 |
|---|---|
| ARC logits | 每个 VAD 维独立 `2` 类；pair logits 为 `arc_i - arc_j`。官方当前轴为 `arc_j - arc_i`，但 pair 矩阵对称、targets 同步反向，整体损失等价 |
| pair 正反方向 | class 0 = 第一个样本该维更大；与 targets 的 `sign(y_i-y_j)` 一致 |
| V/A/D 独立排序 | `Linear(768, 3×2)`，reshape `[B,3,2]`；3D 必需适配，非原论文标量 ARC |
| 相等标签 | target 0.5/0.5；官方 `(1+sign(diff))/2` 在 diff=0 时也是 0.5，语义一致 |
| 无标签 pair 置信度 | 随机初始化 batch 上覆盖率 **0%**；因此 1e-4 首次崩塌不能归因于早期 pseudo-rank，需区分后续激活/优化问题 |
| weak/strong 增强 | **不完全忠实**：官方使用预生成文本增强；本实现用 dropout / token masking。已明确记录为适配偏差 |
| pseudo stop-grad | 是，weak pair 概率与 pseudo label 均在 `torch.no_grad()` 中生成 |
| 损失归一化 | labeled pair 与 unlabeled pair 均按有效 pair×维求 mean CE；与官方 scalar mean 的 3D 扩展一致 |
| 梯度 | 初始 batch：reg 4.25、ARC supervised 1.96、cos=0.016；`arc_weight=0.2` 时 ARC 梯度约 9%，不是初始即压倒性 |
| 官方配方差异 | 官方 lr 1e-5、102400 iter、layer decay 0.75；原匹配实验 lr 1e-4、1600 iter，是崩塌的主要配方风险 |

**未发现 pair 方向、stop-grad 或归一化的程序错误。** 主要偏差是文本强增强缺失和官方训练预算未匹配。

## 2. 一次受控官方配方复核

预先锁定：

- lr **1e-5**、weight decay **5e-4**、layer decay **0.75**；
- 3200 步、batch 16（官方 102400 步的预算受限适配）；
- ARC ratio 0.2 / 1.0，T=0.5，p_cutoff=0.95；
- strong view = token masking 15%（无官方预生成增强语料时的文本级替代）；
- 所有臂共用 96 步纯监督读出。

### Seed0 fold0

| 组 | 方法 | MAE | Pearson | CCC | R² |
|---|---|---|---:|---:|---:|---:|
| B0 | Raw 纯监督 | **0.16978** | 0.5388 | 0.4838 | 0.2617 |
| B1 | Raw + ARC | 0.17712 | **0.5835** | **0.5548** | 0.2107 |
| B2 | UVR 纯监督 | 0.17220 | 0.4621 | 0.4053 | 0.1817 |
| B3 | UVR + ARC | **0.17100** | 0.5644 | 0.5507 | **0.2379** |

B3 vs B2：ΔMAE **−0.0012**，Pearson +0.102，CCC +0.145，R² +0.056。
B3 vs B1：MAE 更好，Pearson 略低，CCC 接近。

这是**辅助指标强、MAE 弱的混合信号**，且 fold1：

| fold | B2 MAE | B3 MAE |
|---:|---:|---:|
| 0 | 0.17220 | 0.17100 |
| 1 | 0.18205 | 0.18826 |

方向翻转，不能支持稳定增益。

### B2 官方配方五折

| MAE | Pearson | CCC | R² |
|---:|---:|---:|---:|
| **0.18087** | 0.47772 | 0.42543 | 0.21968 |

该配方本身已明显差于冻结 UVR seed0 0.16762，因此无法满足“达到或超过 UVR 基线”的最终条件。B3 未继续跑完五折，避免在已明确欠拟合的配方上消耗 GPU。

## 3. 判定

- `RANKUP_UVR_OFFICIAL_RECIPE = NO-GO`（无稳定 MAE 增益，且配方低于 UVR）。
- RankUp 核心框架保留；ARC + 当前无标签 pair 信息不足以成为组件一。
- 下一候选应改变**无标签信息源/训练信号**，不是继续改 `arc_weight`、学习率或 layer decay。

## 4. 自动化状态

`dsh headless` 仍为 `MISSING_CREDENTIAL`，Supervisor 不能启动真实 Worker；本轮实验在当前会话直接执行，不声称跨会话自动循环已恢复。

## 5. 资产

- 工作树：`/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-rankup-m1`
- 分支：`m1/rankup-m1`
- 审计脚本：`scripts/vad_research/audit_rankup.py`
- 官方配方训练：`scripts/vad_research/train_rankup_official.py`
- 结果：`outputs/rankup_m1/rankup_audit.json`、`outputs/rankup_m1/official_seed0_summary.json`、`outputs/rankup_m1/off_*/...`
- 未删除历史 RankUp/UVR 资产；未读取 DEV/TEST。

## 6. 追加候选：Rank–Regression Consistency (RRC)

在官方配方 B3 上增加回归 pair 顺序与 ARC pseudo-rank 的 hinge 一致性项（权重 0.1，margin 0.05）：

| 组 | MAE | Pearson | CCC | R² |
|---|---:|---:|---:|---:|
| B3 | 0.17100 | 0.5644 | 0.5507 | 0.2379 |
| B3 + RRC | **0.18668** | 0.2482 | 0.2281 | **−0.1761** |

RRC 在 fold0 明显破坏回归，判定 NO-GO；不继续该方向。该结果不改变“当前 ARC 无标签信息利用不足”的诊断，但说明简单把排序概率耦合进回归头不是有效补偿。
