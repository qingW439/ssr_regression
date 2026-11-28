# RankUp + UVR-SimCSE 融合首轮：NO-GO（2026-11-05）

## 0. 结论

**RankUp-3D ARC + UVR-SimCSE 在匹配的 EmoBank lb500/seed0 fold0 配方下失败。** B3 的 MAE 从 B2 的 0.15104 退化到 0.33920，CCC/R² 接近崩溃；B1（Raw + ARC）同样退化到 0.35349。带标签 ARC（B4）只造成小幅退化（0.15716），说明**崩塌主要来自无标签排序项**，不是 UVR 编码器本身。

未继续跑 seed0 五折：fold0 已出现 R² 为负、MAE 翻倍的机制性失败；随后 B1 fold1 仍为 **0.31906**（B0 0.17222 / B2 0.16277），确认不是单折偶发。继续训练只会消耗 GPU，按预算纪律停止。按任务书“先定位冲突、最多一次受控修复”的规则处理。

## 1. 自动化状态

`dsh headless` 仍返回：

```text
MISSING_CREDENTIAL: llm-deepseek: no API key for provider route "deepseek-official"
```

因此 Supervisor 仍不能启动真实 Worker，本轮实验由当前会话直接执行。没有声称持续自动化已恢复。

## 2. RankUp-3D 适配实现

- 共享 RoBERTa/UVR 编码器 + 回归头；
- ARC：`Linear(768, 3×2)`，每个 VAD 维独立二分类；
- pair：batch 内所有 `(i,j)`，`i=j` 屏蔽；
- 类 0 = 第 i 个样本该维更大；标签相等时 soft target 为 `[0.5,0.5]`；
- 有标签 ranking：soft CE，按 pair/维平均；
- 无标签 ranking：同一无标签文本两次 dropout 为 weak/strong；weak pair 概率置信度 ≥0.95 的 pair，用 `T=0.5` 锐化后的 hard pseudo label 训练 strong；
- loss：`L1 + 0.2*(rank_sup + 1.0*rank_unsup)`（RankUp 官方 `arc_loss_ratio=0.2`, `arc_ulb_loss_ratio=1.0`）；
- 96 步读出阶段保持纯监督，以匹配 B0/B2。

## 3. Seed0 fold0 匹配结果

| 组 | 初始化 | 训练目标 | MAE | CCC | R² |
|---|---|---|---:|---:|---:|
| B0 | Raw | 纯监督 | 0.15624 | 0.4414 | 0.3074 |
| B1 | Raw | + full ARC | **0.35349** | 0.1593 | **−3.3655** |
| B2 | UVR | 纯监督 | **0.15104** | **0.4984** | **0.3469** |
| B3 | UVR | + full ARC | **0.33920** | −0.0465 | **−5.2684** |
| B4 | UVR | + labeled-only ARC | 0.15716 | 0.4900 | 0.3149 |
| B3w002 | UVR | + full ARC, weight 0.02 | 0.18063 | 0.4409 | 0.0864 |

B3 的 base（1600 步）MAE 0.35608，sup96 后仍 0.33920；模型基本退化为均值预测。B1 同样崩塌。

## 4. 失败归因

1. **无标签 ARC 是主要破坏源。** B4 去掉无标签排序后只退化到 0.15716；B3 保留它则崩溃。
2. **训练预算/学习率与官方 RankUp 不匹配。** 官方配置为 lr 1e-5、102400 iter、layer decay 0.75；本匹配实验使用 UVR 基线的 lr 1e-4、1600 步。ARC 的 pair CE 约 0.52，而 L1 约 0.009，辅助项在梯度上占主导。
3. **一次受控修复**：`arc_weight 0.2 → 0.02`，仍为 0.18063，未恢复到 B2；不继续扫权重。
4. 不是 UVR 编码器单独的问题：B1（Raw+ARC）也崩塌，说明当前 ARC 训练方式本身与匹配预算不兼容。

## 5. 判定

- **B3 vs B1**：UVR 没有改善 RankUp；两者都崩。
- **B3 vs B2**：RankUp 在 UVR 上没有额外收益，严重退化。
- **B3 vs B4**：无标签排序有明确负贡献。

`RANKUP_UVR_FUSION = NO-GO`。不把 UVR 强行保留在 RankUp 中；也不宣称 RankUp 对所有三维回归无效，只判定**当前匹配配方 + 当前 ARC 实现**不成立。

## 6. 资产

- 工作树：`/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-rankup-m1`
- 分支：`m1/rankup-m1`
- 代码：`scripts/vad_research/train_rankup_base.py`
- 结果：`outputs/rankup_m1/{b1,b3,b4,b3w002}_seed0_fold0/sup_predictions.parquet`、`.../sup96/oof_predictions.parquet`
- 未删除任何历史 RankUp/UVR 资产；未读取 DEV/TEST。

## 7. 下一步（按任务书自动搜索）

RankUp 核心框架保留，但不再使用当前未经验证的 ARC 无标签排序配方。下一候选应更换**无标签信息利用方式**，而不是继续调 `arc_weight`/`lr` 扫描；优先选择与 ARC 信息源不同的方法，并继续与 B0/B2 匹配。
