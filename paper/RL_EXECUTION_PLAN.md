# BlindSpot-RL 执行规划

## 一句话目标

先用 SFT 让 policy 稳定输出可判定的 evaluation criteria，再用
BSC-RL 检验一个研究假设：开放式评估维度 elicitation 能否通过
verifiable semantic reward 产生可报告的 human-gold coverage change，同时不靠冗余或幻觉刷分。

核心定位不是“又训了一轮模型”，而是：

- SFT 解决格式、基本质量、常见维度覆盖
- RL 阶段测试 hard blind-spot coverage 是否出现可报告 coverage change，且冗余、幻觉和无效项受控
- 论文卖点是 `verifiable semantic rewards for evaluation blind spots`
- `dimension-level recovery` 只能在 C12/C14、downstream utility 和 human-audit gates 通过后作为结论写入

---

## 核心结论

这条线里最难的部分不是把 GRPO 跑起来，而是把 reward 定义、reward 校准和 anti-hacking 机制做对。

风险优先级如下：

1. `reward 校准`：coverage 是否可靠对应 human-gold evaluation dimensions
2. `verifier 噪声`：会不会把模型带偏
3. `reward hacking`：会不会通过多写、空话、重复刷高 reward
4. `数据隔离`：clean split 和最终 C0 audit 是否覆盖所有训练产物
5. `GRPO 工程`：必须绑定训练完成门控和 serving 信息

---

## 总体路线图

| Stage | 目标 | 数据规模 | 产物 | 是否必须 |
|---|---|---:|---|---|
| Stage 0 | Clean split 与训练命令准备 | 1,785 clean proxy-train + seed | `outputs/training_commands/*` | 必须 |
| Stage 1 | SFT-only 训练 | `data/processed/blindspot_sft.jsonl` | `outputs/checkpoints/evaluation_criteria_policy_sft` | 必须 |
| Stage 2 | SFT+GRPO 训练 | `data/processed/proxy_gold_verl.parquet` | `outputs/checkpoints/evaluation_criteria_policy_rl` | 必须 |
| Stage 3 | 训练完成门控 | base/sft_only/sft_rl | `outputs/training_commands/training_done.json` | 必须 |
| Stage 4 | hard-gold + downstream 评测 | RubricBench `test_main` + downstream holdouts | 主表、下游表、消融表、dimension-transition 表 | 必须 |
| Stage 5 | 证据矩阵 + 论文资产同步 | C0-C14 | `outputs/evidence_real/*`, `paper/asset_index.md` | 必须 |

当前 canonical 执行入口是：

```bash
python3 scripts/run_experiment_pipeline.py \
  --config configs/pipeline_real_run.generated.json \
  --from-stage training_commands
```

训练前置条件：

- RewardBench proxy-train 必须已经被过滤为 `data/processed/splits/rewardbench_pref_sft_proxy_train.clean.jsonl`。
- clean proxy-train 当前锁定为 1,785 条，且必须通过 hard-gold、RewardBench、JudgeBench、RewardBench-2 的 pre-SFT 零重叠审计。
- `RubricBench test_main` 只用于 hard-gold 主评测，不进入 SFT、proxy criteria elicitation 或 reward tuning。
- 下游 RewardBench、JudgeBench、RewardBench-2 保持 non-overlap holdout。

---

## Stage 0：SFT 准备

### 目标

把模型训练成一个能稳定生成 evaluation criteria 的 policy；这里的目标是提供
可控的候选维度集合，不是把论文卖点写成“更顺滑的 criteria policy”。

### 数据建议

- RubricBench train seed 和 ResearchRubrics train seed 只作为少量 human-gold seed。
- RewardBench proxy-train 使用过滤后的 `rewardbench_pref_sft_proxy_train.clean.jsonl`。
- IFBench、WritingBench、HealthBench、BEIR/NQ 等只在 multi-teacher generation 和 verifier filtering 后进入 proxy-gold。

约束：

- `RubricBench test_main` 不进入训练。
- downstream RewardBench / JudgeBench / RewardBench-2 holdouts 不进入训练。
- `blindspot_sft.jsonl` 和 `proxy_gold_verl.parquet` 都必须被最终 C0 contamination audit 覆盖。

### 完成标准

- 输出格式稳定
- criteria 数量可控
- 常见维度覆盖正常
- 在 dev 集上具备可审计的格式、有效性和覆盖变化

### 待办清单

- [ ] 确认 Stage 0 训练集清单和 manifest
- [ ] 确认 RubricBench train/dev/test 严格隔离
- [ ] 训练并导出 `sft_model/`
- [ ] 在小样本上对比 `base vs sft`

---

## Stage 1：Offline Reward Sanity Check

### 目标

在真正 RL 前验证 reward 是否有正确排序能力。

### 对照对象

- Base model outputs
- SFT model outputs
- Teacher / human-like outputs
- Random / bad outputs

### 候选排序 sanity check

预注册候选关系：`teacher / human-like`、`sft`、`base`、`bad/random`
应按人工可解释顺序排列；若不满足，应先修 reward 或 verifier，不把该现象写成结果。

### 正式 reward

```text
R = 1.0 * R_cov
  + 0.5 * R_valid
  - 0.5 * R_red
```

### 各项定义

#### 1. `R_cov`

主奖励，衡量 human-gold evaluation dimensions 被覆盖的比例。

- 主实验阈值：`tau = 0.75`
- 分析 sweep：`0.70 / 0.75 / 0.80`

#### 2. `R_valid`

有效性奖励，避免“看起来高级但不可判定”的 criteria。

检查项：

- atomic
- yes/no decidable
- relevant to query
- non-hallucinatory
- not overly vague

#### 3. `R_red`

冗余惩罚，避免重复和语义堆叠。

- 内部 similarity `>= 0.85` 视为重复

#### 4. 格式 fail-closed

格式惩罚，防止 RL 后输出崩坏。

直接触发惩罚的情况：

- 不是 JSON list
- criteria 数量 < 3
- criteria 数量 > 12
- 存在空字符串
- 出现解释性废话

建议：

```text
if parse_failed:
    reward = -1.0
```

### 必过门槛

- [ ] `SFT reward > base reward` 作为 reward sanity 候选关系通过
- [ ] `bad output reward < 0`
- [ ] `valid output reward > invalid output reward`
- [ ] reward 排序与人工判断一致

### 产物

- `outputs/training_commands/run_sft.sh`
- `outputs/training_commands/run_grpo.sh`
- `outputs/training_commands/training_manifest.json`
- `outputs/training_commands/training_done.template.json`

---

## Stage 2：SFT+GRPO Debug

### 目标

先验证训练链路稳定，不追求最终 paper-facing 结果。

### 数据建议

- 从 `data/processed/proxy_gold_verl.parquet` 抽小样本 debug。
- debug split 只能来自已经通过 pre-SFT 与最终 C0 审计范围的数据。
- debug 结果不写入论文主表，只用于训练链路和 reward hacking 排查。

### 推荐配置

```yaml
model:
  base: sft_checkpoint
  max_prompt_length: 2048
  max_response_length: 1024

rl:
  algorithm: grpo
  rollout_n: 4
  train_batch_size: 32
  mini_batch_size: 8
  learning_rate: 1.0e-6
  epochs: 1
  max_steps: 300

reward:
  tau_coverage: 0.75
  tau_redundancy: 0.85
  use_validity: true
  weights: {coverage: 1.0, validity: 0.5, redundancy: 0.5}
```

### 重点观察

- reward 是否出现预期方向的 sanity change
- format 是否崩坏
- redundancy 是否 materially 偏移
- KL 是否失控
- criteria 数量是否膨胀
- 是否出现 reward hacking

### 失败信号

- coverage 出现变化但 validity materially 下降
- criteria 数量爆炸
- redundancy materially 升高
- 输出变成空话或模板堆叠

### 待办清单

- [ ] 跑通 debug 配置
- [ ] 保存训练曲线和样例输出
- [ ] 记录 hacking case
- [ ] 根据现象调整 reward 惩罚项

---

## Stage 3：主 SFT+GRPO

### 目标

训练论文主方法，并在 hard-gold holdout 上检验 RLVR/GRPO 是否带来
dimension-level coverage change。是否能描述为 dimension-level recovery，必须由
C12/C14 的 per-gold dimension-transition audit、redundancy/hallucination 控制和
downstream utility gate 共同决定。

### 数据建议

- 使用 `configs/training_commands.example.json` 绑定的 `data/processed/proxy_gold_verl.parquet`。
- 不再手工拼 3k/5k 训练清单；训练数据由 real-run 流水线、manifest 和 contamination audits 决定。

### 推荐配置

```yaml
model:
  base: sft_checkpoint
  max_prompt_length: 4096
  max_response_length: 1536

rl:
  algorithm: grpo
  rollout_n: 8
  train_batch_size: 64
  mini_batch_size: 8
  learning_rate: 5.0e-7
  epochs: 1
  max_steps: 1000-3000
  kl_coef: 0.02

reward:
  tau_coverage: 0.75
  tau_redundancy: 0.85
  use_validity: true
  weights: {coverage: 1.0, validity: 0.5, redundancy: 0.5}
```

### 成功标准

- 相同 hard-gold 协议下，SFT+GRPO 相对 SFT-only 出现预注册的 BSC coverage change
- C12 显示 recovered human-gold dimensions 多于 lost dimensions，且 query 对齐、BGE、
  threshold 和 verifier valid-flag 协议一致
- downstream utility 只有在 RewardBench / JudgeBench / RewardBench-2
  都通过 API/model scorer、budget report、join audit 和
  `paper_claim_eligible=true` 后，才能作为 judge-utility 支撑
- redundancy 和 hallucination 不发生 material degradation

### 待办清单

- [ ] 使用 `outputs/training_commands/run_grpo.sh` 启动主训练
- [ ] 保存 checkpoint 与训练日志
- [ ] 产出主实验候选模型

---

## RL 训练增强策略

### 1. 主算法

主线使用 `GRPO / RLVR`，不建议把论文重心放到“发明新 RL 算法”。

推荐表述：

`GRPO-style RLVR with a verifier-backed BSC reward for open-ended criteria elicitation`

### 2. Defensive Reward Gates

建议直接截断 reward 的情况：

- `parse_failed -> -1.0`
- `rubric_count < 3 -> -0.5`
- `rubric_count > 12 -> penalty`
- `validity < 0.6 -> cap reward at 0.2`
- `redundancy > 0.4 -> cap reward at 0.3`

---

## 推理测试规划

### 统一评测设置

- 解码：`temperature = 0.2` 或 greedy
- 固定输出预算：`K in {3, 5, 8, 10, 15}`
- 主结果：`K = 8`
- 主阈值：`tau = 0.75`

### 必测指标

- `BSC Cov`
- `Category-Balanced Cov`
- `Validity`
- `Redundancy`
- `Recovered-Dimension Rate`
- `Downstream Utility Support`

### 必做测试

#### 1. Budget-Controlled Curve

比较固定 criteria budget 下的表现，用于判定 coverage change 是否仍成立，
而不是由“生成更多 criteria”解释。

#### 2. Threshold Sweep

对 `tau = 0.70 / 0.75 / 0.80` 做稳定性分析。

#### 3. Bootstrap Confidence Intervals

至少对以下指标做 bootstrap CI；是否写成 paper-facing coverage change 或
dimension-level recovery 取决于 evidence gate：

- BSC
- recovered-dimension rate

### 待办清单

- [ ] 固定预算评测脚本
- [ ] 固定阈值主表
- [ ] threshold sweep
- [ ] bootstrap CI

---

## 主实验矩阵

### 主表

| Method | 说明 |
|---|---|
| Base | 原始 base model |
| SFT-only | proxy-gold SFT checkpoint |
| SFT+GRPO | SFT checkpoint 继续用 BSC reward 做 GRPO/RLVR |

### 建议指标表头

| Method | BSC Cov ↑ | Validity ↑ | Redundancy ↓ | Recovered-Dimension Rate ↑ | Downstream Utility |
|---|---:|---:|---:|---:|---:|

---

## 消融实验规划

### 建议消融表

1. `Full`
2. `No redundancy penalty`
3. `No hallucination/validity term`
4. `No verifier filtering`
5. `SFT-only vs. SFT+GRPO`
6. `Single teacher vs. multi-teacher union`

### 目的

回答以下问题：

- validity gate 是否防止不可判定 criteria
- redundancy penalty 是否抑制堆叠
- verifier filtering 是否让 proxy-gold 更可审计、更少无效项
- RL 阶段是否不是简单增加长度，而是在控制冗余和幻觉时产生可报告 semantic coverage change

---

## RL 成功判定 Gate

### Gate 1：主 BSC coverage change

- SFT+GRPO 相对 SFT-only 必须出现预注册、同协议、bootstrap-CI 支撑的
  hard-gold BSC coverage change
- 同时审计 `mean_n_gen`、`gen_to_gold_ratio`、
  `coverage_per_generated_criterion`、redundancy 和 hallucination，
  防止把长度扩张或重复堆叠误写成 blind-spot coverage
- 必须通过 C2/C14 evidence gates 后才能写成主结论

### Gate 2：Dimension-Transition Audit

- `RecoveredDimensionRate(RL) > RecoveredDimensionRate(SFT) + 5 points` 只是候选阈值
- 必须同时检查 lost dimensions、query alignment、fixed BGE embeddings、
  verifier valid-flag filtering 和 paired bootstrap CI
- 未通过 C12/C14 时，只能写 coverage change，不能写 dimension-level recovery

### Gate 3：不靠堆数量

- 在相同 `K` 下，检查 RL 相对 SFT 的 coverage change 是否仍成立

### Gate 4：不牺牲 validity / hallucination

- `Validity_RL >= Validity_SFT - 1 point`
- `Hallucination_RL` 不超过 evidence gate 配置

### Gate 5：downstream utility support

- `Downstream_RL >= Downstream_SFT` 是候选判据，不是预设结论
- RewardBench / JudgeBench / RewardBench-2 都必须使用 API/model scorer、scorer provider 配置和 budget report

---

## 论文写法建议

### Method Section

#### 3.1 Blind-Spot Diagnosis

定义：

- human-gold evaluation dimensions
- generated criteria
- blind spots

#### 3.2 Blind-Spot Coverage Reward

介绍：

- `R_cov`
- `R_valid`
- `R_red`

#### 3.3 RLVR with GRPO

说明：

- 从 SFT checkpoint 初始化
- 每个 query 采样多个 criteria sets
- 用 verifier-backed BSC reward 做候选排序和 GRPO 更新
- 用 GRPO 更新模型
- 不需要人工在线标注
- 训练完成后必须写入 `outputs/training_commands/training_done.json`
- `served_generators` 必须包含 `base`, `sft_only`, `sft_rl`
- `sft_data` 必须等于 `data/processed/blindspot_sft.jsonl`
- `rl_data` 必须等于 `data/processed/proxy_gold_verl.parquet`
- `"rl_data_report": "outputs/sft_data/proxy_gold_verl_report.json"` 必须记录 GRPO 数据转换 provenance

### 最终 claim

```text
RL is not used as a generic post-training recipe.
It formulates an evidence-gated optimization test for verifiable
blind-spot coverage in open-ended evaluation-criteria elicitation, and
empirical dimension-recovery claims are permitted only after hard-gold,
downstream, ablation, dimension-transition, and semantic-space gates pass.
```

中文表达：

RL 不是泛化的 post-training 装饰，也不是“更会写 criteria”的工程卖点；它把
“human-gold evaluation dimensions 是否被覆盖”转成可验证 reward，并测试在控制冗余
和幻觉时，coverage change 是否具备写成 evaluation dimension-recovery statement
的证据。只有当 C12/C14、downstream utility、human audit 和 semantic-space gates
同时支持时，才把结果写成 dimension-level recovery。

---

## 当前建议的第一批落地任务

按顺序做，不要并行开太多战线。

### 第一优先级

- [ ] 运行 real-run data/filter/pre-SFT audit stages
- [ ] 确认 clean RewardBench proxy-train 为 1,785 条
- [ ] 确认 hard-gold、RewardBench、JudgeBench、RewardBench-2 pre-SFT audits 均为 `overlap_query_count == 0`
- [ ] 生成 `outputs/training_commands/run_sft.sh` 和 `run_grpo.sh`

### 第二优先级

- [ ] 运行 `outputs/training_commands/run_sft.sh`
- [ ] 运行 `outputs/training_commands/run_grpo.sh`
- [ ] 写入 `outputs/training_commands/training_done.json`
- [ ] 通过 `training_completion_gate`

### 第三优先级

- [ ] Serve `base`, `sft_only`, `sft_rl`
- [ ] 运行 RubricBench hard-gold BSC 主评测
- [ ] 运行 RewardBench、JudgeBench、RewardBench-2 downstream utility
- [ ] 确保 downstream scorer 为 API/model scorer，并记录 provider 与 budget report

### 第四优先级

- [ ] 运行 redundancy、hallucination/validity、verifier、SFT-only vs SFT+GRPO、teacher-union 消融
- [ ] 生成 dimension-transition audit 表
- [ ] 生成 semantic-space PDF/SVG/CSV/JSON
- [ ] 运行 evidence matrix、submission readiness、result card、paper asset index check

---

## 一句话执行原则

先验证 reward 与 human-gold dimensions 的对齐，再验证 RL 训练稳定性，最后由
C12/C14、downstream utility 和 human audit 判定是否具备写成 dimension-level recovery
statement 的证据。
