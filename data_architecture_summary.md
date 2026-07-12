# BlindSpot-RL 数据架构与分片梳理 (Data Architecture & Splits)

根据最新的数据处理和物理切分状态，当前项目的数据已经被严格划分为四层结构（遵循 `group_disjoint` Query级隔离，并在 `dataset_manifest.json` 中配置了审计锁）。

## 1. Hard-gold 主评测与种子层 (Human-Gold)
**核心作用**：提供最干净的人类标注 Rubric，用于评估最终的 BSC（盲区覆盖度）主表格，以及提供高质量的 SFT 冷启动种子。
*不允许任何 Proxy 数据污染此层。*

*   **RubricBench (总计 1,147 条)**
    *   `test_main`：**497 条** (完全 Holdout，**唯一** `allowed_in_main_bsc_eval: true` 的核心主评测集)
    *   `dev`：**150 条** (用于阈值、Reward 权重、Prompt 调参)
    *   `train_seed`：**500 条** (用于 SFT Seed 和 Reward Calibration)
*   **ResearchRubrics (总计 101 条)**
    *   `train_seed`：**60 条** (用于增强复杂 Research Task 评估能力)
    *   `dev_test`：**41 条** (用于 OOD / DeepResearch 分析)
    *   *(注：该数据集整体 `allowed_in_main_bsc_eval: false`，避免审稿人质疑主评测集污染)*

---

## 2. Proxy-gold 训练层 (Proxy-Teacher)
**核心作用**：利用多教师模型大规模生成的“代理评估标准 (Proxy Rubrics)”，用于 SFT 和 RLVR 阶段的规模化训练。

*   **RewardBench (总计 2,985 条)**
    *   `sft_proxy_train`：**1,791 条** (约 60%，主力 Pairwise 判断训练语料)
    *   `dev`：**597 条** (约 20%)
*   **IFBench (Query Pool: 300 条)**
    *   用于指令遵循 (Instruction-following) 维度的隐藏约束 (Hidden constraints) 泛化训练。
*   **WritingBench (Query Pool: 1,000 条)**
    *   用于写作任务中的主观质量、风格、结构判定的泛化训练。

---

## 3. 领域泛化层 (Domain Generalization Proxy)
**核心作用**：验证或增强 Rubric-Generator 在特殊领域（医疗安全、搜索等）的泛化能力。

*   **HealthBench (Query Pool: 5,000 条)**
    *   医疗问答领域。用于训练模型在医疗等高风险场景下的安全性、事实性和边界意识 (Risk disclosure)。
*   **BeIR/NQ (Query Pool: 3,452 条)**
    *   自然问题与搜索意图。用于对接 SERP / Search 评估场景，生成与相关性、证据支持度相关的 Rubric。

---

## 4. Downstream Holdout 层 (Downstream Judge Evaluation)
**核心作用**：完全不参与任何形式的训练，纯粹用于在下游任务中证明模型的判别 (Judge) 能力。

*   **RewardBench-2 (1,865 条)**：Full Holdout，不进训练。
*   **JudgeBench (620 条)**：Full Holdout (Hard judge cases)，不进训练。
*   **RewardBench downstream_holdout (597 条)**：从 RewardBench 中切分出的 20% 纯净测试集。

---

## 汇总与下一步提示

*   **当前可直接用于训练的 Seed**：RubricBench (500) + ResearchRubrics (60) = **560 条** (Human-gold)。
*   **当前主评测基准 (Main BSC Eval)**：RubricBench (497)。
*   **待生成的 Proxy-Gold 候选池规模**：RewardBench (1791) + IFBench (300) + WritingBench (1000) + HealthBench (5000) + BeIR (3452) ≈ **11,543 条**。
*   **状态**：Query 候选池已经就绪并归一化至 `data/processed/` 目录下。等待 API Budget 闸口解锁后，即可执行多教师模型 Rubric 生成与 Meta-Verifier 过滤流水线。