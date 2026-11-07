# Codex 任务说明：第一阶段：4 种 embedding 的对比（统一线性探针）

> DSAI4205 小组项目，进阶建模的**第一阶段**。
> 目标：分类器固定不变，只更换 embedding，比较 4 种 embedding 在我们数据上的表现。分类器选进阶模型（6 个 analytics model）是**第二阶段**的事，等本阶段结果出来再写任务文档。本阶段**不碰测试集**。
> 请严格按本文档执行。本文档没有覆盖到的决定，记录到 `reports/stage1/OPEN_QUESTIONS.md`，选最保守的做法继续，不要擅自扩大范围。

---

## 0. 输入：固定使用 v3 数据集

`DATA_DIR` 指向解压后的 `DSAI4205_unified_cleaning_seed42_v3_stemming/` 文件夹。**这个文件夹只读，里面的任何文件都不能修改。**

| 项目 | 值 |
|---|---|
| 训练数据 | `DATA_DIR/train.csv`，563 行，两列：`Phrase`、`Sentiment` |
| 模型输入 | `Phrase` 列，**原样使用**，一个字都不能改 |
| 标签 | `Sentiment`，0 = Negative，1 = Positive，2 = Irrelevant，3 = Neutral。不要用 `LabelEncoder` |
| 行号 | `DATA_DIR/fold_assignments.csv` 的 `row_index`（0–562，和 `train.csv` 的行顺序一致），以及 `sample_id` |
| 主划分 | `fold_assignments.csv` 的 `validation_fold`，取值 **1–5** |
| 分组 | `fold_assignments.csv` 的 `group_id`，共 13 个多行组 |
| 测试集 | 本阶段**不读取** `test.csv` |

v3 的 `Phrase` 已经做过这些处理：全部小写；去掉了标点；先做 WordNet 词形还原，再做 Snowball 词干化（例如 `realli`、`pleas`、`was→be`）；URL 变成 `url` 加域名词；`@` 已去掉，账号名保留成普通词；emoji 变成名称词；无法恢复的字符保留为 `�`。词之间用空格分隔，**分词就是按空格切开**。

数据包里的审计文件（`*_audit_views.csv`、`all_rows_audit.csv`、`text_transformation_audit.csv` 等）**不能读取**，既不能当输入，也不能当特征。唯一的例外是 `code/unified_data.py` 里的 `normalize_input` 函数，只在 3.3 中用来转换 GloVe 的**词表**。

---

## 1. 背景与原则

### 1.1 为什么分两个阶段

如果 embedding 和分类器一起换，分数变了也说不清是谁的作用。本阶段把分类器固定成同一个简单的线性模型，分数的差异就只来自 embedding。

### 1.2 课程规则

- TF-IDF 只能用在基线里，而且不算 embedding。本阶段只有"基线参照"（第 5 节）可以用 TF-IDF。
- 本阶段用 Logistic Regression 作为**测量工具（探针）**，它**不是**进阶模型，也不算在 6 个 analytics model 里。报告里要写清楚这一点。
- BERTweet 必须冻结，只做特征提取，权重不更新。

### 1.3 原则

1. **探针对所有 embedding 一视同仁**：同样的预处理、同样的分类器、同样的超参数网格、同样的选参流程（第 4 节）。
2. **防泄漏**：标准化、PCA、从头训练的词向量、超参数选择，全部只在当前折的训练部分上进行。
3. **噪声意识**：折间标准差大约 2.6 个点。所以除了 v3 的 5 折，还要做 4 次额外的重复划分（4.1），排序要看 25 折的结果。
4. **所有随机种子固定**，结果可以复现。

---

## 2. 目录结构与交付物

```
stage1/
├── config.yaml
├── data_io.py            # 读取 v3、校验、生成重复划分
├── embeddings.py         # 4 种 embedding → 句向量
├── probe.py              # 线性探针 + 嵌套交叉验证
├── baseline_ref.py       # TF-IDF + LR 参照，是唯一允许出现 TF-IDF 的文件
├── figures.py
├── build_notebook.py
└── run.py                # 命令行：check / extract / probe / figures / build-notebook
features/stage1/          # 缓存
results/stage1/
figures/stage1/
reports/stage1/
├── embedding_comparison.md   # 中文结果报告
├── method_en.md              # 英文方法描述，可以直接贴进最终报告
└── OPEN_QUESTIONS.md
notebooks/stage1_embedding_comparison.ipynb
```

---

## 3. 四种 embedding（`stage1/embeddings.py`）

每种 embedding 最后都输出**一条推文一个句向量**，做法统一为：**对所有词（或子词）的向量取平均**。

### 3.1 Word2Vec（从头训练）

- `gensim.models.Word2Vec(vector_size=100, window=5, min_count=1, sg=1, negative=5, epochs=50, workers=1, seed=42)`。
- **只用当前外层折训练部分的 `Phrase` 训练**，每个外层折单独训练一个模型。验证部分和测试集的文本一律不参与训练。
- 训练部分没有出现过的词，在验证部分就是未登录词（OOV），直接跳过。如果一行里所有词都是 OOV，这一行的句向量设为全零，并记录行数。
- 内层交叉验证直接使用外层折训练出来的词向量，不再按内层折重新训练。这样内层选参时，内层验证部分的文本已经被无监督地"看过"了，会让选参稍微乐观一点，报告里要注明。

### 3.2 FastText（从头训练）

- `gensim.models.FastText(vector_size=100, window=5, min_count=1, sg=1, negative=5, epochs=50, min_n=3, max_n=6, workers=1, seed=42)`。
- 训练方式和 OOV 的处理与 3.1 完全相同。唯一的区别是：FastText 能用字母片段给没见过的词拼出向量，所以不会出现 OOV。
- 3.1 和 3.2 只差"有没有子词"这一点，其他参数必须完全一致。

### 3.3 GloVe-Twitter（预训练）

- 通过 `gensim.downloader.load("glove-twitter-100")` 加载：100 维，在 20 亿条推文上训练，全部小写。
- **两种查词方式都要做**：
  - **G-stem（主方案）**：把 GloVe 词表里的每个词，用 `DATA_DIR/code/unified_data.py` 里的 `normalize_input(word, morphology="lemma_stem")` 转换成 v3 的形式，只保留转换后是单个词的结果。转换结果相同的 GloVe 词，向量取平均，得到一张"v3 形式 → 向量"的新表，再用 `Phrase` 的词去查这张表。例如 `really` 转成 `realli`，`Phrase` 里的 `realli` 就能查到 `really` 的向量。**这一步改的是 GloVe 的词表，不是数据。**
  - **G-direct（对照）**：直接用 `Phrase` 的词查原始的 GloVe 词表。
- 查不到的词跳过；一行全部查不到时，句向量设为全零，并记录行数。
- 词表转换比较慢，结果缓存到 `features/stage1/glove_stem_map.pkl`。如果要先筛一遍再转换以节省时间，必须保证结果和不筛完全一样：随机抽 1 万个词，分别用"筛选"和"不筛选"两种方式处理，结果要一致。
- G-stem 有多少个 v3 形式是由多个 GloVe 词合并而来的，在报告里列出合并最多的前 20 个，例如 `be` 合并了 `is`、`was`、`are`……

### 3.4 BERTweet（预训练，上下文，冻结）

- 模型：`vinai/bertweet-base`，只用这一个检查点。
- 分词器：`AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=False)`。**必须关掉 normalization**，否则分词器会改写文本。断言：送进分词器的字符串，和 `Phrase` 逐字节相同。
- 参数：`max_length=128`，开启截断；`output_hidden_states=True`；float32；`model.eval()` 加 `torch.no_grad()`。设备按 `cuda` → `mps` → `cpu` 的顺序自动选择。
- 句向量：取**最后 4 层（第 9–12 层）的平均**，再对所有真实 token 取平均，**不包括** `<s>`、`</s>` 和 padding。输出 768 维。
- **冻结检查**：抽取前后各计算一次 `state_dict` 的校验和，两次必须相同；同时断言所有参数的 `requires_grad` 都是 False。
- 抽取不需要拟合，可以对 563 行一次性算好并缓存。顺便把 token 级特征（每个 token 取最后 4 层的平均，float16）也存下来，第二阶段的序列模型会用到。本阶段不使用这部分。
- **维度对照**：BERTweet 是 768 维，其他三种是 100 维。为了排除"维度高所以分数高"的可能，额外跑一个 **BERTweet-PCA100**：在每个训练部分上拟合 PCA，降到 100 维后再接探针。

### 3.5 适配度统计（不用标签，只用训练集文本）

写进 `results/stage1/coverage.csv`：

| embedding | 统计项 |
|---|---|
| Word2Vec | 每个外层折中，验证部分的词有多少比例没在训练部分出现过（按出现次数和按不同词分别统计）；全零向量的行数 |
| FastText | 同上，作为参照（FastText 实际上不会出现 OOV） |
| GloVe G-direct / G-stem | OOV 率（按出现次数和按不同词分别统计）；全零向量的行数；被 G-stem 救回来的词里，出现次数最多的前 20 个 |
| BERTweet | 平均每个词被切成几个子词；被切成 2 个及以上子词的词所占比例；`<unk>` 的 token 数和行数；被截断的行数 |

---

## 4. 线性探针与评估协议（`stage1/probe.py`）

### 4.1 划分

- **R0**：v3 自带的 5 折，用来和基线直接比较。
- **R1–R4**：`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=r)`，r = 1, 2, 3, 4，分组用 `group_id`。
- 一共 5 次重复 × 5 折 = **25 折**，保存到 `results/stage1/folds_repeats.csv`，列为 `row_index, repeat, fold`。生成后断言：同一组的行在每一次重复里都落在同一折。

### 4.2 探针

```python
pipeline = [StandardScaler(),                     # 只在训练部分上拟合
            LogisticRegression(C=C, class_weight="balanced",
                               solver="lbfgs", max_iter=5000)]
C_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
```

- 所有 embedding 都用这同一个探针，同一个网格。
- **选 C**：在每个外层折的训练部分上，做 `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)` 的内层交叉验证，选内层平均 macro-F1 最高的 C。如果两个 C 的分数相差小于 0.001，选较小的 C（正则更强）。
- 用选出的 C 在整个外层训练部分上重新训练，预测外层验证部分。
- 如果最优的 C 落在网格边缘（0.001 或 10）的折数超过一半，**不要改网格**，只在 OPEN_QUESTIONS 里记录。

### 4.3 要跑的配置（一共 7 种）

| 名称 | 说明 |
|---|---|
| `w2v` | 3.1 |
| `ft` | 3.2 |
| `glove_stem` | 3.3 主方案 |
| `glove_direct` | 3.3 对照 |
| `bertweet` | 3.4 |
| `bertweet_pca100` | 3.4 维度对照 |
| `baseline_ref` | 第 5 节 |

### 4.4 指标与比较

- 每一折记录：macro-F1（主指标）、accuracy、宏平均 precision、宏平均 recall、4 个类别各自的 F1、选中的 C。
- 汇总时：
  - 每种配置在 R0 上的合并 OOF macro-F1，和中期报告的基线 43.66 对应；
  - 25 折的均值 ± 标准差。
- **两两配对比较**：任意两种配置 A、B，在同一折上算 Δ = A − B，报告 25 折的 Δ 均值、Δ 标准差、A 赢的折数。p 值不作为判断依据，最多作参考。
- **三个关键对比**，对应 4 种 embedding 的设计逻辑：

| 对比 | 要回答的问题 |
|---|---|
| `ft` − `w2v` | 子词信息有没有用 |
| `glove_stem` − `w2v` | 大规模预训练有没有用（这一对的训练算法也不同，报告里注明） |
| `bertweet` − `glove_stem` | 上下文有没有用 |

另外两个辅助对比：`glove_stem` − `glove_direct`（词表对齐的作用）、`bertweet` − `bertweet_pca100`（维度的作用）。

- 判断规则（事先定好，不能事后修改）：Δ 均值的绝对值 < 1 个点，**或者**赢的折数不到 18/25，就写"无法区分"。

### 4.5 结果文件

| 文件 | 列 |
|---|---|
| `fold_scores.csv` | `config, repeat, fold, C, macro_f1, acc, macro_p, macro_r, f1_neg, f1_pos, f1_irr, f1_neu` |
| `tuning.csv` | `config, repeat, fold, C, inner_f1_mean, inner_f1_sd`，**网格里每个 C 都要记录** |
| `oof_<config>.csv` | 只记录 R0：`row_index, sample_id, fold, y_true, y_pred, p0, p1, p2, p3` |
| `pairwise_delta.csv` | `config_a, config_b, mean_delta, sd_delta, wins_a, n_folds` |

---

## 5. 基线参照（`stage1/baseline_ref.py`）

- 特征沿用中期报告选定的配置：词 unigram 加 char_wb 3–5 的 TF-IDF（`sublinear_tf=True`，`min_df=2`，两部分分别 L2 归一化后拼接），只在训练部分上拟合。
- 分类器和第 4 节完全一样：同一个 LR、同一个 C 网格、同一个内层选参流程。唯一的区别是 TF-IDF 是稀疏矩阵，所以**不做** `StandardScaler`。
- 这样，基线参照和 embedding 探针之间只差"表示方式"，可以直接回答"embedding 比 TF-IDF 好还是差"。
- 报告里注明：这是按相同流程复现的参照，数值和中期报告的 43.66 可能略有差异。

---

## 6. 图（英文标签，300 dpi PNG，每张图另存对应的数据 CSV）

| 编号 | 图 |
|---|---|
| F1 | 7 种配置的 macro-F1 柱状图：25 折均值 ± 标准差，同时用点标出 R0 的值；基线参照画成虚线 |
| F2 | 各类别 F1 热力图：行是 7 种配置，列是 4 个类别 |
| F3 | 适配度对比：OOV 率、子词切碎比例等（3.5） |
| F4 | t-SNE 2×2 小图：`w2v`、`ft`、`glove_stem`、`bertweet` 四种句向量，按类别着色（`perplexity=30`，`random_state=42`）。只用于可视化：`w2v` 和 `ft` 这里用全部 563 行训练出来的模型 |
| F5 | 两两 Δ 矩阵热力图：格子里的数字是 Δ 均值，下面附赢的折数 |

---

## 7. 报告

### 7.1 `reports/stage1/embedding_comparison.md`（中文）

- 总表：7 种配置的 R0 合并 OOF macro-F1、25 折均值 ± 标准差、accuracy、各类别 F1，并标出名次。
- 三个关键对比和两个辅助对比的结论，按 4.4 的规则写"更好 / 更差 / 无法区分"。
- 单独讨论 Irrelevant 这一类。
- 适配度统计和分数之间的关系。只陈述观察到的现象，解释原因时用"可能"。
- 每种配置选中的 C 是如何分布的（这是调参证据）。
- 最后给出**建议**：哪 2 种 embedding 适合带进第二阶段，附上理由。最终由小组决定。

### 7.2 `reports/stage1/method_en.md`（英文）

用一段话描述本阶段的方法：四种 embedding、统一的平均池化、统一的线性探针和嵌套分组交叉验证、维度对照和词表对齐，以及"探针不属于 N 个 analytics model"这一点。

---

## 8. 自包含 notebook

生成 `notebooks/stage1_embedding_comparison.ipynb`：按顺序把 `stage1/*.py` 的代码内联进 notebook，**不 import `stage1/` 里的任何模块**，每一节前面加英文 markdown 说明。开头放一个开关 `RECOMPUTE = False`：为 False 时读取缓存，只生成表格和图。在这个设置下完整执行一遍，确认没有报错。

---

## 9. 运行方式与环境

```bash
pip install torch transformers gensim scikit-learn nltk pandas numpy matplotlib pyyaml nbformat nbconvert
python -c "import nltk; [nltk.download(p) for p in ['wordnet','averaged_perceptron_tagger_eng','words']]"
python -m stage1.run check
python -m stage1.run extract       # BERTweet 特征 + GloVe 词表转换，只需要跑一次
python -m stage1.run probe         # 7 种配置 × 25 折
python -m stage1.run figures
python -m stage1.run build-notebook
```

- **可复现**：运行前设置 `PYTHONHASHSEED=0`（gensim 用 Python 的哈希函数初始化词向量）；gensim 设 `workers=1`。同一台机器上重跑两次，`fold_scores.csv` 必须完全一致。
- `transformers` 用 4.x。如果更新的版本加载 BERTweet 分词器报错，就退回 4.x，并在 env 文件里记录。
- **Windows**：所有读写文件的地方都显式写 `encoding="utf-8"`；运行前设置 `PYTHONUTF8=1`；入口加 `if __name__ == "__main__":` 保护；路径一律用 `pathlib`。
- **Mac**：先设置 `export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false`；在 MPS 上不要开 fp16。
- 每次运行都把 `pip freeze` 的输出和机器信息写进 `results/stage1/env_<机器名>.txt`。
- 预计耗时：BERTweet 抽取在 GPU 上大约 1 分钟，在 CPU 上几分钟；GloVe 词表转换几分钟，有缓存；探针部分全是线性模型，总共应该在 30 分钟以内。

---

## 10. 验收清单

- [ ] `check` 通过：`SHA256SUMS.json` 里的哈希全部一致；563 行；`fold_assignments.csv` 和 `train.csv` 对齐；同组的行不跨折（R0–R4 都要检查）。
- [ ] 每个阶段读到的 `Phrase` 的 sha256 都和 `check` 时记录的相同；分词器的输入和 `Phrase` 逐字节相同。
- [ ] BERTweet 抽取前后的校验和相同。
- [ ] Word2Vec 和 FastText 的训练文本里，没有任何一行来自当前外层折的验证部分（加断言）。
- [ ] 在 `stage1/` 里搜索 `Tfidf`，只能出现在 `baseline_ref.py` 里。
- [ ] 代码里没有读取 `test.csv`，也没有读取任何审计文件。
- [ ] `tuning.csv` 里包含所有配置、所有折、所有 C。
- [ ] 同一台机器上重跑，`fold_scores.csv` 完全一致。

---

## 11. 不要做的事

- 不要改动 `Phrase`（不还原词干、不恢复大小写或标点），不要读取审计列。
- 不要读取或评估测试集。
- 不要给不同的 embedding 用不同的分类器、网格或预处理。
- 不要因为某个 embedding 分数低就去调它的参数，3.1–3.4 的参数是固定的。
- 不要更新 BERTweet 的权重。
- 不要在 `baseline_ref.py` 以外的地方使用 TF-IDF。
- 不要引入外部数据（GloVe 和 BERTweet 的预训练权重除外）。