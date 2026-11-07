# 第一阶段运行与结果

目标、固定参数和验收要求见原始 `../STAGE1_TASK.md`。本实现只使用原样训练 Phrase、既定标签及分组；不读取测试、审计或归档 ZIP，不执行第二阶段。

## 安装

本次实际运行环境为 Python 3.12，完整版本和机器信息保存在 `results/stage1/env_*.txt`。推荐建立独立环境；Linux CPU 的安装方式如下：

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-stage1.txt pytest
NLTK_ALLOW_PROXIED_URLOPEN=1 .venv/bin/python -c "import nltk; [nltk.download(p, download_dir='.nltk_data', raise_on_error=True) for p in ['wordnet', 'averaged_perceptron_tagger_eng', 'words']]"
.venv/bin/python -m pip check
```

NLTK 的代理开关仅适用于本次受管可信代理；无需代理的环境可省略。预训练资源来自任务指定的 GloVe 与 BERTweet 检查点。

## 执行

从仓库根目录运行。包装脚本在 Python 启动前固定哈希种子、线程数、UTF-8 及缓存目录：

```bash
bash stage1/run_stage1.sh check
bash stage1/run_stage1.sh extract
bash stage1/run_stage1.sh probe --verify-reproducibility
bash stage1/run_stage1.sh figures
bash stage1/run_stage1.sh build-notebook
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

`probe --verify-reproducibility` 连续运行两遍完整外层词模型训练和嵌套探针，要求 `fold_scores.csv` 字节相同。冻结的预训练特征复用已验证缓存。普通 `probe` 复用折内向量；`probe --recompute` 强制重训词模型。

缓存同时保存输入、配置和 embedding 实现的指纹，失配时报错，不静默使用陈旧特征。若主动变更实现，应将旧的生成缓存移到其他位置后重新运行，不能修改原始数据。

`build-notebook` 创建自包含 notebook，使用当前 Python 的私有 kernelspec 完整执行 `RECOMPUTE=False`，并保存执行输出。缓存模式要求所有特征、评分与 t-SNE 坐标已存在，不训练任何模型；`RECOMPUTE=True` 才重新运行实验。

## 交付物与证据

- `reports/stage1/embedding_comparison.md`：中文结果、预先规则判定与第二阶段候选建议。
- `reports/stage1/method_en.md`：英文方法段落。
- `reports/stage1/OPEN_QUESTIONS.md`：协议冲突与未指定决定。
- `results/stage1/`：分组划分、逐折分数、全网格选参、R0 OOF、配对差、覆盖率、环境及验收记录。
- `figures/stage1/`：F1–F5 的 300 dpi PNG 与对应 CSV。
- `features/stage1/`：冻结特征、每个外层折词向量、GloVe 词表映射、token 特征与完整性元数据。
- `notebooks/stage1_embedding_comparison.ipynb`：代码内联、缓存模式已执行的 notebook。

完整原始清单涉及禁读文件，本实现只验证已提取且允许读取的 5 个文件；其余条目明确未验证。不能将 `check` 的有限范围通过解释成整个原始数据包哈希全部通过。
