# 第一阶段云端环境准备

执行状态：第一阶段现已实际运行并完成同机两次分数一致性检查。下文保留最初准备建议；实际运行命令见 `stage1/README.md`，结果与协议差异见 `reports/stage1/`，准确环境与验收证据见 `results/stage1/`。

此文件是环境准备建议，不替代 `STAGE1_TASK.md` 的实验协议。
本次仅上传资料，未创建或验证实际云端环境。

## 环境

- 建议以 Python 3.11 和独立 Python 环境起步。
- `python -m pip install -r requirements-stage1.txt`：安装任务依赖及原始归一化代码所需依赖。
- `python -m pip check`：检查依赖一致性。
- 确认终端 Python 与 notebook kernel 使用同一环境。
- 若安装版本不兼容，解决后记录具体版本；不要将此清单当作已验证锁定文件。

在云端环境变量设置中配置下列值，保证在 Python 进程启动前生效：

```text
PYTHONHASHSEED=0
PYTHONUTF8=1
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MPLBACKEND=Agg
```

从仓库根目录运行；代码用 `pathlib` 解析路径。`DATA_DIR` 的仓库相对路径为：

```text
data/DSAI4205_unified_cleaning_seed42_v3_stemming
```

## 依赖与模型准备

1. 下载 NLTK 的 `wordnet`、`averaged_perceptron_tagger_eng`、`words`。
2. 预下载 `gensim.downloader` 的 `glove-twitter-100`。
3. 预下载 `vinai/bertweet-base` 的 tokenizer 与模型权重。
4. 使用 `AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=False)` 验证分词器加载。
5. 用人工示例字符串验证 float32 CPU 推理、Word2Vec/FastText 极小语料训练和 notebook 绘图。
6. 检查原始 `normalize_input` 函数返回签名；只允许它转换 GloVe 词表。
7. 配置稳定的 NLTK、Hugging Face 和 gensim 缓存路径，并验证缓存加载。
8. 保存实际 Python 版本、`pip freeze`、机器信息、可用设备和验证结果。

CPU 是可用的验证起点；是否有 GPU 以实际环境检测为准，不承诺固定耗时。
下载公开权重不需要 OpenAI API key。不要把任何凭证写入仓库。

## 网络

允许包管理器、Hugging Face、GloVe 和 NLTK 下载来源所需的访问。
具体域名以实际下载及重定向日志为准；下载失败要给出失败地址与原因。
准备阶段只下载资源和做人工语料检查，不开始完整词表转换、特征抽取或交叉验证。

## 正式任务

环境验证后再按 `STAGE1_TASK.md` 实现并执行：

```bash
python -m stage1.run check
python -m stage1.run extract
python -m stage1.run probe
python -m stage1.run figures
python -m stage1.run build-notebook
```

这些命令需要先完成实验实现，当前资料提交不包含对应模块。
最终按任务要求完成同机两次结果一致性检查，以及 `RECOMPUTE=False` 的 notebook 完整执行。

## 参考材料

- `references/Project description.pdf`：课程项目要求。

用户明确要求不上传两份中期报告，因此仓库不包含它们。基线实现依据原始第一阶段任务说明；不足之处记录在 OPEN_QUESTIONS，不猜测历史结果或参数。

参考材料不参与特征构造或实验数据补全；与任务要求有冲突时记录问题，不静默改变协议。
