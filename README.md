# DSAI4205 Group Project — Stage 1 preparation

本仓库提供第一阶段 embedding 对比所需的输入、任务说明与参考材料。
当前状态：资料已整理；`stage1/` 实验实现、模型下载、完整实验及云端依赖验证尚未执行。

## 从这里开始

1. 阅读 `AGENTS.md`、`STAGE1_TASK.md` 和 `docs/CLOUD_SETUP.md`。
2. 云端环境使用 Python 3.11，安装 `requirements-stage1.txt`，验证后保存实际依赖锁定文件。
3. 从仓库根目录将 `DATA_DIR` 设为 `data/DSAI4205_unified_cleaning_seed42_v3_stemming`。
4. 按任务说明实现 `stage1/`，先检查输入，再抽取特征并运行探针。
5. 未覆盖的决定及已知说明差异见 `reports/stage1/OPEN_QUESTIONS.md`。

## 文件导航

| 路径 | 内容 |
|---|---|
| `STAGE1_TASK.md` | 用户提供的原始第一阶段说明，逐字节保留 |
| `AGENTS.md` | 用户的工作规则及本仓库的运行边界 |
| `data/DSAI4205_unified_cleaning_seed42_v3_stemming/` | 从原始 ZIP 提取的第一阶段输入子集，文件字节不变 |
| `data/archives/` | 原始完整 v3 ZIP，仅作归档，不作为第一阶段递归扫描的输入 |
| `references/` | 原始课程项目说明 PDF；按用户要求不上传两份中期报告 |
| `requirements-stage1.txt` | 环境安装输入；尚未解析或锁定版本 |
| `docs/CLOUD_SETUP.md` | 云端环境准备和验证要求 |
| `UPLOAD_MANIFEST.json` | 本次上传资料的大小和 SHA-256，不包含自身 |

## 数据边界

解压目录是原始数据包的输入子集，并非完整解压包。原始 `SHA256SUMS.json` 保留未修改，因此其中包含本目录没有提取的文件；不能将这些文件误报为损坏，也不能声称整包文件都已验证。

第一阶段只把 `train.csv` 的 `Phrase` 作为模型输入，并使用 `fold_assignments.csv`。`code/unified_data.py` 只允许用于 GloVe 词表对齐。禁止读取测试集、审计文件或以归档包中的其他内容扩充输入。不要修改任何数据副本或重新清洗 `Phrase`。

完整 ZIP 包含测试集及审计材料；上传归档不代表允许在实验中读取它们。
预训练 GloVe/BERTweet 权重应由云端下载，不提交到 Git。
