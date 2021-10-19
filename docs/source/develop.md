# Develop

### 代码风格

我们采用 [PEP8](https://www.python.org/dev/peps/pep-0008/) 作为首选代码风格。

我们使用以下工具进行 美化纠错 和格式化：

- [flake8](http://flake8.pycqa.org/en/latest/)：美化纠错(linter)
- [yapf](https://github.com/google/yapf)：格式化程序
- [isort](https://github.com/timothycrosley/isort)：对 import 进行排序整合

我们在每次提交时都会自动使用 [pre-commit hook](https://pre-commit.com/) , 来检查和格式化 `flake8`、`yapf`、`isort`、`trailing whitespaces`、修复 `end-of-files`问题，对 `requirments.txt` 进行排序。

yapf 和 isort 的样式配置可以在[setup.cfg](setup.cfg) 中找到。

pre-commit hook 的配置存储在 [.pre-commit-config](.pre-commit-config.yaml) 中。

在克隆git仓库后，您需要安装初始化pre-commit hook:

```bash
pip install -U pre-commit
```

定位到存储库文件夹

```bash
pre-commit install
```

在此之后，每次提交检查代码 linters 和格式化程序将被强制执行。

如果您只想格式化和整理代码，则可以运行

```bash
pre-commit run -a
```

### 测试

#### 单元测试

TEST_DEVICES=0,1 sh scripts/ci_test.sh

```bash
TEST_DEVICES=0,1 sh scripts/ci_test.sh
```

#### Odps 测试

```bash
TEMPDIR=/tmp python -m easy_rec.python.test.odps_run --oss_config ~/.ossutilconfig [--odps_config {ODPS_CONFIG} --algo_project {ALOG_PROJ}  --arn acs:ram::xxx:role/yyy TestPipelineOnOdps.*]
```
#### 测试数据

下载测试数据
```bash
wget https://easyrec.oss-cn-beijing.aliyuncs.com/data/easyrec_data_20210818.tar.gz
tar -xvzf easyrec_data_20210818.tar.gz
```

如果您要添加新数据，请在“git commit”之前执行以下操作,以将其提交到 git-lfs：

```bash
python git-lfs/git_lfs.py add data/test/new_data
python git-lfs/git_lfs.py push
```

### 文档

我们支持 [MarkDown](https://guides.github.com/features/mastering-markdown/) 格式和 [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) 格式的文档。

如果文档包含公式或表格，我们建议您使用 reStructuredText 格式或使用
[md-to-rst](https://cloudconvert.com/md-to-rst) 将现有的 Markdown 文件转换为 reStructuredText 。

**构建文档** # 在python3环境下运行

```bash
bash scripts/build_docs.sh
```

### 构建安装包

构建pip包

```bash
python setup.py sdist bdist_wheel
```

### [部署](./release.md) 