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

### 增加新的运行命令（cmd）

增加新的运行命令需要修改`xflow`的配置和脚本，文件位置：

- 弹内用户： `pai_jobs/easy_rec_flow`
- 公有云用户：`pai_jobs/easy_rec_flow_ex`

升级xflow对外发布之前，需要严格测试，影响面很大，会影响所有用户。

更建议的方式是不增加新的运行命令，新增功能通过`cmd=custom`命令来运行，通过`entryFile`参数指定新增功能的运行脚本，
需要额外参数时，通过`extra_params`参数传递。示例如下：

```
pai -name easy_rec_ext
  -Dcmd='custom'
  -DentryFile='easy_rec/python/tools/feature_selection.py'
  -Dextra_params='--topk 1000'
```

### 测试

#### 单元测试

```bash
sh scripts/ci_test.sh
```

- 运行单个测试用例

```bash
TEST_DEVICES='' python -m easy_rec.python.test.train_eval_test TrainEvalTest.test_tfrecord_input
```

#### Odps 测试

```bash
TMPDIR=/tmp python -m easy_rec.python.test.odps_run --oss_config ~/.ossutilconfig [--odps_config {ODPS_CONFIG} --algo_project {ALOG_PROJ}  --arn acs:ram::xxx:role/yyy TestPipelineOnOdps.*]
```

#### 测试数据

测试数据放在data/test目录下面, remote存储在oss://easyrec bucket里面, 使用git-lfs组件管理测试数据.

- 从remote同步数据:

  ```bash
  python git-lfs/git_lfs.py pull
  ```

- 增加新数据:

- git-lfs配置文件: .git_oss_config_pub

  ```yaml
  bucket_name = easyrec
  git_oss_data_dir = data/git_oss_sample_data
  host = oss-cn-beijing.aliyuncs.com
  git_oss_cache_dir = ${TMPDIR}/${PROJECT_NAME}/.git_oss_cache
  git_oss_private_config = ~/.git_oss_config_private
  ```

  - bucket_name: 数据存储的oss bucket, 默认是easyrec
  - git_oss_data_dir: oss bucket内部的存储目录
  - host: oss bucket对应的endpoint
  - git_oss_cache_dir: 更新数据时使用的本地的临时dir
  - git_oss_private_config: [ossutil](https://help.aliyun.com/document_detail/120075.html)对应的config，用于push数据到oss bucket.
    - 考虑到安全问题, oss://easyrec暂不开放提交数据到oss的权限
    - 如需要提交测试数据, 可以先提交到自己的oss bucket里面, 等pull requst merge以后，再同步到oss://easyrec里面.

- git-lfs提交命令:

```bash
python git-lfs/git_lfs.py add data/test/new_data
python git-lfs/git_lfs.py push
```

git-commit也会自动调用pre-commit hook, 执行git_lfs.py push操作.

### 文档

我们支持 [MarkDown](https://guides.github.com/features/mastering-markdown/) 格式和 [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) 格式的文档。

如果文档包含公式或表格，我们建议您使用 reStructuredText 格式或使用
[md-to-rst](https://cloudconvert.com/md-to-rst) 将现有的 Markdown 文件转换为 reStructuredText 。

**构建文档**

```bash
# 在python3环境下运行
bash scripts/build_docs.sh
```

### 构建安装包

**构建pip包**

```bash
python setup.py sdist bdist_wheel
```

### 部署

- MaxCompute和DataScience[部署文档](./release.md)
