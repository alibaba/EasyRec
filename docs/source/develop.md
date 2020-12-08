# Develop

### Code Style

We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:

- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [yapf](https://github.com/google/yapf): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

Style configurations of yapf and isort can be found in [setup.cfg](setup.cfg).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `isort`, `trailing whitespaces`,
fixes `end-of-files`, sorts `requirments.txt` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](.pre-commit-config.yaml).

After you clone the repository, you will need to install initialize pre-commit hook.

```bash
pip install -U pre-commit
```

From the repository folder

```bash
pre-commit install
```

After this on every commit check code linters and formatter will be enforced.

If you only want to format and lint your code, you can run

```bash
pre-commit run -a
```

### Test

#### Unit test

```bash
TEST_DEVICES=0,1 sh scripts/ci_test.sh
```

#### Odps Test

```bash
TEMPDIR=/tmp python -m easy_rec.python.test.odps_run --oss_config ~/.ossutilconfig [--odps_config {ODPS_CONFIG} --algo_project {ALOG_PROJ}  --arn acs:ram::xxx:role/yyy TestPipelineOnOdps.*]
```

#### Test data

If you add new data, please do the following to commit it to git-lfs before "git commit":

```bash
python git-lfs/git_lfs.py add data/test/new_data
python git-lfs/git_lfs.py push
```

### Document

We support [markdown](https://guides.github.com/features/mastering-markdown/) format documents and
[restructuredtext](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) format.

If the document contains formulas or tables, we suggest you use restructuredtext format or use
[md-to-rst](https://cloudconvert.com/md-to-rst) converting the existing markdown files to restructuredtext.

build documents

```bash
# Run in python3 environment
sh scripts/build_docs.sh
```

### Build Package

build pip package

```bash
python setup.py sdist bdist_wheel
```

### Deploy

```bash
sh pai_jobs/deploy_ext.sh
```
