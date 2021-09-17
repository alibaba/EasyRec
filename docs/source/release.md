# Release & Upgrade

### Release Notes

| **Version** | **URL**                                                                                                  | **Desc**                                                                                                        |
| ----------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 20200922    | [Download](https://easyrec.oss-cn-beijing.aliyuncs.com/releases/easy_rec-20200922-py2.py3-none-any.whl)  | add support for hpo                                                                                             |
| 20201102    | [Download](https://easyrec.oss-cn-beijing.aliyuncs.com/releases/easy_rec-20201102-py2.py3-none-any.whl)  | add support for json config file; add dropout and custom activation function for dnn; add support for dssm/mmoe |
| 20201221    | [Download](https://easyrec.oss-cn-beijing.aliyuncs.com/releases/easy_rec-0.1.0-py2.py3-none-any.whl)     | add new models: DCN, DBMTL, AUTOINT                                                                             |
| 20210315    | [Download](https://easyrec.oss-cn-beijing.aliyuncs.com/release/whls/easy_rec-0.1.3-py2.py3-none-any.whl) | add support for knowledge distillation, MIND models                                                             |

### 本地升级

```bash
pip install -U https://easyrec.oss-cn-beijing.aliyuncs.com/releases/easy_rec-0.1.0-py2.py3-none-any.whl
```

### EMR EasyRec升级

```bash
su hadoop
cd /home/hadoop
wget https://easyrec.oss-cn-beijing.aliyuncs.com/releases/releases/upgrade_easy_rec.sh -O upgrade_easy_rec.sh
chmod a+rx upgrade_easy_rec.sh
sh upgrade_easy_rec.sh https://easyrec.oss-cn-beijing.aliyuncs.com/releases/easy_rec-0.1.0-py2.py3-none-any.whl
```

### PAI(Max Compute) EasyRec升级

```bash
sh pai_jobs/deploy_ext.sh -V ${VERSION} -O
```

执行的时候需要加上

```
pai -name easy_rec_ext
-Dres_project=${YOUR_PROJECT_NAME}
-Dversion=${VERSION}
...
;
```
