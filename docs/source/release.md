# Release & Upgrade

### Release Notes

| **Version** | **URL** | **Desc** |
| \-\-\-\-\-\-\-\-\--- | \-\-\-\-\-\-- | \-\-\-\-\-\--- |
|             |         |          |

### 本地升级

```bash
pip install -U https://easy-rec.oss-cn-hangzhou.aliyuncs.com/releases/easy_rec-0.1.0-py2.py3-none-any.whl
```

### EMR EasyRec升级

```bash
su hadoop
cd /home/hadoop
wget https://easy-rec.oss-cn-hangzhou.aliyuncs.com/releases/releases/upgrade_easy_rec.sh -O upgrade_easy_rec.sh
chmod a+rx upgrade_easy_rec.sh
sh upgrade_easy_rec.sh https://easy-rec.oss-cn-hangzhou.aliyuncs.com/releases/easy_rec-0.1.0-py2.py3-none-any.whl
```
