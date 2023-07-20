# EasyRecProcessor Client

Demo

```bash
python -m easy_rec.python.client.client_demo \
  --endpoint 1301055xxxxxxxxx.cn-hangzhou.pai-eas.aliyuncs.com \
  --service_name ali_rec_rnk_sample_rt_v3 \
  --token MmQ3Yxxxxxxxxxxx \
  --table_schema data/test/client/user_table_schema \
  --table_data data/test/client/user_table_data \
  --item_lst data/test/client/item_lst

# output:
#   results {
#     key: "item_0"
#     value {
#       scores: 0.0
#       scores: 0.0
#     }
#   }
#   results {
#     key: "item_1"
#     value {
#       scores: 0.0
#       scores: 0.0
#     }
#   }
#   results {
#     key: "item_2"
#     value {
#       scores: 0.0
#       scores: 0.0
#     }
#   }
#   outputs: "probs_is_click"
#   outputs: "probs_is_go"
```
