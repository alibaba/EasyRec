

# 6.8 sequence 类 feature

## 基本场景

⽤户的历史⾏为也是⼀个很重要的 feature。历史⾏为通常是⼀个序列，例如点击序列、购买序列等，组成这个序列的实体可能是商品本身。
例如我们需要对⽤户的点击序列进⾏ fg，序列⻓度为 30，每个序列提取 nid 和 price, seq_context 特征。正常 item 维度有⼀个 feat0 特征。配置如下：

```json
{
    "features":[
        {
            "feature_type":"raw_feature",
            "feature_name":"feat0",
            "expression":"user:feat0"
        },
        {
            "sequence_name":"click",
            "sequence_column":"click_field",
            "sequence_length":10,
            "sequence_delim":";",
            "attribute_delim":"#",
            "sequence_table":"item",
            "sequence_pk":"user:user_behavior_seq",
            "features":[
                {
                    "feature_name":"nid",
                    "feature_type":"id_feature",
                    "value_type":"String",
                    "expression":"item:nid"
                },
                {
                    "feature_name":"price",
                    "feature_type":"raw_feature",
                    "expression":"item:price"
                },
                {
                    "feature_name":"seq_context",
                    "feature_type":"raw_feature",
                    "expression":"user:seq_context"
                }
            ]
        }
    ]
}
```

## 在线 FG

我们⽀持两种⽅式获取⾏为序列，⼀种如例⼦所示，我们以 sequence_pk 配置的字段为主键，RTP 会帮忙从 item 表中查到序列的对应字段值；另⼀种⽤户需要在 qinfo 中准备好所有的字段。

### RTP 取 sequence 字段

第⼀种情况，`sequence_pk` 的⻓度应该⼩于等于 `sequence_length` 。如果 `sequence_pk` 指定的值不⾜ `sequence_length` 个会补⻬到 `sequence_length` ⻓度，fg 的结果会出默认值（dense 类是 0，sparse 类为空）。
qinfo 例⼦：

```json
 {
 	"user:user_behavior_seq" : ["item_id_1", "item_id_2"]
 }
```

### qinfo 传递 sequence 字段

第⼆种情况，sequence feature 也⽀持所有的序列内容都从 qinfo 中传递。例如这⾥的user:seq_context 数组，他的值分别对应 click_0 和 click_1 。这种情况下⽤户可以忽略sequence_table 和 sequence_pk 。
qinfo 例⼦：

```json
 {
  "user:feat0" : 1.0,
  "user:user_behavior_seq" : [0, 1],
  "user:seq_context" : [2, 3]
 }
```

### context seq使⽤

```
{
	"features": [{
		"sequence_name": "click",
		"sequence_column": "click_field",
		"sequence_length": 30,
		"sequence_delim": ";",
		"attribute_delim": "#",
		"sequence_table": "context_table",
		"sequence_pk": "context:context_seq_id",
		"features": [{
				"feature_name": "cid",
				"feature_type": "id_feature",
				"value_type": "String",
				"expression": "context_table:cid"
			},
			{
				"feature_name": "price",
				"feature_type": "raw_feature",
				"expression": "context_table:price"
			},
			{
				"feature_name": "seq_context",
				"feature_type": "raw_feature",
				"expression": "context:seq_context"
			}
		]
	}]
}
```

context seq特征与user seq类似，区别是每个context是batch size维度的，user seq是⼀维的
配置如上，context_seq_id为输⼊的context字段
第⼀类特征：需要查context_table，如price特征，会根据context_seq_id查询context_table中的price，然后做fg，
第⼆类特征：不需要context_table，如seq_context特征，会直接取seq_context做fg，

### item seq使⽤

增加"is_item_seq": true配置，如下，

```json
{
	"features": [{
		"sequence_name": "item_pic_seq",
		"sequence_column": "item__pic_vec_seq",
		"sequence_table": "pic_table",
		"sequence_pk": "item:pic_sop_id_list",
		"attribute_delim": "#",
		"feature_name": "item_pic_seq",
		"sequence_length": 10,
		"is_item_seq": true,
		"features": [{
				"normalizer": "method=log10",
				"feature_type": "id_feature",
				"shared_name": "pic_pv",
				"hash_bucket_size": 10,
				"need_prefix": false,
				"embedding_dimension": 8,
				"value_type": "String",
				"feature_name": "pic_pv",
				"expression": "pic_table:pv"
			},
			{
				"normalizer": "method=log10",
				"feature_type": "id_feature",
				"shared_name": "pic_ipv",
				"hash_bucket_size": 10,
				"need_prefix": false,
				"embedding_dimension": 8,
				"value_type": "String",
				"feature_name": "pic_ipv",
				"expression": "pic_table:ipv"
			},
			{
				"feature_type": "id_feature",
				"shared_name": "bandit_level",
				"hash_bucket_size": 100,
				"need_prefix": false,
				"embedding_dimension": 4,
				"value_type": "String",
				"feature_name": "bandit_level",
				"expression": "pic_table:bandit_level"
			},
			{
				"feature_type": "id_feature",
				"shared_name": "is_fake_long",
				"hash_bucket_size": 100,
				"need_prefix": false,
				"embedding_dimension": 4,
				"value_type": "String",
				"feature_name": "is_fake_long",
				"expression": "pic_table:is_fake_long"
			}
		]
	}]
}
```

## 离线 FG

​		⽬前使⽤ sequence feature 要求使⽤ 新新版 feature_generator_java ， tensorflow 训练流程要求使⽤ rtp_fg.parse_genreated_fg。
​		离线阶段没有sequence表去查，⽽是通过`sequence_column` 读取本来应该去表⾥查的字段。因此，`sequence_column ，sequence_delim ，attribute_delim` 这三个字段只有在离线 fg 阶段有⽤。`sequence_column` 是数据源odps表⾥所有 sequence 特征输⼊的字段名，离线fg会根据这个字段⾥的值⽣成sequence feature，该字段内容是 kv 格式的。`sequence_delim` 是sequence 中⾏为之间的分隔符，`attribute_delim` 是实际字段名字和字段值的分隔符。
​		sequence_length 是 sequence 的⻓度，⽤户需要保证字段内容⼀定是补⻬到这个⻓度的。以上⾯的配置为例，⽤户需要有⼀个名字叫 click_field 的字段。假设某条record⾥它的内容是：

```
1 item__nid:11#item__price:2.0\u001D3.0;item__nid:22#item__price:4.0\u001D5.0
```

表示 `click_0` 和 `click_1` 中的字段分别是 `item__nid:11 item__price:2.0\u001D3.0` 和`item__nid:22 item__price:4.0\u001D5.0` 。fg 的结果会是：

```
"click_0_nid", "nid_11"
"click_0_price", "2.0\u001D3.0"
"click_0_seq_context", "0"
"click_1_nid", "nid_22"
"click_1_price", "4.0\u001D5.0"
"click_1_seq_context", "0"
```

`rtp_fg.parse_genreated_fg` 的结果中我们可以获得 `click_0_nid , click_0_price ,click_0_seq_context ，click_1_nid , click_1_price , click_1_seq_context ，`分别对应 sequence 中两个 item 的结果。