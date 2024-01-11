# match_feature

## 功能介绍

match_feature一般用来做特征之间的匹配关系，要用到user，item和category三个字段的值。
match_feature支持两种类型，hit和multi hit。
match_feature本质是是一个两层map的匹配，user字段使用string的方式描述了一个两层map，|为第一层map的item之间的分隔符，^为第一层map的key与value之间的分隔符。,为第二层map的item之间的分隔符，:第二层map的key与value之间的分隔符。例如对于50011740^50011740:0.2,36806676:0.3,122572685:0.5|50006842^16788:0.1这样的一个string，转化为二层map就是

```json
{
	"50011740" : {
		"50011740" : 0.2,
		"36806676" : 0.3,
		"122572685" : 0.5
	},
	"50006842" : {
		"16788" : 0.1
	}
}
```

对于hit match 匹配的方式，就是用category的值在第一层map中查找，然后使用item的值在第二层map中查找，最终得到一个结果。 如果不需要使用两层匹配，只需要一层匹配，则可以在map的第一层key中填入ALL， 然后在fg配置的category一项中也填成"ALL"即可。具体见实例一。

## 配置方式

配置文件使用json格式

```json
{
    "feature_name": "user__l1_ctr_1",
    "feature_type": "match_feature",
    "category": "ALL",
    "needDiscrete": false,
    "item": "item:category_level1",
    "user": "user:l1_ctr_1",
    "matchType": "hit"
}
```

- needDiscrete
  - true: 模型使用 match_feature 输出的特征名，忽略特征值。默认为 true;
  - false: 模型取 match_feature 输出的特征值，而忽略特征名.
- matchType：
  - hit: 输出命中的feature
  - multihit: 允许category和item字段的值为MATCH_WILDCARD选项, 即"ALL", 可以匹配出多个值

## Normalizer

match_feature 支持和 raw_feature 一样的 normalizer，具体可见 [raw_feature](./RawFeature.md)。

## 配置示例

### hit

配置

```json
{
    "feature_name": "brand_hit",
    "feature_type": "match_feature",
    "category": "item:auction_root_category",
    "needDiscrete": true,
    "item": "item:brand_id",
    "user": "user:user_brand_tags_hit",
    "matchType": "hit"
}
```

假设各字段的值如下：

| user_brand_tags_hit   | `50011740^107287172:0.2,36806676:0.3,122572685:0.5\|50006842^16788816:0.1,10122:0.2,29889:0.3,30068:19` |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| brand_id              | 30068                                                                                                   |
| auction_root_category | 50006842                                                                                                |

- needDiscrete=true，结果为: brand_hit_50006842_30068_19
- needDiscrete=false，结果为: 19.0

如果只使用一层匹配，则需要将上面配置里的 category 的值改为 ALL。 假设各字段的值如下

| user_brand_tags_hit | ALL^16788816:40,10122:40,29889:20,30068:20 |
| ------------------- | ------------------------------------------ |
| brand_id            | 30068                                      |

- needDiscrete=true, 结果: brand_hit_ALL_30068_20
- needDiscrete=false, 结果: 20.0

这种情况，用户也可以考虑使用lookup_feature, user_brand_tags_hit 里面的值的格式需要更改为: "16788816:40^\]10122:40^\]29889:20^\]30068:20", 注意: '^\]'是多值分隔符(\\u001d), 是一个不可见字符。
