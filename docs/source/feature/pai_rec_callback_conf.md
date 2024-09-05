# PAI-REC 全埋点配置

## PAI-Rec引擎的callback服务文档

- [文档](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pairec/docs/pairec/html/intro/callback_api.html)

## 模板

```json
{
   ...,
   "DatahubConfs": {
     "aliyun-main-page-callback": {
       "AccessId": "LTAIxxxxx",
       "AccessKey": "Q82Mxxxxx",
       "Endpoint": "http://dh-cn-beijing-int-vpc.aliyuncs.com",
       "ProjectName": "odl_sample",
       "TopicName": "odl_callback_log"
     }
   },
   ...,
   "CallBackConfs": {
     "main_page": {
       "DataSource": {
         "Name": "aliyun-main-page-callback",
         "Type": "datahub"
       },
       "RankConf": {
         "RankAlgoList": [
           "callback_fg"
         ],
         "ContextFeatures": [
           "none"
         ],
         "Processor": "EasyRec"
       }
     }
   },
   ...,
   "FeatureConfs": {
     "main_page_callback": {
       "AsynLoadFeature": true,
       "FeatureLoadConfs": [
         {
          "FeatureDaoConf": {
            "AdapterType": "hologres",
            "HologresName": "holo-data-source-name",
            "FeatureKey": "user:uid",
            "UserFeatureKeyName": "userid",
            "HologresTableName": "public.user_feature_holo_table_name",
            "UserSelectFields": "*",
            "FeatureStore": "user"
          },
          "Features": [
            {
              "FeatureType": "new_feature",
              "FeatureName": "hour",
              "Normalizer": "hour_in_day",
              "FeatureStore": "user"
            },
            {
              "FeatureType": "new_feature",
              "FeatureName": "week_day",
              "Normalizer": "weekday",
              "FeatureStore": "user"
            }
          ]
         },
         {
          "FeatureDaoConf": {
            "AdapterType": "hologres",
            "HologresName": "holo-data-source-name",
            "FeatureKey": "user:uid",
            "UserFeatureKeyName": "user_id",
            "ItemFeatureKeyName": "item_id",
            "TimestampFeatureKeyName": "timestamp",
            "EventFeatureKeyName": "event",
            "HologresTableName": "public.online_events_table",
            "FeatureType": "sequence_feature",
            "SequenceOfflineTableName": "public.offline_events_table",
            "SequenceLength": 50,
            "SequenceDelim": ";",
            "SequenceDimFields": "",
            "SequenceName": "click_50_seq",
            "FeatureStore": "user",
            "SequenceEvent": "click,download"
          },
          "Features": []
         }
       ]
     }
   },
   ...,
   "AlgoConfs": [
     {
       "Name": "callback_fg",
       "Type": "EAS",
       "EasConf": {
         "Processor": "EasyRec",
         "Timeout": 600,
         "ResponseFuncName": "easyrecMutValResponseFunc",
         "Url": "http://xxx.vpc.cn-beijing.pai-eas.aliyuncs.com/api/predict/callback_fg",
         "Auth": "xxxxx"
       }
     }
   ],
   ...
}
```

## 配置说明

- DatahubConfs: datahub参数配置
- CallBackConfs:
  - main_page: 场景名称, 可以自定义
  - DataSource:
    - Name: 引用DatahubConfs里面的key
  - RankConf:
    - RankAlgoList: 引用AlgoConfs定义的算法Name
    - ContextFeatures: 定义context特征, 如召回分数、召回算法等
- FeatureConfs: 定义EAS callback服务需要的特征
  - main_page_callback: 命名规则: 场景名称(main_page) + "\_callback"
    - FeatureLoadConfs: 定义请求EAS callback服务的特征
      - 离线特征:
        - FeatureDaoConf: 配置来自Hologres的一组特征
          - HologresTableName: Hologres数据表名称
            - user_feature_holo_table_name: User离线数据表名
          - UserSelectFields: 选择的特征列, * 选择全部列
        - Features: PAI-REC内置特征
          - FeatureType: "new_feature"
          - FeatureName: 内置特征名字("hour")
          - Normalizer: 发送给EAS服务的特征名("hour_in_day")
      - 序列特征:
        - FeatureDaoConf: 配置离线+实时序列特征
          - FeatureType: "sequence_feature"
          - HologresTableName: 实时行为表(online_events_table), Schema:
            - user_id text
            - item_id text
            - event text: 行为类型, 如: click, download, ...
            - timestamp int8: event发生的时间戳
          - SequenceOfflineTableName: 离线行为表(offline_events_table)
            - Schema同实时行为表
          - EventFeatureKeyName: 对应行为表schema里面的event 字段名
          - SequenceEvent: 用户行为类型
          - SequenceName: sequence特征名, 对应FG里面的sequence_pk
          - SequenceLength: sequence的长度
        - 可以配置多组序列特征, 如可以再配置一组: like_seq_50
