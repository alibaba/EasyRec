# sample join

## label join 特征

```sql
  create temporary table odl_sample_with_lbl(
    `request_id`    STRING,
    `user_id`       STRING,
    `item_id`       STRING,
    `ln_play_time`  DOUBLE,
    `is_valid_play` BIGINT,
    `min_ts`        BIGINT,
    `max_ts`        BIGINT,
    `ts`            AS TO_TIMESTAMP(
          FROM_UNIXTIME(if (min_ts is not null and min_ts < UNIX_TIMESTAMP(),
           min_ts, UNIX_TIMESTAMP()), 'yyyy-MM-dd HH:mm:ss')),
    WATERMARK FOR `ts` AS `ts` - INTERVAL '5' SECOND
  ) WITH (
    'connector' = 'datahub',
    'endPoint' = 'http://dh-cn-beijing-int-vpc.aliyuncs.com/',
    'project' = 'easy_rec_proj',
    'topic' = 'odl_sample_with_lbl',
    'subId' = '165519436817538OG0',
    'accessId' = 'LTAIxxx',
    'accessKey' = 'xxxxxxxxx',
    'startTime' = '2022-07-02 14:30:00'
  );

  create temporary table odl_callback_log(
    `request_id`        STRING,
    `request_time`      BIGINT,
    `module`    STRING,
    `user_id`   STRING,
    `item_id`   STRING,
    `scene`     STRING,
    `generate_features` STRING,
    `ts`              AS
        TO_TIMESTAMP(FROM_UNIXTIME(if(request_time is not null and request_time < UNIX_TIMESTAMP(),
              request_time, UNIX_TIMESTAMP()), 'yyyy-MM-dd HH:mm:ss')),
    WATERMARK FOR `ts` AS `ts` - INTERVAL '5' SECOND
  ) WITH (
    'connector' = 'datahub',
    'endPoint' = 'http://dh-cn-beijing-int-vpc.aliyuncs.com/',
    'project' = 'easy_rec_proj',
    'topic' = 'odl_callback_log',
    'subId' = '16567769418786B4JH',
    'accessId' = 'LTAIxxx',
    'accessKey' = 'xxxxxx'
    'startTime' = '2022-07-02 14:30:00'
  );


  create temporary view sample_view as
  select a.request_id, a.user_id, a.item_id, a.ln_play_time, a.is_valid_play, feature, b.request_time
  from  odl_sample_with_lbl a
  inner join (
       select * from (
        select request_id, item_id, request_time, generate_features as feature, ts,
          row_number() over(partition by request_id, item_id order by proctime() asc) as rn
        from odl_callback_log
        where `module` = 'item' and (generate_features is not null and generate_features <> '')
       ) where rn = 1
  ) b
  on a.request_id = b.request_id and a.item_id = b.item_id
  where a.ts between b.ts - INTERVAL '30' SECONDS  and b.ts + INTERVAL '30' MINUTE;
```

- create temporary table注意事项:
  - ts作为watermark需要限制小于当前时间, 防止因为异常的timestamp导致watermark混乱
  - temporary table可以只列举需要的字段，不必枚举所有字段
  - datahub connector更多参数请参考[文档](https://help.aliyun.com/document_detail/177534.html)
  - kafka connector参考[文档](https://help.aliyun.com/document_detail/177144.html)
- odl_callback_log需要做去重, 防止因为重复调用造成样本重复
- flink配置开启ttl(millisecond), 控制state大小:
  ```sql
    table.exec.state.ttl: '2400000'
  ```
  - ttl(miliseconds)的设置考虑两个因素:
    - odl_sample_with_lbl相对请求时间request_time的延迟
      - ttl \< 相对延迟, 就会有样本丢失
      - 统计相对延迟:
        - 将odl_sample_with_lbl / odl_callback_log落到MaxCompute
        - 按request_id join 计算ts的差异
    - ttl越大state越大, 保存checkpoint时间越长, 性能下降
- 存储引擎开启gemini kv分离(generate_features字段值很大):
  ```sql
    state.backend.gemini.kv.separate.mode: GLOBAL_ENABLE
    state.backend.gemini.kv.separate.value.size.threshold: '500'
  ```

## 实时样本写入Datahub / Kafka

```sql
  create temporary table odl_sample_with_fea_and_lbl(
    `request_id`    string,
    `user_id`        string,
    `item_id`          string,
    `ln_play_time`  double,
    `is_valid_play` bigint,
    `feature`       string,
    `request_time`  bigint
  ) WITH (
    'connector' = 'datahub',
    'endPoint' = 'http://dh-cn-beijing-int-vpc.aliyuncs.com/',
    'project' = 'odl_sample',
    'topic' = 'odl_sample_with_fea_and_lbl',
    'subId' = '1656xxxxxx',
    'accessId' = 'LTAIxxxxxxx',
    'accessKey' = 'Q82Mxxxxxxxx'
  );
  insert into odl_sample_with_fea_and_lbl
  select * from sample_view;
```

- subId: datahub subscription id
