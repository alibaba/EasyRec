drop TABLE IF EXISTS deepfm_train_{TIME_STAMP};
create table deepfm_train_{TIME_STAMP}(
    label BIGINT
    ,`hour` string
    ,c1 STRING
    ,banner_pos STRING
    ,site_id STRING
    ,site_domain STRING
    ,site_category STRING
    ,app_id STRING
    ,app_domain STRING
    ,app_category STRING
    ,device_id STRING
    ,device_ip STRING
    ,device_model STRING
    ,device_type STRING
    ,device_conn_type STRING
    ,c14 STRING
    ,c15 STRING
    ,c16 STRING
    ,c17 STRING
    ,c18 STRING
    ,c19 STRING
    ,c20 STRING
    ,c21 STRING
)
;

tunnel upload {TEST_DATA_DIR}/train_{TIME_STAMP} deepfm_train_{TIME_STAMP};

drop TABLE IF EXISTS deepfm_test_{TIME_STAMP};
create table deepfm_test_{TIME_STAMP}(
    label BIGINT
    ,`hour` string
    ,c1 STRING
    ,banner_pos STRING
    ,site_id STRING
    ,site_domain STRING
    ,site_category STRING
    ,app_id STRING
    ,app_domain STRING
    ,app_category STRING
    ,device_id STRING
    ,device_ip STRING
    ,device_model STRING
    ,device_type STRING
    ,device_conn_type STRING
    ,c14 STRING
    ,c15 STRING
    ,c16 STRING
    ,c17 STRING
    ,c18 STRING
    ,c19 STRING
    ,c20 STRING
    ,c21 STRING
)
;

tunnel upload {TEST_DATA_DIR}/test_{TIME_STAMP} deepfm_test_{TIME_STAMP};
