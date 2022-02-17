drop TABLE IF EXISTS external_query_vector_{TIME_STAMP};
create EXTERNAL table external_query_vector_{TIME_STAMP}(
    query_id BIGINT
    ,vector string
)
STORED BY 'com.aliyun.odps.CsvStorageHandler'
WITH SERDEPROPERTIES (
 'odps.properties.rolearn'='{ROLEARN}'
)
LOCATION 'oss://{OSS_ENDPOINT_INTERNAL}/{OSS_BUCKET_NAME}/{EXP_NAME}/test_data/query/'
;

drop TABLE IF EXISTS external_doc_vector_{TIME_STAMP};
create EXTERNAL table external_doc_vector_{TIME_STAMP}(
    doc_id BIGINT
    ,vector string
)
STORED BY 'com.aliyun.odps.CsvStorageHandler'
WITH SERDEPROPERTIES (
 'odps.properties.rolearn'='{ROLEARN}'
)
LOCATION 'oss://{OSS_ENDPOINT_INTERNAL}/{OSS_BUCKET_NAME}/{EXP_NAME}/test_data/doc/'
;

