drop TABLE IF EXISTS external_ev_test_{TIME_STAMP} ;
create EXTERNAL table external_ev_test_{TIME_STAMP}(
   clk   bigint
   ,user_id   string
   ,item_id string
   ,features string
)
STORED BY 'com.aliyun.odps.CsvStorageHandler'
WITH SERDEPROPERTIES (
 'odps.properties.rolearn'='{ROLEARN}',
 'odps.text.option.delimiter'=';'
)
LOCATION 'oss://{OSS_BUCKET_NAME}/{EXP_NAME}/test_data/fg_data/test/'
;


drop TABLE IF EXISTS external_ev_train_{TIME_STAMP} ;
create EXTERNAL table external_ev_train_{TIME_STAMP}(
   clk   bigint
   ,user_id   string
   ,item_id string
   ,features string
)
STORED BY 'com.aliyun.odps.CsvStorageHandler'
WITH SERDEPROPERTIES (
 'odps.properties.rolearn'='{ROLEARN}',
 'odps.text.option.delimiter'=';'
)
LOCATION 'oss://{OSS_BUCKET_NAME}/{EXP_NAME}/test_data/fg_data/train/'
;

drop table if exists inner_ev_test_{TIME_STAMP};
create table inner_ev_test_{TIME_STAMP} as select * from external_ev_test_{TIME_STAMP};

drop table if exists inner_ev_train_{TIME_STAMP};
create table inner_ev_train_{TIME_STAMP} as select * from external_ev_train_{TIME_STAMP};
