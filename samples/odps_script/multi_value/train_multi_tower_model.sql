pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/taobao_multi_tower_multi_value_test.config
-Dcmd=train
-Dtables=odps://{ODPS_PROJ_NAME}/tables/multi_value_train_{TIME_STAMP},odps://{ODPS_PROJ_NAME}/tables/multi_value_test_{TIME_STAMP}
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":2, "cpu":1000, "memory":40000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
-Deval_method=separate
;
