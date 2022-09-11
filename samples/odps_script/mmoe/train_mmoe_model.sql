pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/mmoe_demo.config
-Dcmd=train
-Dtables=odps://{ODPS_PROJ_NAME}/tables/mmoe_train_{TIME_STAMP},odps://{ODPS_PROJ_NAME}/tables/mmoe_test_{TIME_STAMP}
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":2, "cpu":1000, "memory":40000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
-Deval_method=separate
;
