pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/dwd_avazu_ctr_deepmodel_ext_v5.config
-Dcmd=train
-Dtrain_tables=odps://{ODPS_PROJ_NAME}/tables/deepfm_train_{TIME_STAMP}
-Deval_tables=odps://{ODPS_PROJ_NAME}/tables/deepfm_test_{TIME_STAMP}
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":5, "cpu":1000, "memory":40000}}'
-Deval_method='separate'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
;
