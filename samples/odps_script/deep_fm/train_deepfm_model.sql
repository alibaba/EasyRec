pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/dwd_avazu_ctr_deepmodel_ext.config
-Dcmd=train
-Dtables=odps://{ODPS_PROJ_NAME}/tables/deepfm_train_{TIME_STAMP},odps://{ODPS_PROJ_NAME}/tables/deepfm_test_{TIME_STAMP}
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":2, "cpu":1000, "memory":40000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
-Deval_method=separate
;
