pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/dwd_avazu_ctr_deepmodel_ext_v5.config
-Dcmd=evaluate
-Dcheckpoint_path=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/dwd_avazu_ctr2/checkpoints5/model.ckpt-100
-Dtables=odps://{ODPS_PROJ_NAME}/tables/deepfm_test_{TIME_STAMP}
-Dcluster='{"worker" : {"count":1, "cpu":1000, "memory":40000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
;
