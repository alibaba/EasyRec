pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/dwd_avazu_ctr_deepmodel_ext_v2.config
-Dcmd=train
-Dtables=odps://{ODPS_PROJ_NAME}/tables/deepfm_train_{TIME_STAMP},odps://{ODPS_PROJ_NAME}/tables/deepfm_test_{TIME_STAMP}
-Ddistribute_strategy=mirrored
-DgpuRequired=300
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
;
