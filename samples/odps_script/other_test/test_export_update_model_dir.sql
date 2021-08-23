pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/dwd_avazu_ctr_deepmodel_ext_v5_export_test.config
-Dcmd=export
-Dmodel_dir=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/dwd_avazu_ctr2/checkpoints5/
-Dexport_dir=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/dwd_avazu_ctr2/checkpoints5/savemodel_v1/
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
;
