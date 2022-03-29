pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/dwd_avazu_ctr_deepmodel_ext.config
-Dcmd=export_checkpoint
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
-Dexport_dir=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/dwd_avazu_ctr/export_model/savemodel/
-Dbatch_size=256;
