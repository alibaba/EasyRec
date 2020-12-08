pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/taobao_fg.config
-Dcmd=export
-Dexport_dir=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/ev/export
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
;
