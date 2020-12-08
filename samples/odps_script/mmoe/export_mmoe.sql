pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/mmoe_demo.config
-Dcmd=export
-Dexport_dir=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/mmoe/savemodel/
-Dcluster='{"worker" : {"count":1, "cpu":1000, "memory":40000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
;
