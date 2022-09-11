pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/multi_tower_bst.config
-Dcmd=evaluate
-Dtables=odps://{ODPS_PROJ_NAME}/tables/multi_tower_test_{TIME_STAMP}
-Dcluster='{"worker" : {"count":1, "cpu":1000, "memory":40000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
;
