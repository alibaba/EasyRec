pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/multi_tower_bst.config
-Dcmd=export
-Dexport_dir=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/multil_tower/savemodel/
-Dcluster='{"worker" : {"count":1, "cpu":1000, "memory":40000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-Dextra_params="--clear_export"
-DossHost={OSS_ENDPOINT}
;
