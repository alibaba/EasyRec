drop table if exists multi_tower_test_output_v1_{TIME_STAMP};
pai -name easy_rec_ext
-Dcmd=predict
-Dcluster='{"worker" : {"count":2, "cpu":1000,  "memory":20000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-Dsaved_model_dir=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/multil_tower/savemodel/
-Dinput_table=odps://{ODPS_PROJ_NAME}/tables/multi_tower_test_{TIME_STAMP}
-Doutput_table=odps://{ODPS_PROJ_NAME}/tables/multi_tower_test_output_v1_{TIME_STAMP}
-Dexcluded_cols=label
-Dreserved_cols=ALL_COLUMNS
-Dbatch_size=1024
-DossHost={OSS_ENDPOINT}
;
