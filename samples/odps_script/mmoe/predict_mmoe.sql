drop table if exists mmoe_test_output_v1_{TIME_STAMP};
pai -name easy_rec_ext
-Dcmd=predict
-Dcluster='{"worker" : {"count":2, "cpu":1000,  "memory":20000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-Dsaved_model_dir=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/mmoe/savemodel/
-Dinput_table=odps://{ODPS_PROJ_NAME}/tables/mmoe_test_{TIME_STAMP}
-Doutput_table=odps://{ODPS_PROJ_NAME}/tables/mmoe_test_output_v1_{TIME_STAMP}
-Dexcluded_cols=label
-Dreserved_cols=ALL_COLUMNS
-Doutput_cols='logits_cvr double,probs_ctr double,logits_ctr double,probs_cvr double'
-Dbatch_size=1024
-DossHost={OSS_ENDPOINT}
;
