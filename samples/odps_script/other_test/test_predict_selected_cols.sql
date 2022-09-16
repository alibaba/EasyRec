drop table if exists deepfm_output_v1_{TIME_STAMP};
pai -name easy_rec_ext
-Dcmd=predict
-Dcluster='{"worker" : {"count":2, "cpu":1000,  "memory":20000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-Dsaved_model_dir=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/dwd_avazu_ctr2/checkpoints5/savemodel/
-Dinput_table=odps://{ODPS_PROJ_NAME}/tables/deepfm_test_{TIME_STAMP}
-Doutput_table=odps://{ODPS_PROJ_NAME}/tables/deepfm_output_v1_{TIME_STAMP}
-Dselected_cols=hour,c1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,c14,c15,c16,c17,c18,c19,c20,c21
-Dreserved_cols=label,banner_pos,site_id
-Dbatch_size=1024
-DossHost={OSS_ENDPOINT}
;
