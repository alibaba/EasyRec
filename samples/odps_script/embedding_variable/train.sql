pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/taobao_fg.config
-Dcmd=train
-Dtrain_tables=odps://{ODPS_PROJ_NAME}/tables/inner_ev_train_{TIME_STAMP},odps://{ODPS_PROJ_NAME}/tables/inner_ev_train_{TIME_STAMP}
-Deval_tables=odps://{ODPS_PROJ_NAME}/tables/inner_ev_test_{TIME_STAMP}
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "memory":40000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
-Dselected_cols=clk,features
-Deval_method=separate
;
