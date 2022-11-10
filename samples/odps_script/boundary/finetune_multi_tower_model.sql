pai -name easy_rec_ext
-Dconfig=oss://{OSS_BUCKET_NAME}/{EXP_NAME}/configs/taobao_multi_tower_boundary_test.config
-Dcmd=train
-Dboundary_table=odps://{ODPS_PROJ_NAME}/tables/boundary_info_table_{TIME_STAMP}
-Dmodel_dir="oss://{OSS_BUCKET_NAME}/easy_rec_odps_test/{EXP_NAME}/edit_boundary_test/finetune/"
-Dfine_tune_checkpoint='oss://{OSS_BUCKET_NAME}/easy_rec_odps_test/{EXP_NAME}/edit_boundary_test/checkpoints/'
-Dedit_config_json='{"train_config.num_steps": 200}'
-Dtrain_tables=odps://{ODPS_PROJ_NAME}/tables/boundary_train_{TIME_STAMP}
-Deval_tables=odps://{ODPS_PROJ_NAME}/tables/boundary_test_{TIME_STAMP}
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":2, "cpu":1000, "memory":40000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
-Deval_method=separate
;
