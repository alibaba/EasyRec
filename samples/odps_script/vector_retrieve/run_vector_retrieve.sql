pai -name easy_rec_ext
-Dcmd='vector_retrieve'
-DentryFile='run.py'
-Dquery_table='odps://{ODPS_PROJ_NAME}/tables/query_vector_{TIME_STAMP}'
-Ddoc_table='odps://{ODPS_PROJ_NAME}/tables/doc_vector_{TIME_STAMP}'
-Doutput_table='odps://{ODPS_PROJ_NAME}/tables/result_vector_{TIME_STAMP}'
-Dcluster='{"worker" : {"count":1, "cpu":800, "memory":10000}}'
-Darn={ROLEARN}
-Dbuckets=oss://{OSS_BUCKET_NAME}/
-DossHost={OSS_ENDPOINT}
-Dknn_distance=inner_product
-Dknn_num_neighbours=2
-Dknn_feature_dims=4
-Dknn_index_type='ivfflat'
-Dknn_feature_delimiter=','
;
