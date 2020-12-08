drop TABLE IF EXISTS external_dssm_test_{TIME_STAMP} ;
create EXTERNAL table external_dssm_test_{TIME_STAMP}(
   clk   bigint
   ,buy   bigint
   ,pid   string
   ,adgroup_id   string
   ,cate_id   string
   ,campaign_id   string
   ,customer   string
   ,brand   string
   ,user_id   string
   ,cms_segid   string
   ,cms_group_id   string
   ,final_gender_code   string
   ,age_level   string
   ,pvalue_level   string
   ,shopping_level   string
   ,occupation   string
   ,new_user_class_level   string
   ,tag_category_list   string
   ,tag_brand_list   string
   ,price   bigint
)
STORED BY 'com.aliyun.odps.CsvStorageHandler'
WITH SERDEPROPERTIES (
 'odps.properties.rolearn'='{ROLEARN}'
)
LOCATION 'oss://{OSS_BUCKET_NAME}/{EXP_NAME}/test_data/tb_data/train/'
;


drop TABLE IF EXISTS external_dssm_train_{TIME_STAMP} ;
create EXTERNAL table external_dssm_train_{TIME_STAMP}(
   clk   bigint
   ,buy   bigint
   ,pid   string
   ,adgroup_id   string
   ,cate_id   string
   ,campaign_id   string
   ,customer   string
   ,brand   string
   ,user_id   string
   ,cms_segid   string
   ,cms_group_id   string
   ,final_gender_code   string
   ,age_level   string
   ,pvalue_level   string
   ,shopping_level   string
   ,occupation   string
   ,new_user_class_level   string
   ,tag_category_list   string
   ,tag_brand_list   string
   ,price   bigint
)
STORED BY 'com.aliyun.odps.CsvStorageHandler'
WITH SERDEPROPERTIES (
 'odps.properties.rolearn'='{ROLEARN}'
)
LOCATION 'oss://{OSS_BUCKET_NAME}/{EXP_NAME}/test_data/tb_data/test/'
;
