drop TABLE IF EXISTS multi_tower_test_{TIME_STAMP} ;
create table multi_tower_test_{TIME_STAMP}(
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
;

tunnel upload {TEST_DATA_DIR}/tb_data/test_{TIME_STAMP} multi_tower_test_{TIME_STAMP};

drop TABLE IF EXISTS multi_tower_train_{TIME_STAMP} ;
create  table multi_tower_train_{TIME_STAMP}(
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
;

tunnel upload {TEST_DATA_DIR}/tb_data/train_{TIME_STAMP} multi_tower_train_{TIME_STAMP};
