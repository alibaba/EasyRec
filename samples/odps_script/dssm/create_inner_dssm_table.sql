drop TABLE IF EXISTS dssm_test_{TIME_STAMP} ;
create table dssm_test_{TIME_STAMP}(
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

INSERT OVERWRITE TABLE dssm_test_{TIME_STAMP}
select * from external_dssm_test_{TIME_STAMP} ;


drop TABLE IF EXISTS dssm_train_{TIME_STAMP} ;
create  table dssm_train_{TIME_STAMP}(
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

INSERT OVERWRITE TABLE dssm_train_{TIME_STAMP}
select * from external_dssm_train_{TIME_STAMP} ;
