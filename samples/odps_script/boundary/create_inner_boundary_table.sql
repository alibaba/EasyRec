drop TABLE IF EXISTS boundary_test_{TIME_STAMP} ;
create table boundary_test_{TIME_STAMP}(
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
   ,price   double
)
;

INSERT OVERWRITE TABLE boundary_test_{TIME_STAMP}
select * from external_boundary_test_{TIME_STAMP} ;


drop TABLE IF EXISTS boundary_train_{TIME_STAMP} ;
create  table boundary_train_{TIME_STAMP}(
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
   ,price   double
)
;

INSERT OVERWRITE TABLE boundary_train_{TIME_STAMP}
select * from external_boundary_train_{TIME_STAMP} ;


drop TABLE IF EXISTS boundary_info_table_{TIME_STAMP} ;
create table boundary_info_table_{TIME_STAMP}(
    feature STRING
    ,json STRING
)
;

INSERT OVERWRITE TABLE boundary_info_table_{TIME_STAMP}
select * from external_boundary_info_table_{TIME_STAMP} ;
