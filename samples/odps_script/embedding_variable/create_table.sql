drop TABLE IF EXISTS inner_ev_test_{TIME_STAMP};
create table inner_ev_test_{TIME_STAMP}(
   clk   bigint
   ,user_id   string
   ,item_id string
   ,features string
);

tunnel upload {TEST_DATA_DIR}/fg_data/test_${TIME_STAMP} inner_ev_test_{TIME_STAMP} -fd=';';

drop TABLE IF EXISTS inner_ev_train_{TIME_STAMP};
create table inner_ev_train_{TIME_STAMP}(
   clk   bigint
   ,user_id   string
   ,item_id string
   ,features string
);

tunnel upload {TEST_DATA_DIR}/fg_data/train_${TIME_STAMP} inner_ev_train_{TIME_STAMP} -fd=';';
