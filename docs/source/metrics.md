# 常见Metrics计算

## AUC计算

```sql
pai -name=evaluate -project=algo_public
    -DoutputMetricTableName=output_metric_table
    -DoutputDetailTableName=output_detail_table
    -DinputTableName=input_data_table
    -DlabelColName=label
    -DscoreColName=score;
```

## Group AUC计算

```sql
select group_name, (rank_pos - pos_cnt * (pos_cnt+1) / 2) / (pos_cnt * neg_cnt) as gauc
from (
    select group_name,
            sum(if(label=1, rn, 0)) as rank_pos,
            sum(if(label=1, 1,0)) as pos_cnt,
            sum(if(label=0, 1, 0)) as neg_cnt
    from (
        select group_name, label, rank() over(partition by group_name order by probs asc) as rn
        from your_table
    )
    group by group_name
);
```
