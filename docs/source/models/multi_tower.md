# MultiTower

### 简介

- 多塔模型效果比单塔模型有明显的提升
- 不采用FM，所以embedding可以有不同的dimension。

![multi_tower.png](../../images/models/multi_tower.png)

### 模型配置

```protobuf
model_config: {
  model_class: 'MultiTower'
  feature_groups: {
    group_name: 'user'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    ...
    feature_names: 'new_user_class_level'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'item'
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    ...
    feature_names: 'price'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'combo'
    feature_names: 'pid'
    feature_names: 'tag_category_list'
    feature_names: 'tag_brand_list'
    wide_deep: DEEP
  }

  multi_tower {
    towers {
      input: "user"
      dnn {
        hidden_units: [256, 128, 96, 64]
      }
    }
    towers {
      input: "item"
      dnn {
        hidden_units: [256, 128, 96, 64]
      }
    }
    towers {
      input: "combo"
      dnn {
        hidden_units: [128, 96, 64, 32]
      }
    }
    final_dnn {
      hidden_units: [128, 96, 64, 32, 16]
    }
    l2_regularization: 1e-6
  }
  embedding_regularization: 1e-4
}
```

- feature\_groups: 不同的特征组，如user feature为一组，item feature为一组, combo feature为一组
  - group\_name: 可以根据实际情况取
  - wide\_deep: 必须是DEEP
- towers:
  - 每个feature\_group对应了一个tower, tower的input必须和feature\_groups的group\_name对应
  - dnn: 深度网络
    - hidden\_units: 定义不同层的channel数目，即神经元数目
- final\_dnn 整合towers和din\_towers的输入
  - hidden\_units: dnn每一层的channel数目，即神经元的数目
- l2\_regularization: L2正则，防止overfit
- embedding\_regularization: embedding的L2正则

### 示例config

[multi\_tower\_demo.config](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/config/multi-tower.config)

### 参考论文

自研模型，暂无参考论文
