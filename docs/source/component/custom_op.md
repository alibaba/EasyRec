# 使用自定义 OP

当内置的tf算子不能满足业务需求，或者通过组合现有算子实现需求的性能较差时，可以考虑自定义tf的OP。

1. 实现自定义算子，编译为动态库
   - 参考官方示例：[TensorFlow Custom Op](https://github.com/tensorflow/custom-op/)
   - 注意：自定义Op的编译依赖tf版本需要与执行时的tf版本保持一致
   - 您可能需要为离线训练 与 在线推理服务 编译两个不同依赖环境的动态库
     - 在PAI平台上需要依赖 tf 1.12 版本编译
     - 在EAS的 [EasyRec Processor](https://help.aliyun.com/zh/pai/user-guide/easyrec) 中使用自定义Op需要依赖 tf 2.10.1 编译
1. 在`EasyRec`中使用自定义Op的步骤
   1. 下载EasyRec的最新[源代码](https://github.com/alibaba/EasyRec)
   1. 把上一步编译好的动态库放到`easy_rec/python/ops/${tf_version}`目录，注意版本要子目录名一致
   1. 开发一个使用自定义Op的组件
      - 新组件的代码添加到 `easy_rec/python/layers/keras/custom_ops.py`
      - `custom_ops.py` 提供了一个自定义Op组件的示例
      - 声明新组件，在`easy_rec/python/layers/keras/__init__.py`文件中添加导出语句
   1. 编写模型配置文件，使用组件化的方式搭建模型，包含新定义的组件（参考下文）
   1. 运行`pai_jobs/deploy_ext.sh`脚本，打包EasyRec，并把打好的资源包（`easy_rec_ext_${version}_res.tar.gz`）上传到MaxCompute项目空间
   1. (在DataWorks里 or 用odpscmd客户端工具) 训练 & 评估 & 导出 模型

## 导出自定义Op的动态库到 saved_model 的 assets 目录

```bash
pai -name easy_rec_ext
-Dcmd='export'
-Dconfig='oss://cold-start/EasyRec/custom_op/pipeline.config'
-Dexport_dir='oss://cold-start/EasyRec/custom_op/export/final_with_lib'
-Dextra_params='--asset_files oss://cold-start/EasyRec/config/libedit_distance.so'
-Dres_project='pai_rec_test_dev'
-Dversion='0.7.5'
-Dbuckets='oss://cold-start/'
-Darn='acs:ram::XXXXXXXXXX:role/aliyunodpspaidefaultrole'
-DossHost='oss-cn-beijing-internal.aliyuncs.com'
;
```

**注意**：

1. 在 训练、评估、导出 命令中需要用`-Dres_project`指定上传easyrec资源包的MaxCompute项目空间名
1. 在 训练、评估、导出 命令中需要用`-Dversion`指定资源包的版本
1. asset_files参数指定的动态库会被线上推理服务加载，因此需要在与线上推理服务一致的tf版本上编译。（目前是EAS平台的EasyRec Processor依赖 tf 2.10.1版本）。
   - 如果 asset_files 参数还需要指定其他文件路径（比如 fg.json），多个路径之间用英文逗号隔开。
1. 再次强调一遍，**导出的动态库依赖的tf版本需要与推理服务依赖的tf版本保持一致**

## 自定义Op的示例

```protobuf
feature_config: {
  ...
  features: {
    feature_name: 'raw_genres'
    input_names: 'genres'
    feature_type: PassThroughFeature
  }
  features: {
    feature_name: 'raw_title'
    input_names: 'title'
    feature_type: PassThroughFeature
  }
}
model_config: {
  model_class: 'RankModel'
  model_name: 'MLP'
  feature_groups: {
    group_name: 'text'
    feature_names: 'raw_genres'
    feature_names: 'raw_title'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'features'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'gender'
    feature_names: 'age'
    feature_names: 'occupation'
    feature_names: 'zip_id'
    feature_names: 'movie_year_bin'
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: 'text'
      inputs {
        feature_group_name: 'text'
      }
      raw_input {
      }
    }
    blocks {
      name: 'edit_distance'
      inputs {
        block_name: 'text'
      }
      keras_layer {
        class_name: 'EditDistance'
      }
    }
    blocks {
      name: 'mlp'
      inputs {
        feature_group_name: 'features'
      }
      inputs {
        block_name: 'edit_distance'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [256, 128]
        }
      }
    }
  }
  model_params {
    l2_regularization: 1e-5
  }
  embedding_regularization: 1e-6
}
```

1. 如果自定义Op需要处理原始输入特征，则在定义特征时指定 `feature_type: PassThroughFeature`
   - 非 `PassThroughFeature` 类型的特征会在预处理阶段做一些变换，组件代码里拿不到原始值
1. 自定义Op需要处理的原始输入特征按照顺序放置到同一个`feature group`内
1. 配置一个类型为`raw_input`的输入组件，获取原始输入特征
   - 这是目前EasyRec支持的读取原始输入特征的唯一方式
