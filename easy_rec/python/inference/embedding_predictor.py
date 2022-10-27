from easy_rec.python.inference.predictor import Predictor
from easy_rec.python.utils import config_util
from easy_rec.python.protos.dataset_pb2 import DatasetConfig
import numpy as np
import pandas as pd

class EmbeddingPredictor:

    def __init__(self,
                 model_path,
                 user_feature_local_path,
                 pipeline_config_local_path,
                 get_type_defaults,
                 process_type_and_default_val,
                 group_name):

        self._model_path = model_path
        self._user_feature_local_path = user_feature_local_path
        self._pipeline_config_local_path = pipeline_config_local_path
        self._get_type_defaults = get_type_defaults
        self._process_type_and_default_val = process_type_and_default_val


    def get_type_defaults(field_type, default_val=''):
        type_defaults = {
            DatasetConfig.INT32: 0,
            DatasetConfig.INT64: 0,
            DatasetConfig.STRING: '',
            DatasetConfig.BOOL: False,
            DatasetConfig.FLOAT: 0.0,
            DatasetConfig.DOUBLE: 0.0
        }
        assert field_type in type_defaults, 'invalid type: %s' % field_type
        if default_val == '':
            default_val = type_defaults[field_type]

        if field_type == DatasetConfig.INT32:
            return int, int(default_val)
        elif field_type == DatasetConfig.INT64:
            return np.int64, np.int64(default_val)
        elif field_type == DatasetConfig.STRING:
            return str, default_val
        elif field_type == DatasetConfig.BOOL:
            return bool, default_val.lower() == 'true'
        elif field_type in [DatasetConfig.FLOAT]:
            return float, float(default_val)
        else: #field_type in [DatasetConfig.DOUBLE]:
            return np.float64, np.float64(default_val)


    def process_type_and_default_val(data, fea_name, input_types, defaults):
        data.columns = fea_name
        for idx in range(len(data.keys())):
            fea = fea_name[idx]
            default = defaults[fea]
            input_type = input_types[fea]
            data[fea].fillna(default, inplace=True)
            data[fea] = data[fea].astype(input_type)


    def inference(model_path, data, fea_keys, emb_name, id_col_name, vec_engine):
        predictor = Predictor(model_path)
        inputs = []
        for line in data.values:
            line_req = {}
            for index in range(len(line)):
                key = fea_keys[index]
                val = line[index]
                line_req[key] = val
        inputs.append(line_req)

        outputs = []
        for id_, emb in zip(data[id_col_name], predictor.predict(inputs, batch_size=1000)):
            if vec_engine=='holo' or vec_engine=='mysql':
                res = "%s %s" % (id_, emb[emb_name].decode("utf-8"))
                outputs.append(res)
            elif vec_engine=='faiss':
                res = [str(id_), ]
                res.extend(["%s:%s"%(idx+1, emb) for idx, emb in enumerate(emb[emb_name].decode("utf-8").split(','))])
                outputs.append(' '.join(res))
            if len(outputs)%10000==0:
                print("predict size: %s" % len(outputs))
        return outputs


    def predict_impl(self):

        input_types = {}
        defaults = {}
        feature_groups = {}
        pipeline_config = config_util.get_configs_from_pipeline_file(self._pipeline_config_local_path)

        for input_field in pipeline_config.data_config.input_fields:
            input_type, default_val = self._get_type_defaults(input_field.input_type, input_field.default_val)
            input_types[input_field.input_name] = input_type
            defaults[input_field.input_name] = default_val

        for fg in pipeline_config.model_config.feature_groups:
            feature_groups[fg.group_name] = fg.feature_names

        user_feas = feature_groups['user']
        user_feature = pd.read_csv(self._user_feature_local_path, header=None, sep=FLAGS.sep)
        self._process_type_and_default_val(user_feature, user_feas, input_types, defaults)


