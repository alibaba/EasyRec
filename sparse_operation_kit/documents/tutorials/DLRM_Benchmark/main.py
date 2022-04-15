import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--global_batch_size', type=int)
parser.add_argument('--amp', action='store_true', help='use auto mixed precision')
parser.add_argument('--xla', action='store_true', help='enable xla of tensorflow')
parser.add_argument('--compress', action='store_true', help='use tf.unique/tf.gather to compress/decompress embedding keys')
parser.add_argument('--custom_interact', action='store_true', help='use custom interact op')
parser.add_argument('--eval_in_last', action='store_true', help='evaluate only after the last iteration')
parser.add_argument('--use_synthetic_dataset', action='store_true', help='use synthetic dataset for profiling')
parser.add_argument('--use_splited_dataset', action='store_true', help='categories features were splited into different binary files')
parser.add_argument('--early_stop', type=int, default=-1)
args = parser.parse_args()
args.lr_schedule_steps = [
    int(2750 * 55296 / args.global_batch_size),
    int(49315 * 55296 / args.global_batch_size),
    int(27772 * 55296 / args.global_batch_size),
]
print('[Info] args:', args)

import os
if args.xla:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'

import json
import time
start_time = time.time()

import sparse_operation_kit as sok
import horovod.tensorflow as hvd
import tensorflow as tf
import numpy as np

from dataset import BinaryDataset, SplitedBinaryDataset, SyntheticDataset
from model import DLRM
from trainer import Trainer


def set_affinity(rank):
    affinity_map = {0: list(range(48,64)) + list(range(176,192)),
                    1: list(range(48,64)) + list(range(176,192)),
                    2: list(range(16,32)) + list(range(144,160)),
                    3: list(range(16,32)) + list(range(144,160)),
                    4: list(range(112,128)) + list(range(240,256)),
                    5: list(range(112,128)) + list(range(240,256)),
                    6: list(range(80,96)) + list(range(208,224)),
                    7: list(range(80,96)) + list(range(208,224))}

    my_affinity = affinity_map[rank]
    os.sched_setaffinity(0, my_affinity)


if __name__ == '__main__':

    if args.amp:
        print('[Info] use amp mode')
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    hvd.init()
    # set_affinity(hvd.rank())

    global_batch_size = args.global_batch_size
    sok.Init(global_batch_size=global_batch_size)

    with open(os.path.join(args.data_dir, 'train/metadata.json'), 'r') as f:
        metadata = json.load(f)
    print(metadata)

    model = DLRM(
        metadata['vocab_sizes'],
        num_dense_features=13,
        embedding_vec_size=128,
        bottom_stack_units=[512, 256, 128],
        top_stack_units=[1024, 1024, 512, 256, 1],
        num_gpus=hvd.size(),
        use_cuda_interact=args.custom_interact,
        compress=args.compress,
    )

    if args.use_synthetic_dataset or args.data_dir is None:
        print('[Info] Using synthetic dataset')
        dataset = SyntheticDataset(
            batch_size=global_batch_size // hvd.size(),
            num_iterations=args.early_stop if args.early_stop > 0 else 30,
            vocab_sizes=metadata['vocab_sizes'],
            prefetch=20,
        )
        test_dataset = SyntheticDataset(
            batch_size=global_batch_size // hvd.size(),
            num_iterations=args.early_stop if args.early_stop > 0 else 30,
            vocab_sizes=metadata['vocab_sizes'],
            prefetch=20,
        )
    elif args.use_splited_dataset:
        print('[Info] Using splited dataset in %s'%args.data_dir)
        dataset = SplitedBinaryDataset(
            os.path.join(args.data_dir, 'train/label.bin'),
            os.path.join(args.data_dir, 'train/dense.bin'),
            [os.path.join(args.data_dir, 'train/category_%d.bin'%i) for i in range(26)],
            metadata['vocab_sizes'],
            batch_size=global_batch_size // hvd.size(),
            drop_last=True,
            global_rank=hvd.rank(),
            global_size=hvd.size(),
            prefetch=20,
        )
        test_dataset = SplitedBinaryDataset(
            os.path.join(args.data_dir, 'test/label.bin'),
            os.path.join(args.data_dir, 'test/dense.bin'),
            [os.path.join(args.data_dir, 'test/category_%d.bin'%i) for i in range(26)],
            metadata['vocab_sizes'],
            batch_size=global_batch_size // hvd.size(),
            drop_last=False,
            global_rank=hvd.rank(),
            global_size=hvd.size(),
            prefetch=20,
        )
    else:
        print('[Info] Using dataset in %s'%args.data_dir)
        dtype = {'int32': np.int32, 'float32': np.float32}
        dataset_dir = args.data_dir
        dataset = BinaryDataset(
            os.path.join(dataset_dir, 'train/label.bin'),
            os.path.join(dataset_dir, 'train/dense.bin'),
            os.path.join(dataset_dir, 'train/category.bin'),
            batch_size=global_batch_size // hvd.size(),
            drop_last=True,
            global_rank=hvd.rank(),
            global_size=hvd.size(),
            prefetch=20,
            label_raw_type=dtype[metadata['label_raw_type']],
            dense_raw_type=dtype[metadata['dense_raw_type']],
            category_raw_type=dtype[metadata['category_raw_type']],
            log=metadata['dense_log'],
        )
        test_dataset = BinaryDataset(
            os.path.join(dataset_dir, 'test/label.bin'),
            os.path.join(dataset_dir, 'test/dense.bin'),
            os.path.join(dataset_dir, 'test/category.bin'),
            batch_size=global_batch_size // hvd.size(),
            drop_last=False,
            global_rank=hvd.rank(),
            global_size=hvd.size(),
            prefetch=20,
            label_raw_type=dtype[metadata['label_raw_type']],
            dense_raw_type=dtype[metadata['dense_raw_type']],
            category_raw_type=dtype[metadata['category_raw_type']],
            log=metadata['dense_log'],
        )

    trainer = Trainer(
        model,
        dataset,
        test_dataset,
        auc_thresholds=8000,
        base_lr=24.0,
        warmup_steps=args.lr_schedule_steps[0],
        decay_start_step=args.lr_schedule_steps[1],
        decay_steps=args.lr_schedule_steps[2],
        amp=args.amp,
    )

    if args.eval_in_last:
        trainer.train(eval_interval=None, eval_in_last=True, early_stop=args.early_stop)
    else:
        trainer.train(eval_in_last=False, early_stop=args.early_stop)

    print('main time: %.2fs'%(time.time() - start_time))
