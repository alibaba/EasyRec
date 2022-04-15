import os
import argparse
import json
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('num_iters', type=int)
    args = parser.parse_args()

    os.system('rm -rf %s'%args.output_dir)
    os.makedirs(args.output_dir)

    # Big embedding table, ~80GB
    vocab_sizes = [
        39884406,   39043,      17289,      7420,       20263,     3,
        7120,       1543,       63,         38532951,   2953546,   403346,
        10,         2208,       11938,      155,        4,         976,
        14,         39979771,   25641295,   39664984,   585935,    12972,
        108,        36
    ]

    label = np.random.randint(0, 2, args.batch_size).astype(np.int32)
    dense = np.random.rand(args.batch_size * 13).astype(np.float32)
    category = []
    for size in vocab_sizes:
        category.append(np.random.randint(0, size, args.batch_size).reshape([-1, 1]).astype(np.int32))
    category = np.concatenate(category, axis=1).reshape([-1])

    for file, tensor in [['label.bin', label], ['dense.bin', dense], ['category.bin', category]]:
        with open(os.path.join(args.output_dir, file), 'wb') as f:
            raw_bytes = tensor.tobytes()
            for _ in range(args.num_iters):
                f.write(raw_bytes)

    train_dir = os.path.join(args.output_dir, 'train')
    test_dir = os.path.join(args.output_dir, 'test')
    os.system('mkdir %s'%train_dir)
    os.system('mkdir %s'%test_dir)

    bin_files = os.path.join(args.output_dir, '*.bin')
    os.system('cp %s %s'%(bin_files, train_dir))
    os.system('mv %s %s'%(bin_files, test_dir))

    metadata = {
        'vocab_sizes': vocab_sizes,
        'label_raw_type': 'int32',
        'dense_raw_type': 'float32',
        'category_raw_type': 'int32',
        'dense_log': False
    }
    with open(os.path.join(train_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
