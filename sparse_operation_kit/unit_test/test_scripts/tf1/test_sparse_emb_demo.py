"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../")))
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sparse_operation_kit as sok
import tensorflow as tf
import utils
from sparse_models import SOKDemo, TFDemo, create_SOKDemo
from test_dense_emb_demo import check_saved_embedding_variables
import strategy_wrapper
import numpy as np


def get_sok_results(args, init_tensors, *random_samples):
    if args.distributed_tool == "onedevice":
        strategy = strategy_wrapper.OneDeviceStrategy()
    elif args.distributed_tool == "horovod":
        import horovod.tensorflow as hvd
        hvd.init()
        strategy = strategy_wrapper.HorovodStrategy()
    else:
        raise ValueError(f"{args.distributed_tool} is not supported.")

    with strategy.scope():
        sok_init_op = sok.Init(global_batch_size=args.global_batch_size)

        embedding_initializer = tf.keras.initializers.Ones() if args.use_tf_initializer else None

        if not args.functional_api:
            sok_sparse_demo = SOKDemo(max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                    embedding_vec_size=args.embedding_vec_size,
                                    combiner=args.combiner,
                                    slot_num=args.slot_num,
                                    max_nnz=args.max_nnz,
                                    use_hashtable=args.use_hashtable,
                                    num_of_dense_layers=0,
                                    key_dtype=args.key_dtype,
                                    embedding_initializer=embedding_initializer)
        else:
            sok_sparse_demo = create_SOKDemo(combiner=args.combiner,
                                             max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                             embedding_vec_size=args.embedding_vec_size[0],
                                             slot_num=args.slot_num[0],
                                             max_nnz=args.max_nnz,
                                             use_hashtable=args.use_hashtable)
        
        emb_opt = utils.get_embedding_optimizer(args.optimizer)(learning_rate=0.1)
        dense_opt = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)
        if args.mixed_precision:
            emb_opt = sok.tf.keras.mixed_precision.LossScaleOptimizer(emb_opt, 1024)

    sok_saver = sok.Saver()
    restore_op = list()
    for i, embedding_layer in enumerate(sok_sparse_demo.embedding_layers):
        control_inputs = [restore_op[-1]] if restore_op else None
        with tf.control_dependencies(control_inputs):
            if args.restore_params:
                filepath = r"./embedding_variables"
                op = sok_saver.restore_from_file(embedding_layer.embedding_variable, filepath)
            else:
                if not args.use_tf_initializer:
                    op = sok_saver.load_embedding_values(embedding_layer.embedding_variable, init_tensors[i])
                else:
                    op = tf.constant(1.0)
            restore_op.append(op)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        _dtype = loss.dtype
        loss = tf.cast(loss, tf.float32)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)
        return tf.cast(loss, _dtype)

    def _train_step(inputs, labels, training):
        def _step_fn(inputs, labels):
            logit, embedding_vector = sok_sparse_demo(inputs, training=training)
            loss = _replica_loss(labels, logit)
            if args.mixed_precision:
                _loss = emb_opt.get_scaled_loss(loss)
            else:
                _loss = loss
            emb_var, other_var = sok.split_embedding_variable_from_others(sok_sparse_demo.trainable_variables)
            grads = tf.gradients(_loss, emb_var + other_var, colocate_gradients_with_ops=True,
                                 unconnected_gradients=tf.UnconnectedGradients.NONE)
            emb_grads, other_grads = grads[:len(emb_var)], grads[len(emb_var):]
            if args.mixed_precision:
                other_grads = emb_opt.get_unscaled_gradients(other_grads)
                emb_grads = emb_opt.get_unscaled_gradients(emb_grads)
            if "plugin" in args.optimizer:
                emb_train_op = emb_opt.apply_gradients(zip(emb_grads, emb_var))
            else:
                with sok.OptimizerScope(emb_var):
                    emb_train_op = emb_opt.apply_gradients(zip(emb_grads, emb_var))
            with tf.control_dependencies([*emb_grads]):
                # in case NCCL runs concurrently via SOK and horovod
                other_grads = strategy.reduce("sum", other_grads)
            other_train_op = dense_opt.apply_gradients(zip(other_grads, other_var))

            with tf.control_dependencies([emb_train_op, other_train_op]):
                total_loss = strategy.reduce("sum", loss)
                total_loss = tf.identity(total_loss)
                return total_loss, embedding_vector
        return strategy.run(_step_fn, inputs, labels)

    replica_batch_size = args.global_batch_size // args.gpu_num
    dataset = utils.tf_dataset(*random_samples, batchsize=replica_batch_size,
                               to_sparse_tensor=True, repeat=1, args=args)
    train_iterator = dataset.make_initializable_iterator()
    iterator_init = train_iterator.initializer

    inputs, labels = train_iterator.get_next()
    graph_results = _train_step(inputs, labels, training=True)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    if "plugin" in args.optimizer:
        init_op = tf.group(init_op, emb_opt.initializer)

    save_op = list()
    for i, embedding_layer in enumerate(sok_sparse_demo.embedding_layers):
        control_inputs = [save_op[-1]] if save_op else None
        with tf.control_dependencies(control_inputs):
            if args.save_params:
                filepath = r"./embedding_variables/"
                utils.try_make_dirs(filepath)
                op = sok_saver.dump_to_file(embedding_layer.embedding_variable, filepath)
            else:
                op = tf.constant(1.0)
            save_op.append(op)

    sok_results = list()

    with tf.Session() as sess:
        sess.run(sok_init_op)
        sess.run([init_op, iterator_init])
        sess.run(restore_op)
        sess.graph.finalize()

        for step in range(args.iter_num):
            loss_v, emb_vector_v = sess.run([*graph_results])
            print("*" * 80)
            print(f"Step: {step}, loss: {loss_v}")#", embedding_vector:\n{emb_vector_v}")
            sok_results.append(emb_vector_v)

        sess.run(save_op)

    name = list()
    for embedding_layer in sok_sparse_demo.embedding_layers:
        name.append(embedding_layer.embedding_variable.m_var_name)
    
    return sok_results, name

def get_tf_results(args, init_tensors, *random_samples):
    graph = tf.Graph()
    with graph.as_default():
        tf_sparse_demo = TFDemo(vocabulary_size=args.max_vocabulary_size_per_gpu * args.gpu_num,
                                embedding_vec_size=args.embedding_vec_size,
                                combiner=args.combiner,
                                slot_num=args.slot_num,
                                max_nnz=args.max_nnz,
                                use_hashtable=args.use_hashtable,
                                num_of_dense_layers=0)
        
        optimizer = utils.get_dense_optimizer(args.optimizer)(learning_rate=0.1)
        if args.mixed_precision:
            optimizer = sok.tf.keras.mixed_precision.LossScaleOptimizer(optimizer, 1024)

        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        def _train_step(inputs, labels, training):
            logit, embedding_vector = tf_sparse_demo(inputs, training=training)
            loss = loss_fn(labels, logit)
            if args.mixed_precision:
                _loss = optimizer.get_scaled_loss(loss)
            else:
                _loss = loss
            grads = tf.gradients(_loss, tf_sparse_demo.trainable_variables,
                                 colocate_gradients_with_ops=True,
                                 unconnected_gradients=tf.UnconnectedGradients.NONE)
            if args.mixed_precision:
                grads = optimizer.get_unscaled_gradients(grads)
            train_op = optimizer.apply_gradients(zip(grads, tf_sparse_demo.trainable_variables))
            with tf.control_dependencies([train_op]):
                loss = tf.identity(loss)
                return loss, embedding_vector


        dataset = utils.tf_dataset(*random_samples, batchsize=args.global_batch_size,
                                   to_sparse_tensor=True, repeat=1)
        train_iterator = dataset.make_initializable_iterator()
        iterator_init = train_iterator.initializer

        inputs, labels = train_iterator.get_next()
        graph_results = _train_step(inputs, labels, training=True)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        restore_op = list()
        for i, embedding_weight in enumerate(tf_sparse_demo.embedding_weights):
            restore_op.append(embedding_weight.assign(tf.concat(init_tensors[i], axis=0)))

        emb_values = list()
        for embedding_weight in tf_sparse_demo.embedding_weights:
            if args.save_params:
                filepath = r"./embedding_variables/"
                utils.try_make_dirs(filepath)
                emb_values.append(embedding_weight.read_value())
            else:
                emb_values = tf.constant(1.0)

    tf_results = list()
    with tf.Session(graph=graph) as sess:
        sess.run([init_op, iterator_init])
        sess.run(restore_op)
        sess.graph.finalize()

        for step in range(args.iter_num):
            loss_v, emb_vector_v = sess.run([*graph_results])
            print("*" * 80)
            print(f"step: {step}, loss: {loss_v}")#", embedding_vector:\n{emb_vector_v}")
            tf_results.append(emb_vector_v)

        emb_values_v = sess.run(emb_values)
        if args.save_params:
            for i, value in enumerate(emb_values_v):
                utils.save_to_file(os.path.join(filepath, r"tf_variable_" + str(i) + r".file"),
                                    value)
        
    name = list()
    for embedding_weight in tf_sparse_demo.embedding_weights:
        name.append(embedding_weight.name)

    return tf_results, name

def compare_sparse_emb_sok_with_tf(args):
    if args.global_batch_size % args.gpu_num != 0:
        raise ValueError(f"global_batch_size: {args.global_batch_size} is not divisible "
                         f"by gpu_num: {args.gpu_num}")

    if args.use_hashtable:
        vocabulary_size = args.max_vocabulary_size_per_gpu * args.gpu_num
    else:
        vocabulary_size = args.max_vocabulary_size_per_gpu

    if args.generate_new_datas:
        replica_batch_size = args.global_batch_size // args.gpu_num
        random_samples = utils.generate_random_samples(num_of_samples=replica_batch_size * args.iter_num,
                                                       vocabulary_size=vocabulary_size,
                                                       slot_num=sum(args.slot_num),
                                                       max_nnz=args.max_nnz,
                                                       use_sparse_mask=True)
        utils.save_to_file(r"./random_samples_" + str(args.rank_idx) + r".file", *random_samples)
    else:
        random_samples = utils.restore_from_file(r"./random_samples_" + str(args.rank_idx) + r".file")

    if args.restore_params:
        filepath = r"./embedding_variables"
        # because we already checked the variable consistency when saving
        # so that we can directly use TF Variable file to initialize
        # TF's Variable and SOK's Variable
        init_tensors = list()
        for i in range(len(args.slot_num)):
            tf_values_filename = os.path.join(filepath, r"tf_variable_" + str(i) + r".file")
            init_tensors.append(utils.restore_from_file(tf_values_filename))
    else:
        init_tensors = list()
        for i in range(len(args.slot_num)):
            init_tensors.append(utils.get_ones_tensor(max_vocab_size_per_gpu=args.max_vocabulary_size_per_gpu,
                                                      embedding_vec_size=args.embedding_vec_size[i],
                                                      num=args.gpu_num))
    sok_results, variable_names = get_sok_results(args, init_tensors, *random_samples)
    utils.save_to_file(r"./sok_embedding_vectors_" + str(args.rank_idx) + r".file", *sok_results)

    if args.rank_idx != 0:
        return

    # aggregate dataset from different worker
    dataset_filenames = [r"./random_samples_" + str(rank_idx) + r".file"
                         for rank_idx in range(args.rank_size)]
    random_samples_total = [list() for _ in range(args.iter_num)]
    random_labels_total = [list() for _ in range(args.iter_num)]
    local_batch_size = args.global_batch_size // args.gpu_num
    for rank_idx in range(args.rank_size):
        samples, labels = utils.restore_from_file(dataset_filenames[rank_idx])
        for i in range(args.iter_num):
            random_samples_total[i].extend(samples[i * local_batch_size : (i + 1) * local_batch_size])
            random_labels_total[i].extend(labels[i * local_batch_size : (i + 1) * local_batch_size])
    random_samples_total = np.concatenate(random_samples_total, axis=0)
    random_labels_total = np.concatenate(random_labels_total, axis=0)

    tf_results, _ = get_tf_results(args, init_tensors, random_samples_total, random_labels_total)

    # aggregate sok forward results from different worker
    sok_results_filenames = [r"./sok_embedding_vectors_" + str(rank_idx) + r".file"
                             for rank_idx in range(args.rank_size)]
    sok_results_total = list()
    for filename in sok_results_filenames:
        sok_results = utils.restore_from_file(filename)
        sok_results_total.append(sok_results)

    if len(sok_results_total[0]) != len(tf_results):
        raise ValueError("The length of sok results is not equal to that of tensorflow.")
    if len(sok_results) != args.iter_num:
        raise ValueError("The length of embedding vectors: %d is not equal to iteration number: %d."
                        %(len(sok_results), args.iter_num))

    rtol, atol = 1e-3, 1e-3
    if args.restore_params:
        rtol, atol = rtol * 10, atol * 10
    elif args.distributed_tool == "horovod":
        rtol, atol = rtol * 10, atol * 10
    elif args.mixed_precision:
        rtol, atol = 1e-2, 1e-2

    for i in range(args.iter_num):
        sok_vector = np.concatenate([sok_results_total[rank_idx][i]
                                     for rank_idx in range(args.rank_size)], axis=0)
        allclose = np.allclose(sok_vector, tf_results[i], rtol=rtol, atol=atol)
        if not allclose:
            raise ValueError(f"\n{sok_vector} \nis not near to \n{tf_results[i]} \nat rtol={rtol}, atol={atol}")

    print(f"\n[INFO]: For {len(args.slot_num)} Sparse Embedding layer, using {args.gpu_num} GPUs + {args.optimizer} optimizer, "
          f"using hashtable? {args.use_hashtable}, combiner = {args.combiner}, the embedding vectors"
          f" obtained from sok and tf are consistent for {args.iter_num} iterations, "
          f"with mixed_precision = {args.mixed_precision}, key_dtype = {args.key_dtype}"
          f" use_tf_initializer = {args.use_tf_initializer}")

    if args.save_params:
        check_saved_embedding_variables(args, variable_names,
                                        use_hashtable=args.use_hashtable, 
                                        gpu_num=args.gpu_num,
                                        atol=atol, rtol=rtol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_num", type=int, required=False, default=1)
    parser.add_argument("--distributed_tool", type=str, required=False, 
                        choices=["horovod", "onedevice"], default="onedevice")
    parser.add_argument("--iter_num", type=int, required=False, default=50)
    parser.add_argument("--max_vocabulary_size_per_gpu", type=int,
                        required=False, default=1024)
    parser.add_argument("--combiner", type=str, required=False, default="sum",
                        choices=["sum", "mean"])
    parser.add_argument("--slot_num", type=int, nargs="+",
                        help="the number of feature fileds",
                        required=False, default=1)
    parser.add_argument("--max_nnz", type=int,
                        help="the maximum of valid inputs",
                        required=False, default=1)
    parser.add_argument("--embedding_vec_size", type=int, nargs="+",
                        required=False, default=1)
    parser.add_argument("--global_batch_size", type=int, required=False,
                        default=16)
    parser.add_argument("--optimizer", type=str, required=False, 
                        default="adam", choices=["plugin_adam", "adam", "sgd", "compat_adam"])
    parser.add_argument("--generate_new_datas", type=int, choices=[0, 1],
                        required=False, default=1)
    parser.add_argument("--save_params", type=int, choices=[0, 1],
                        required=False, default=1)
    parser.add_argument("--restore_params", type=int, choices=[0, 1],
                        required=False, default=0)
    parser.add_argument("--use_hashtable", type=int, choices=[0, 1],
                        required=False, default=1)
    parser.add_argument("--mixed_precision", type=int, choices=[0, 1],
                        required=False, default=0)
    parser.add_argument("--key_dtype", type=str, choices=["int64", "uint32"], default="int64")
    parser.add_argument("--use_tf_initializer", type=int, choices=[0, 1], default=0)
    parser.add_argument("--functional_api", type=int, choices=[0, 1], default=0)

    args = parser.parse_args()

    args.generate_new_datas = True if args.generate_new_datas == 1 else False
    args.save_params = True if args.save_params == 1 else False
    args.restore_params = True if args.restore_params == 1 else False
    args.use_hashtable = True if args.use_hashtable == 1 else False
    args.mixed_precision = True if args.mixed_precision == 1 else False
    args.use_tf_initializer = True if args.use_tf_initializer == 1 else False
    args.functional_api = True if args.functional_api == 1 else False

    if (args.distributed_tool == "onedevice" and args.gpu_num != 1):
        raise ValueError(f"When 'onedevice' is used as the distributed_tool, "
                         f"gpu_num must be 1, which is {args.gpu_num}")

    if args.distributed_tool == "onedevice":
        available_gpus = ",".join(map(str, range(args.gpu_num)))
        rank_size = args.gpu_num
        rank_idx = 0
    else:
        # gpu_num will be ignored.
        rank_size = os.getenv("OMPI_COMM_WORLD_SIZE")
        if rank_size is None:
            raise ValueError(f"When distributed_tool is set to {args.distributed_tool}, "
                             "mpiexec / mpirun must be used to launch this program.")
        rank_size = int(rank_size)
        rank_idx = int(os.getenv("OMPI_COMM_WORLD_RANK"))

        available_gpus = str(rank_idx)

    os.environ["CUDA_VISIBLE_DEVICES"] = available_gpus

    args.rank_size = rank_size
    args.rank_idx = rank_idx
    args.gpu_num = rank_size

    if args.mixed_precision:
        from tensorflow.python.keras.engine import base_layer_utils
        base_layer_utils.enable_v2_dtype_behavior()
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)

    compare_sparse_emb_sok_with_tf(args)