import time

import sparse_operation_kit as sok
import horovod.tensorflow as hvd
import tensorflow as tf
import nvtx


def evaluate(model, dataset, thresholds):
    auc = tf.keras.metrics.AUC(num_thresholds=thresholds,
                               curve='ROC',
                               summation_method='interpolation',
                               from_logits=True)

    @tf.function
    def _step(samples, labels):
        probs = model(samples, training=False)
        auc.update_state(labels, probs)

    for idx, (samples, labels) in enumerate(dataset):
        _step(samples, labels)

    auc.true_positives.assign(hvd.allreduce(
        auc.true_positives, name='true_positives', op=hvd.mpi_ops.Sum))
    auc.true_negatives.assign(hvd.allreduce(
        auc.true_negatives, name='true_negatives', op=hvd.mpi_ops.Sum))
    auc.false_positives.assign(hvd.allreduce(
        auc.false_positives, name='false_positives', op=hvd.mpi_ops.Sum))
    auc.false_negatives.assign(hvd.allreduce(
        auc.false_negatives, name='false_negatives', op=hvd.mpi_ops.Sum))

    return auc.result().numpy()


def evaluate_wilcoxon(model, dataset):
    @tf.function
    def _step(samples, labels):
        probs = model(samples, training=False)
        return tf.concat([probs, labels], axis=1)

    results = []
    for idx, (samples, labels) in enumerate(dataset):
        result = _step(samples, labels)
        results.append(result)
    results = tf.concat(results, axis=0)

    results = hvd.allgather(results, name='wilcoxon_AUC')

    sort_order = tf.argsort(results[:, 0])
    sorted_label = tf.gather(results[:, 1], sort_order)
    rank = tf.cast(tf.range(1, sorted_label.shape[0]+1), tf.float32)
    num_true = tf.reduce_sum(sorted_label)
    num_false = sorted_label.shape[0] - num_true
    auc = (tf.reduce_sum(rank * sorted_label) - (num_true * (num_true + 1) / 2)) / (num_true * num_false)
    return auc.numpy()


class LearningRateScheduler:
    """
    LR Scheduler combining Polynomial Decay with Warmup at the beginning.
    TF-based cond operations necessary for performance in graph mode.
    """

    def __init__(self, optimizers, base_lr, warmup_steps, decay_start_step, decay_steps):
        self.optimizers = optimizers
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self.decay_start_step = tf.constant(decay_start_step, dtype=tf.int32)
        self.decay_steps = tf.constant(decay_steps)
        self.decay_end_step = decay_start_step + decay_steps
        self.poly_power = 2
        self.base_lr = base_lr
        with tf.device('/CPU:0'):
            self.step = tf.Variable(0)

    @tf.function
    def __call__(self):
        with tf.device('/CPU:0'):
            # used for the warmup stage
            warmup_step = tf.cast(1 / self.warmup_steps, tf.float32)
            lr_factor_warmup = 1 - tf.cast(self.warmup_steps - self.step, tf.float32) * warmup_step
            lr_factor_warmup = tf.cast(lr_factor_warmup, tf.float32)

            # used for the constant stage
            lr_factor_constant = tf.cast(1., tf.float32)

            # used for the decay stage
            lr_factor_decay = (self.decay_end_step - self.step) / self.decay_steps
            lr_factor_decay = tf.math.pow(lr_factor_decay, self.poly_power)
            lr_factor_decay = tf.cast(lr_factor_decay, tf.float32)

            poly_schedule = tf.cond(self.step < self.decay_start_step, lambda: lr_factor_constant,
                                    lambda: lr_factor_decay)

            lr_factor = tf.cond(self.step < self.warmup_steps, lambda: lr_factor_warmup,
                                lambda: poly_schedule)

            lr = self.base_lr * lr_factor
            for optimizer in self.optimizers:
                optimizer.lr.assign(lr)

            self.step.assign(self.step + 1)


def scale_grad(grad, factor):
    if isinstance(grad, tf.IndexedSlices):
        # sparse gradient
        grad._values = grad._values * factor
        return grad
    else:
        # dense gradient
        return grad * factor


class Trainer:

    def __init__(
        self,
        model,
        dataset,
        test_dataset,
        auc_thresholds,
        base_lr,
        warmup_steps,
        decay_start_step,
        decay_steps,
        amp,
    ):
        base_lr = float(base_lr)

        self._model = model
        # self._embedding_vars, self._dense_vars = \
        #     sok.split_embedding_variable_from_others(self._model.trainable_variables)
        self._dataset = dataset
        self._test_dataset = test_dataset
        self._auc_thresholds = auc_thresholds
        self._amp = amp

        self._loss_fn = tf.losses.BinaryCrossentropy(from_logits=True)

        self._dense_optimizer = tf.keras.optimizers.SGD(base_lr)
        self._embedding_optimizer = tf.keras.optimizers.SGD(base_lr)
        if self._amp:
            self._embedding_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                self._embedding_optimizer, initial_scale=1024, dynamic=False)
        self._lr_scheduler = LearningRateScheduler(
            [self._dense_optimizer, self._embedding_optimizer],
            base_lr,
            warmup_steps,
            decay_start_step,
            decay_steps,
        )

    @tf.function
    def _step(self, samples, labels, first_batch):
        self._lr_scheduler()

        with tf.GradientTape() as tape:
            probs = self._model(samples, training=True)
            loss = self._loss_fn(labels, probs)
            if self._amp:
                loss = self._embedding_optimizer.get_scaled_loss(loss)

        embedding_vars, dense_vars = sok.split_embedding_variable_from_others(self._model.trainable_variables)
        embedding_grads, dense_grads = tape.gradient(loss, [embedding_vars, dense_vars])
        if self._amp:
            embedding_grads = self._embedding_optimizer.get_unscaled_gradients(embedding_grads)
            dense_grads = self._embedding_optimizer.get_unscaled_gradients(dense_grads)

        # embedding_grads = [scale_grad(g, hvd.size()) for g in embedding_grads]

        with sok.OptimizerScope(embedding_vars):
            self._embedding_optimizer.apply_gradients(zip(embedding_grads, embedding_vars),
                                                      experimental_aggregate_gradients=False)

        # with tf.control_dependencies(embedding_grads):
        dense_grads = [hvd.allreduce(grad, op=hvd.Average, compression=hvd.compression.NoneCompressor) for grad in dense_grads]
        self._dense_optimizer.apply_gradients(zip(dense_grads, dense_vars),
                                              experimental_aggregate_gradients=False)

        if first_batch:
            hvd.broadcast_variables(dense_vars, root_rank=0)
            hvd.broadcast_variables(self._dense_optimizer.variables(), root_rank=0)

        return loss

    def train(self, interval=1000, eval_interval=3793, eval_in_last=False, early_stop=-1):
        eval_time = 0
        iter_time = time.time()
        total_time = time.time()
        throughputs = []
        for idx, (samples, labels) in enumerate(self._dataset):
            # rng = nvtx.start_range(message='Iteration_'+str(idx), color='blue')
            loss = self._step(samples, labels, idx == 0)
            # nvtx.end_range(rng)

            if (idx % interval == 0) and (idx > 0):
                t = time.time() - iter_time
                throughput = interval * self._dataset._batch_size * hvd.size() / t
                print('Iteration:%d\tloss:%.6f\ttime:%.2fs\tthroughput:%.2fM'%(idx, loss, t, throughput / 1000000))
                throughputs.append(throughput)
                iter_time = time.time()

            if (eval_interval is not None) and (idx % eval_interval == 0) and (idx > 0):
                t = time.time()
                auc = evaluate(self._model, self._test_dataset, self._auc_thresholds)
                t = time.time() - t
                eval_time += t
                iter_time += t
                print('Evaluate in %dth iteration, test time: %.2fs, AUC: %.6f.'%(idx, t, auc))
                if auc > 0.8025:
                    break

            if early_stop > 0 and (idx + 1) >= early_stop:
                break

        if eval_in_last:
            t = time.time()
            auc = evaluate(self._model, self._test_dataset, self._auc_thresholds)
            t = time.time() - t
            eval_time += t
            print('Evaluate in the end, test time: %.2fs, AUC: %.6f.'%(t, auc))

        total_time = time.time() - total_time
        training_time = total_time - eval_time
        avg_training_time = training_time / (idx + 1)
        print('total time: %.2fs, in %d iterations'%(total_time, (idx + 1)))
        average_throughput = sum(throughputs[1:]) / len(throughputs[1:])
        print('only training time: %.2fs, average: %.2fms/iter, average throughput: %.2fM(%.2fms/iter)'%(
            training_time, avg_training_time * 1000, average_throughput / 1000000,
            self._dataset._batch_size * hvd.size() / average_throughput * 1000))
        print('only evaluate time: %.2fs'%(eval_time))

