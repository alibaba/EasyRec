import os
import argparse
import time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    os.makedirs(args.output)
    size = os.path.getsize(args.input)
    assert(size % 160 == 0)
    num_samples = size // 160

    chunk_size = 1024 * 1024

    inp_f = open(args.input, 'rb')

    label_f = open(os.path.join(args.output, 'label.bin'), 'wb')
    dense_f = open(os.path.join(args.output, 'dense.bin'), 'wb')
    category_f = open(os.path.join(args.output, 'category.bin'), 'wb')

    num_loops = num_samples // chunk_size + 1
    start_time = time.time()
    for i in range(num_loops):
        t = time.time()

        if i == (num_loops - 1):
            batch = min(chunk_size, num_samples % chunk_size)
            if batch == 0:
                break
        else:
            batch = chunk_size

        raw_buffer = inp_f.read(160 * batch)
        for j in range(batch):
            label_buffer = raw_buffer[j*160: j*160+4]
            dense_buffer = raw_buffer[j*160+4: j*160+56]
            category_buffer = raw_buffer[j*160+56: j*160+160]

            label_f.write(label_buffer)
            dense_f.write(dense_buffer)
            category_f.write(category_buffer)

        print('%d/%d batch finished. write %d samples, time: %.2fms, remaining time: %.2f min'%(
            i+1, num_loops, batch, (time.time() - t)*1000, ((time.time() - start_time) / 60) * (num_loops / (i+1) - 1)))

    inp_f.close()
    label_f.close()
    dense_f.close()
    category_f.close()

