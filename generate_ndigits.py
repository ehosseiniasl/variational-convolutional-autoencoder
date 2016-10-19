
import sys
import os
from conv_deconv_vae_multi import *




if __name__ == "__main__":
    do_fit, do_generate = False, False
    task = sys.argv[1]
    if task == 'fit':
        do_fit = True
    elif task == 'generate':
        n_digit = int(sys.argv[2])
        print 'generated %d-d numbers' % n_digit
        do_generate = True
        generate_root = './generate_%dd_mnist' % n_digit
        if not os.path.isdir(generate_root):
            os.mkdir(generate_root)

    tr, _, _, = mnist()
    trX, trY = tr
    tf = ConvVAE(image_save_root="./2d_images",
                 snapshot_file="./snapshots/mnist_snapshot.pkl")
    trX = floatX(trX)

    if do_fit:
        tf.fit(trX)
    elif do_generate:
        tf._setup_functions(trX)
        tf.generate_nd(trX, generate_root, n_digit)
    recs = tf.transform(trX[:100])
