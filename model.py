from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple

from module import *
from utils import *


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        self.generator_conv = generator_conv
        self.generator_deconv = generator_deconv
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_x_and_B_images')

        self.real_x = self.real_data[:, :, :, :self.input_c_dim]
        self.real_y = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.embedding_x = self.generator_conv(self.real_x, self.options, False, name="generatorX2Y")   # Linked to embedding_gx
        self.embedding_y = self.generator_conv(self.real_y, self.options, False, name="generatorY2X")   # Linked to embedding_fy
        self.g_x = self.generator_deconv(self.embedding_x, self.options, False, name="generatorX2Y")    # G(x) = real_y
        self.f_y = self.generator_deconv(self.embedding_y, self.options, False, name="generatorY2X")    # F(y) = real_x
        self.embedding_gx = self.generator_conv(self.g_X, self.options, False, name="generatorX2Y")     # Linked to embedding_x
        self.embedding_fy = self.generator_conv(self.f_y, self.options, False, name="generatorY2X")     # Linked to embedding_y
        self.f_gx = self.generator_deconv(self.embedding_gx, self.options, False, name="generatorX2Y")  # F(G(x)) = real_x
        self.g_fy = self.generator.deconv(self.embedding_fy, self.options, False, name="generatorY2X")  # G(F(x)) = real_y

        #self.g_x = self.generator(self.real_x, self.options, False, name="generatorX2Y")  # G(X)
        #self.f_y_ = self.generator(self.g_x, self.options, False, name="generatorY2X") # F(G(X))
        #self.f_y = self.generator(self.real_y, self.options, True, name="generatorY2X")   # F(Y)
        #self.g_x_ = self.generator(self.f_y, self.options, True, name="generatorX2Y")  # G(F(Y))

        self.d_x = self.discriminator(self.g_x, self.options, reuse=False, name="discriminator_x")
        self.d_y = self.discriminator(self.f_y, self.options, reuse=False, name="discriminator_y")
        self.g_loss_x2y = self.criterionGAN(self.d_x, tf.ones_like(self.d_x)) \
                          + self.L1_lambda * abs_criterion(self.real_x, self.f_gx) \
                          + self.L1_lambda * abs_criterion(self.real_y, self.g_fy) \ 
                          + self.L1_lambda * abs_criterion(self.embedding_gx, self.embedding_x) \
                          + self.L1_lambda * abs_criterion(self.embedding_fy, self.embedding_y)
                          
        self.g_loss_y2x = self.criterionGAN(self.d_y, tf.ones_like(self.d_y)) \
                          + self.L1_lambda * abs_criterion(self.real_x, self.f_gx) \
                          + self.L1_lambda * abs_criterion(self.real_y, self.g_fy) \
                          + self.L1_lambda * abs_criterion(self.embedding_gx, self.embedding_x) \
                          + self.L1_lambda * abs_criterion(self.embedding_fy, self.embedding_y)

        self.fake_x_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_x_sample')
        self.fake_y_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_y_sample')
        self.d_y_real = self.discriminator(self.real_y, self.options, reuse=True, name="discriminator_y")
        self.d_x_real = self.discriminator(self.real_x, self.options, reuse=True, name="discriminator_y")
        self.d_x_sample = self.discriminator(self.fake_y_sample, self.options, reuse=True, name="discriminator_y")
        self.d_y_sample = self.discriminator(self.fake_x_sample, self.options, reuse=True, name="discriminator_x")
        self.dy_loss_real = self.criterionGAN(self.d_y_real, tf.ones_like(self.d_y_real))
        self.dy_loss_fake = self.criterionGAN(self.d_x_sample, tf.zeros_like(self.d_x_sample))
        self.dy_loss = (self.dy_loss_real + self.dy_loss_fake) / 2
        self.dx_loss_real = self.criterionGAN(self.d_x_real, tf.ones_like(self.d_x_real))
        self.dx_loss_fake = self.criterionGAN(self.d_y_sample, tf.zeros_like(self.d_y_sample))
        self.dx_loss = (self.dx_loss_real + self.dx_loss_fake) / 2

        self.g_x2y_sum = tf.summary.scalar("g_loss_x2y", self.g_loss_x2y)
        self.g_x2y_sum = tf.summary.scalar("g_loss_y2x", self.g_loss_y2x)
        self.dy_loss_sum = tf.summary.scalar("dy_loss", self.dy_loss)
        self.dx_loss_sum = tf.summary.scalar("dx_loss", self.dx_loss)
        self.dy_loss_real_sum = tf.summary.scalar("dy_loss_real", self.dy_loss_real)
        self.dy_loss_fake_sum = tf.summary.scalar("dy_loss_fake", self.dy_loss_fake)
        self.dx_loss_real_sum = tf.summary.scalar("dx_loss_real", self.dx_loss_real)
        self.dx_loss_fake_sum = tf.summary.scalar("dx_loss_fake", self.dx_loss_fake)
        self.dy_sum = tf.summary.merge(
            [self.dy_loss_sum, self.dy_loss_real_sum, self.dy_loss_fake_sum]
        )
        self.dx_sum = tf.summary.merge(
            [self.dx_loss_sum, self.dx_loss_real_sum, self.dx_loss_fake_sum]
        )

        self.test_x = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_x')
        self.test_y = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_y')
        self.test_embedding_x = self.generator_conv(self.test_x, self.options, True, name="generatorX2Y")
        self.test_embedding_y = self.generator_conv(self.test_y, self.options, True, name="generatorY2X")
        self.test_y = self.generator_deconv(self.test_embedding_x, self.options, True, name="generatorX2Y")
        self.test_x = self.generator_deconv(self.test_embedding_y, self.options, True, name="generatorY2X")

        t_vars = tf.trainable_variables()
        self.dy_vars = [var for var in t_vars if 'discriminator_y' in var.name]
        self.dx_vars = [var for var in t_vars if 'discriminator_x' in var.name]
        self.g_vars_x2y = [var for var in t_vars if 'generatorX2Y' in var.name]
        self.g_vars_y2x = [var for var in t_vars if 'generatorY2X' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.dx_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.dx_loss, var_list=self.dx_vars)
        self.dy_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.dy_loss, var_list=self.dy_vars)
        self.g_x2y_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.g_loss_x2y, var_list=self.g_vars_x2y)
        self.g_y2x_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.g_loss_y2x, var_list=self.g_vars_y2x)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(args.epoch):
            dataX = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataY = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataX)
            np.random.shuffle(dataY)
            batch_idxs = min(min(len(dataX), len(dataY)), args.train_size) // self.batch_size

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataX[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataY[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_data(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                # Forward G network
                f_y, g_x = self.sess.run([self.f_y, self.g_x],
                                               feed_dict={self.real_data: batch_images})
                [f_y, g_x] = self.pool([f_y, g_x])
                # Update G network
                _, summary_str = self.sess.run([self.g_x2y_optim, self.g_x2y_sum],
                                               feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)
                # Update D network
                _, summary_str = self.sess.run([self.dy_optim, self.db_sum],
                                               feed_dict={self.real_data: batch_images,
                                                          self.fake_y_sample: g_x})
                self.writer.add_summary(summary_str, counter)
                # Update G network
                _, summary_str = self.sess.run([self.g_y2x_optim, self.g_x2y_sum],
                                               feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)
                # Update D network
                _, summary_str = self.sess.run([self.dx_optim, self.da_sum],
                                               feed_dict={self.real_data: batch_images,
                                                          self.fake_x_sample: f_y})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                       % (epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, 100) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 1000) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataX = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataY = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataX)
        np.random.shuffle(dataY)
        batch_files = list(zip(dataX[:self.batch_size], dataY[:self.batch_size]))
        sample_images = [load_data(batch_file, False, True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        f_y, g_x = self.sess.run(
            [self.f_y, self.g_x],
            feed_dict={self.real_data: sample_images}
        )
        save_images(f_y, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(g_x, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_x) if args.which_direction == 'AtoB' else (
            self.testA, self.test_y)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
            '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
            '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
