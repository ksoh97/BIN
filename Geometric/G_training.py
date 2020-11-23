import tensorflow as tf
import tensorflow.keras as keras
from Geometric import G_network as net, G_config as conf
from Geometric import G_losses, G_utils, G_test
import tqdm
import GPUtil
import os
import numpy as np

l = keras.layers
K = keras.backend

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = devices

file_name = conf.file_name
mode =conf.mode

if mode == 0:
    epoch = conf.epoch
    batch_size = conf.batch_size
    learning_rate = conf.lr
    learning_decay = conf.lr_decay

else:
    epoch = conf.epoch
    batch_size = conf.batch_size
    generator_step, discriminator_step = conf.g_step, conf.d_step
    learning_rate_g, learning_rate_d = conf.lr_g, conf.lr_d
    learning_decay = conf.lr_decay
    smooth = conf.one_sided_label_smoothing
    loss_type = conf.loss_type

class Trainer:
    def __init__(self):
        if mode == 0:
            self.save_path = (conf.save_path + file_name)
        else:
            self.save_path = (conf.save_path + file_name)

        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        self.valid_acc, self.compare_acc, self.valid_loss, self.count = 0, 0, 0, 0
        self.valid_save, self.model_select = False, False
        tf.keras.backend.set_image_data_format("channels_last")
        self.build_model()

    def build_model(self):
        if mode == 0:
            network = net.Geometric_network()
            self.train_vars = []
            self.enc_model, self.cls_model = network.pretraining_clf()
            self.train_vars += self.cls_model.trainable_variables

        elif mode == 1:
            g_network = net.Geometric_network()
            d_network = net.Discriminator()
            self.train_vars = []
            self.train_discri_vars = []

            self.discriminator_model = d_network.build_model()
            self.train_discri_vars += self.discriminator_model.trainable_variables

            # TODO: after training, fix the save path
            self.cls_model, self.dec_model = g_network.CFmap_generator()
            cls_load_weights = conf.cls_weight_path
            self.cls_model.load_weights(cls_load_weights)
            for layer in self.cls_model.layers:
                layer.trainable = False

            enc_load_weights = conf.enc_weight_path
            self.dec_model.load_weights(enc_load_weights)
            save_variables = False

            for variables in self.dec_model.trainable_variables:
                if "dec" in variables.name:
                    save_variables = True
                if save_variables:
                    self.train_vars += [variables]

    def cycle_consistency(self, pseudo_image, lbl):
        c1, c2, c3 = G_utils.codemap(target_c=lbl)
        tilde_map = self.dec_model({"enc_in": pseudo_image, "c1": c1, "c2": c2, "c3": c3}, training=True)["dec_out"]
        like_input = pseudo_image + tilde_map
        return like_input

    def _train_one_batch(self, train_dat, train_lbl, optim, disc_optim, train_vars, train_discri_vars, step, target_c):
        if mode == 0:
            with tf.GradientTape() as tape:
                res = self.cls_model({"All_in": train_dat}, training=True)["cls_out"]
                loss = G_losses.BCE_loss(train_lbl, res)

            grads = tape.gradient(loss, train_vars)
            optim.apply_gradients(zip(grads, train_vars))

            if step % 10 == 0:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar("mode0_train_loss", loss, step=step)

        elif mode == 1:
            c1, c2, c3 = G_utils.codemap(target_c=target_c)
            # Discriminator step
            if step % discriminator_step == 0 or step == 0:
                with tf.GradientTape() as tape:
                    CFmap = self.dec_model({"enc_in": train_dat, "c1": c1, "c2": c2, "c3": c3}, training=True)["dec_out"]
                    pseudo_image = train_dat + CFmap

                    real = self.discriminator_model({"discri_in": train_dat}, training=True)["discri_out"]
                    fake = self.discriminator_model({"discri_in": pseudo_image}, training=True)["discri_out"]
                    total_loss = loss_type["dis"] * G_losses.discriminator_loss(real, fake)

                grads = tape.gradient(total_loss, train_discri_vars)
                disc_optim.apply_gradients(zip(grads, train_discri_vars))

                if step % 10 == 0:
                    with self.discriminator_summary_writer.as_default():
                        tf.summary.scalar("discriminator_loss", total_loss, step=step)

            # Generator step
            if step % generator_step == 0 or step == 0:
                with tf.GradientTape() as tape:
                    CFmap = self.dec_model({"enc_in": train_dat, "c1": c1, "c2": c2, "c3": c3}, training=True)["dec_out"]
                    pseudo_image = train_dat + CFmap

                    like_input = self.cycle_consistency(pseudo_image, train_lbl)
                    discri_res = self.discriminator_model({"discri_in": pseudo_image}, training=False)["discri_out"]
                    res = self.cls_model({"cls_in": pseudo_image}, training=False)["cls_out"]

                    if smooth:
                        cls = loss_type["cls"] * G_losses.BCE_loss(G_losses.one_sided_label_smoothing(target_c), res)
                    else:
                        cls = loss_type["cls"] * G_losses.BCE_loss(target_c, res)

                    gan = loss_type["GAN"] * G_losses.generator_loss(target_c, discri_res)
                    cyc = loss_type["cyc"] * G_losses.cycle_loss(train_dat, like_input)
                    norm = loss_type["norm"] * G_losses.L1_norm(effect_map=CFmap)
                    total_loss = cls + gan + cyc + norm

                grads = tape.gradient(total_loss, train_vars)
                optim.apply_gradients(zip(grads, train_vars))

                if step % 10 == 0:
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar("generator_total_train_loss", total_loss, step=step)
                    with self.generator_summary_writer.as_default():
                        tf.summary.scalar("generator_cls_loss", cls, step=step)
                        tf.summary.scalar("generator_gan_loss", gan, step=step)
                        tf.summary.scalar("generator_cyc_loss", cyc, step=step)
                        tf.summary.scalar("generator_l1_loss", norm, step=step)

    def _valid_logger(self, valid_dat, valid_lbl, epoch, target_c):
        if mode == 0:
            res = self.cls_model({"All_in": valid_dat}, training=False)["cls_out"]

            valid_loss = G_losses.BCE_loss(valid_lbl, res)
            valid_acc = G_losses.acc(valid_lbl, res)

            self.valid_loss += valid_loss
            self.valid_acc += valid_acc
            self.count += 1

            if self.valid_save == True:
                self.valid_acc /= self.count
                self.valid_loss /= self.count

                if self.compare_acc <= self.valid_acc:
                    self.compare_acc = self.valid_acc
                    self.model_select = True
                    print("Valid ACC: %f" % self.compare_acc)

                with self.valid_summary_writer.as_default():
                    tf.summary.scalar("mode0_valid_loss", self.valid_loss, step=epoch)
                    tf.summary.scalar("mode0_valid_acc", self.valid_acc, step=epoch)
                    self.valid_acc, self.valid_loss, self.count = 0, 0, 0
                    self.valid_save = False

        elif mode == 1:
            c1, c2, c3 = G_utils.codemap(target_c=target_c)
            CFmap = self.dec_model({"enc_in": valid_dat, "c1": c1, "c2": c2, "c3": c3}, training=False)["dec_out"]
            pseudo_image = valid_dat + CFmap

            like_input = self.cycle_consistency(pseudo_image, valid_lbl)
            discri_res = self.discriminator_model({"discri_in": pseudo_image}, training=False)["discri_out"]
            res = self.cls_model({"effect_in": pseudo_image}, training=False)["cls_out"]

            if smooth:
                cls = loss_type["cls"] * G_losses.BCE_loss(G_losses.one_sided_label_smoothing(target_c), res)
            else:
                cls = loss_type["cls"] * G_losses.BCE_loss(target_c, res)

            gan = loss_type["GAN"] * G_losses.generator_loss(target_c, discri_res)
            cyc = loss_type["cyc"] * G_losses.cycle_loss(valid_dat, like_input)
            norm = loss_type["norm"] * G_losses.L1_norm(CFmap)

            valid_loss = cls + gan + cyc + norm
            valid_acc = G_losses.acc(target_c, res)
            self.valid_loss += valid_loss
            self.valid_acc += valid_acc
            self.count += 1

            if self.valid_save:
                self.valid_loss /= self.count
                self.valid_acc /= self.count
                print("Epoch:%03d, valid ACC: %f" % (epoch, self.valid_acc))

                if self.compare_acc <= self.valid_acc:
                    self.compare_acc = self.valid_acc
                    self.model_select = True

                elif self.valid_acc >= 0.60 or self.valid_loss <= 1.5:
                    self.model_select = True

                with self.valid_summary_writer.as_default():
                    tf.summary.scalar("mode1_valid_loss", self.valid_loss, step=epoch)
                    tf.summary.scalar("mode1_valid_acc", self.valid_acc, step=epoch)
                    self.valid_acc, self.valid_loss, self.count = 0, 0, 0
                self.valid_save = False

    def train(self):
        train_dat, train_lbl, valid_dat, valid_lbl, red_test_dat, orange_test_dat = G_utils.load_geometric_data()

        train_idx = np.argwhere(train_lbl >= 0)
        train_lbl = np.eye(2)[np.squeeze(train_lbl)]
        valid_lbl = np.eye(2)[np.squeeze(valid_lbl)]

        test_dat = np.concatenate((red_test_dat, orange_test_dat), axis=0)
        red_lbl, orange_lbl = np.eye(2)[np.squeeze(0)], np.eye(2)[np.squeeze(1)]
        red_lbl, orange_lbl = np.full((100, 2), red_lbl), np.full((100, 2), orange_lbl)  # 2400 / 2400
        test_lbl = np.concatenate((red_lbl, orange_lbl), axis=0)

        self.train_summary_writer = tf.summary.create_file_writer(self.save_path + "/train_mode%d" % mode)
        self.valid_summary_writer = tf.summary.create_file_writer(self.save_path + "/valid_mode%d" % mode)
        self.test_summary_writer = tf.summary.create_file_writer(self.save_path + "/test_mode%d" % mode)
        self.build_model()

        if mode == 0:
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=len(train_dat) // batch_size,
                                                                      decay_rate=learning_decay, staircase=True)
            optim = keras.optimizers.Adam(lr_schedule)
            global_step, self.compare_acc = 0, 0

            for cur_epoch in tqdm.trange(epoch):
                train_idx = np.squeeze(np.random.permutation(train_idx))

                # training
                for cur_step in tqdm.trange(0, len(train_idx), batch_size, desc="Training model: epoch%d" % cur_epoch):
                    idx = train_idx[cur_step:cur_step+batch_size]
                    trn_dat, trn_lbl = train_dat[idx], train_lbl[idx]

                    self._train_one_batch(train_dat=trn_dat, train_lbl=trn_lbl, optim=optim, disc_optim=None,
                                          train_vars=self.train_vars, train_discri_vars=None, target_c=None, step=global_step)
                    global_step += 1

                # validation
                for val_step in tqdm.trange(0, len(valid_dat), batch_size, desc="Validation step: epoch%d" % cur_epoch):
                    val_dat, val_lbl = valid_dat[val_step:val_step+batch_size], valid_lbl[val_step:val_step+batch_size]

                    if val_step + batch_size >= len(valid_dat): self.valid_save = True
                    self._valid_logger(valid_dat=val_dat, valid_lbl=val_lbl, epoch=cur_epoch, target_c=None)

                if self.model_select == True:
                    self.cls_model.save(os.path.join(self.save_path + '/cls_model_%03d' % cur_epoch))
                    self.enc_model.save(os.path.join(self.save_path + '/enc_model_%03d' % cur_epoch))
                    self.model_select = False

                Total_test_acc, test_count = 0, 0
                for tst_step in tqdm.trange(0, len(test_dat), batch_size, desc="Test step"):
                    tst_dat, tst_lbl = test_dat[tst_step:tst_step+batch_size], test_lbl[tst_step:tst_step+batch_size]

                    res = self.cls_model({"cls_in": tst_dat}, training=False)["cls_out"]
                    acc = G_losses.acc(tst_lbl, res)

                    if tst_step == 0: Total_test_acc = acc
                    else: Total_test_acc += acc
                    test_count += 1

                print("Test ACC: %f" % (Total_test_acc / test_count))
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("test_acc", (Total_test_acc / test_count), step=cur_epoch)

        elif mode == 1:
            self.discriminator_summary_writer = tf.summary.create_file_writer(self.save_path + "/train_critic")
            self.generator_summary_writer = tf.summary.create_file_writer(self.save_path + "/train_generator")

            lr_schedule_g = keras.optimizers.schedules.ExponentialDecay(learning_rate_g, decay_steps=len(train_dat) // batch_size, decay_rate=learning_decay, staircase=True)
            lr_schedule_d = keras.optimizers.schedules.ExponentialDecay(learning_rate_d, decay_steps=len(train_dat) // batch_size, decay_rate=learning_decay, staircase=True)
            optim_g = keras.optimizers.Adam(lr_schedule_g)
            optim_d = keras.optimizers.Adam(lr_schedule_d)
            global_step = 0

            for cur_epoch in tqdm.trange(0, epoch, desc=file_name):
                train_idx = np.squeeze(np.random.permutation(train_idx))

                # Training step
                for cur_step in tqdm.trange(0, len(train_idx), batch_size, desc="Epoch : %d" % cur_epoch):
                    cur_idx = train_idx[cur_step:cur_step + batch_size]
                    cur_dat, cur_lbl = train_dat[cur_idx], train_lbl[cur_idx]
                    target_c = G_utils.code_creator(len(cur_dat), MINI=True)

                    self._train_one_batch(train_dat=cur_dat, train_lbl=cur_lbl, optim=optim_g, disc_optim=optim_d, train_vars=self.train_vars,
                                          train_discri_vars=self.train_discri_vars, step=global_step, target_c=target_c)
                    global_step += 1

                # validation step
                for val_step in tqdm.trange(0, len(valid_dat), batch_size, desc="Validation step"):
                    val_dat, val_lbl = valid_dat[val_step:val_step + batch_size], valid_lbl[val_step:val_step + batch_size]

                    if val_step == ((len(valid_dat) // batch_size)-1) * batch_size: self.valid_save = True
                    self._valid_logger(valid_dat=val_dat, valid_lbl=val_lbl, epoch=cur_epoch, target_c=G_utils.flip(val_lbl))

                if self.model_select == True:
                    self.dec_model.save(os.path.join(self.save_path + '/dec_model_%03d' % cur_epoch))
                    self.model_select = False

                # Test step
                test_acc = G_test.test(self.dec_model, self.cls_model, red_test_dat, orange_test_dat)
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("mode1_ACC_score", test_acc, step=cur_epoch)

                G_test.Visualization(self.dec_model, red_test_dat, orange_test_dat, self.save_path, cur_epoch)

tr=Trainer()
tr.train()