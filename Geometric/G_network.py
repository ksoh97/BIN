import tensorflow as tf
import tensorflow.keras as keras
from Geometric import G_layers, G_config as conf

class Geometric_network:
    def __init__(self, ch=conf.cfmap_ch):
        self.ch = ch
        self.layers = {}
        self.c1, self.c2, self.c3 = G_layers.geo_c1(name="c1"), G_layers.geo_c2(name="c2"), G_layers.geo_c3(name="c3")
        self.enc_in_layer = G_layers.input_layer2d(name="enc_in")
        self.pretraining_clf(), self.CFmap_generator()

    def conv_bn_act(self, x, f, n, s, k=None, p="same", rank=2, act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = G_layers.conv_transpose
        else:
            c_layer = G_layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, rank=rank, dilation_rate=(1,1), name=n+"_conv")
        else:
            conv_l = c_layer(f=f, p=p, rank=rank, dilation_rate=(1,1),  name=n+"_conv")

        out = conv_l(x)
        norm_l = G_layers.batch_norm(name=n + "_norm")
        out = norm_l(out)

        self.layers[n+"_conv"] = conv_l
        self.layers[n+"_norm"] = norm_l

        if act:
            act_l = G_layers.relu(name=n + "_relu")
            self.layers[n + "_relu"] = act_l
            out = act_l(out)
        return out

    def conv_bn_act_reuse(self, x, n, act=True):
        out = self.layers[n+"_conv"](x)
        out = self.layers[n+"_norm"](out)

        if act:
            out = self.layers[n + "_relu"](out)
        return out

    def concat(self, x, y, n):
        concat_l = G_layers.concat(name=n + "_concat")
        self.layers[n+"_concat"] = concat_l
        return concat_l([x, y])

    def flatten_layer(self, x, n=None):
        flatten_l = G_layers.flatten(n + "_flatten")(x)
        self.layers[n+"_flatten"] = flatten_l
        return flatten_l

    def dense_layer(self, x, f, act="relu", n=None):
        dense_l = G_layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)
        self.layers[n+"_dense"] = dense_l

        if act:
            act_l = G_layers.relu(n + "_relu")
            self.layers[n+"_relu"] = act_l
            out = act_l(out)
        return out

    def pretraining_clf(self):
        # Encoder
        enc_conv1 = self.conv_bn_act(x=self.enc_in_layer, k=4, s=2, f=self.ch, n="enc_conv1")
        enc_conv2 = self.conv_bn_act(x=enc_conv1, k=4, s=2, f=self.ch, n="enc_conv2")
        enc_conv3 = self.conv_bn_act(x=enc_conv2, k=4, s=2, f=self.ch * 2, n="enc_conv3")
        enc_conv4 = self.conv_bn_act(x=enc_conv3, k=4, s=2, f=self.ch * 2, n="enc_conv4")

        # Classifier
        flatten = self.flatten_layer(x=enc_conv4, n="flatten")
        dense1 = self.dense_layer(x=flatten, f=256, n="dense1")
        dense2 = self.dense_layer(x=dense1, f=2, act=None, n="dense2")
        cls_out = G_layers.softmax(dense2, name="softmax")

        self.enc_model = keras.Model({"enc_in": self.enc_in_layer}, {"enc_out": enc_conv4}, name="enc_model")
        self.cls_model = keras.Model({"cls_in": self.enc_in_layer}, {"cls_out": cls_out}, name="cls_model")

        return self.enc_model, self.cls_model

    def CFmap_generator(self):
        # Encoder
        enc_conv1 = self.conv_bn_act(x=self.enc_in_layer, k=4, s=2, f=self.ch,  n="enc_conv1")
        enc_conv2 = self.conv_bn_act(x=enc_conv1, k=4, s=2, f=self.ch, n="enc_conv2")
        enc_conv3 = self.conv_bn_act(x=enc_conv2, k=4, s=2, f=self.ch*2, n="enc_conv3")
        enc_conv4 = self.conv_bn_act(x=enc_conv3, k=4, s=2, f=self.ch*2, n="enc_conv4")

        # Decoder
        dec_up3 = G_layers.upsample(rank=2, name="dec_up3")(enc_conv4)
        dec_code_concat3 = self.concat(enc_conv3, self.c3, n="dec_code_concat3")
        dec_code_conv3 = self.conv_bn_act(x=dec_code_concat3, f=self.ch*2, k=3, s=1, p="same", n="dec_code_conv3")
        dec_concat3 = self.concat(dec_code_conv3, dec_up3, n="dec_concat3")
        dec_conv3 = self.conv_bn_act(x=dec_concat3, f=self.ch*2, k=3, s=1, p="same", n="dec_conv3")

        dec_up2 = G_layers.upsample(rank=2, name="dec_up2")(dec_conv3)
        dec_code_concat2 = self.concat(enc_conv2, self.c2, n="dec_code_concat2")
        dec_code_conv2 = self.conv_bn_act(x=dec_code_concat2, f=self.ch, k=3, s=1, p="same", n="dec_code_conv2")
        dec_concat2 = self.concat(dec_code_conv2, dec_up2, n="dec_concat2")
        dec_conv2 = self.conv_bn_act(x=dec_concat2, f=self.ch, k=3, s=1, p="same", n="dec_conv2")

        dec_up1 = G_layers.upsample(rank=2, name="dec_up1")(dec_conv2)
        dec_code_concat1 = self.concat(enc_conv1, self.c1, n='dec_code_concat1')
        dec_code_conv1 = self.conv_bn_act(x=dec_code_concat1, f=self.ch, k=3, s=1, p="same", n="dec_code_conv1")
        dec_concat1 = self.concat(dec_code_conv1, dec_up1, n="dec_concat1")
        dec_conv1 = self.conv_bn_act(x=dec_concat1, f=self.ch, k=3, s=1, p="same", n="dec_conv1")

        dec_up = G_layers.upsample(rank=2, name="dec_up")(dec_conv1)
        dec_out = self.conv_bn_act(x=dec_up, f=3, k=1, s=1, act=False, p="same", n="dec_out")
        dec_out = G_layers.tanh(x=dec_out, name="dec_out_tanh")

        self.dec_model = keras.Model({"dec_in": self.enc_in_layer, "c1": self.c1, "c2": self.c2, "c3": self.c3},
                                     {"dec_out": dec_out}, name="dec_model")

        # Classifier
        enc_conv1 = self.conv_bn_act_reuse(x=self.enc_in_layer, n="enc_conv1")
        enc_conv2 = self.conv_bn_act_reuse(x=enc_conv1, n='enc_conv2')
        enc_conv3 = self.conv_bn_act_reuse(x=enc_conv2, n="enc_conv3")
        enc_conv4 = self.conv_bn_act_reuse(x=enc_conv3, n='enc_conv4')

        flatten = self.flatten_layer(x=enc_conv4, n="flatten")
        dense1 = self.dense_layer(x=flatten, f=256, n='dense1')
        dense2 = self.dense_layer(x=dense1, f=2, act=None, n="dense2")
        self.cls_out = G_layers.softmax(dense2, name="softmax")

        self.cls_model = keras.Model({"cls_in": self.enc_in_layer}, {"cls_out": self.cls_out}, name="cls_model")

        return self.cls_model, self.dec_model

class Discriminator:
    def __init__(self, ch=conf.disc_ch):
        self.ch = ch
        self.discri_in_layer = G_layers.input_layer2d(name="discriminator_in")
        tf.keras.backend.set_image_data_format("channels_last")
        self.build_model()

    def conv_bn_act(self, x, f, n, s, k=None, p="same", rank=2, batch=True, act=True):
        c_layer = G_layers.conv
        if k:
            conv_l = c_layer(f=f, k=k, p=p, s=s, rank=rank, dilation_rate=(1,1), name=n+"_conv")
        else:
            conv_l = c_layer(f=f, p=p, s=s, rank=rank, dilation_rate=(1,1), name=n + "_conv")

        out = conv_l(x)

        if batch:
            batch_l = G_layers.batch_norm(name=n + "_batchnorm")
            out = batch_l(out)

        if act:
            act_l = G_layers.leaky_relu(name=n + "_leakyrelu")
            out = act_l(out)

        return out

    def flatten_layer(self, x, n=None):
        flatten_l = G_layers.flatten(n + "_flatten")(x)
        return flatten_l

    def dense_layer(self, x, f, act="leakyrelu", n=None):
        dense_l = G_layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)

        if act:
            act_l = G_layers.leaky_relu(n + "_leakyrelu")
            out = act_l(out)

        return out

    def build_model(self):
        discri_conv1 = self.conv_bn_act(x=self.discri_in_layer, k=4, s=2, f=self.ch, batch=False, n="discri_conv1")
        discri_conv2 = self.conv_bn_act(x=discri_conv1, k=4, s=2, f=self.ch, n="discri_conv2")
        discri_conv3 = self.conv_bn_act(x=discri_conv2, k=4, s=2, f=self.ch*2, n="discri_conv3")
        discri_conv4 = self.conv_bn_act(x=discri_conv3, k=4, s=2, f=self.ch*2, n="discri_conv4")

        flatten = self.flatten_layer(x=discri_conv4, n="flatten")
        dense = self.dense_layer(x=flatten, f=1, act=False, n="dense")
        logit = tf.identity(dense)

        self.discriminator_model = keras.Model({"discri_in": self.discri_in_layer}, {"discri_out": logit}, name="cri_model")

        return self.discriminator_model
