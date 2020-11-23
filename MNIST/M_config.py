mode_dict = {"pre-training the classifier": 0, "training the counterfactual map generator": 1}
mode = 1

disc_ch, cfmap_ch = 32, 32
epoch = 100
batch_size = 256

save_path = "Your storage path in here"
cls_weight_path = "Pre-trained classifier model path in here"
enc_weight_path = "Pre-trained encoder model path in here"

if mode == 0:
    file_name = "MNIST_classifier_pretraining"
    lr = 0.0001
    lr_decay = 0.98
elif mode == 1:
    file_name = "MNIST_cfmap"
    g_step, d_step = 1, 1
    lr_g, lr_d = 0.001, 0.001
    lr_decay = 0.99
    beta_1 = 0.9
    one_sided_label_smoothing = True

# Hyper-param
hyper_param = [1.0, 1.0, 1.0, 1.0, 0.5]
loss_type = {'cls': hyper_param[0], 'norm': hyper_param[1], 'GAN': hyper_param[2], 'cyc': hyper_param[3],
             'dis': hyper_param[4]}
