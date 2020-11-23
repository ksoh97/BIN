mode_dict = {"pre-training the classifier": 0, "training the counterfactual map generator": 1}
mode = 1

disc_ch, cfmap_ch = 32, 32
epoch = 50
batch_size = 128
lr_decay = 0.98

data_path = "Your data path in here"
save_path = "Your storage path in here"
cls_weight_path = "Pre-trained classifier model path in here"
enc_weight_path = "Pre-trained encoder model path in here"

if mode == 0:
    file_name = "Geometric_classifier_pertraining"
    lr = 0.0005

else:
    file_name = "Geometric_cfmap"
    lr_g, lr_d = 0.0005, 0.0005
    g_step, d_step = 1, 1
    one_sided_label_smoothing = True

# Hyper-param
hyper_param = [1.0,  10.0,  1.0,  1.0,  0.5]
loss_type = {'cls': hyper_param[0], 'norm': hyper_param[1], 'GAN': hyper_param[2], 'cyc': hyper_param[3], 'dis': hyper_param[4]}
