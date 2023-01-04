import os


class Constants:
    """setting addressed here:"""
    current_path = os.getcwd()
    prev_path = current_path[0:current_path.rfind('/')]
    _path_prefix = prev_path + '/data/paper_data/'

    train_path = _path_prefix + 'train_set/'
    test_path = _path_prefix + 'test_set/'
    weight_path = _path_prefix + 'weights/'
    train_path_noisy = _path_prefix + 'train_set_noisy/'
    test_path_noisy = _path_prefix + 'test_set_noisy/'


class Variables:
    image_input_size_height = 256
    image_input_size_width = 128
    init_weight_file_address = None
    models = ['full_denoising', 'central_denoising', 'central_reconstruction']
    model_name = models[2]
    test_model_weight = 'epo_149__mse_0.000774.h5' if model_name == models[0] else 'epo_47__mse_0.002398.h5' \
                                        if model_name == models[1] else 'epo_57__mse_0.002504.h5'
    noise_type = 'gaussian'
    start_epoch = 0
    end_epoch = 150
    batch_size = 100
    patch_height = 16
    patch_width = 16
    noisy_area = 0.3
    destroyed_area = 0.25


