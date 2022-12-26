import os
from config import Constants, Variables
import cv2
import numpy as np
from tqdm import tqdm


class DataHelper:
    def make_dataset(self):
        train_path = Constants.train_path
        test_path = Constants.test_path
        train_path_noisy = Constants.train_path_noisy
        test_path_noisy = Constants.test_path_noisy
        train_image_name = os.listdir(train_path)
        test_image_name = os.listdir(test_path)
        self._create_set(train_image_name, train_path, train_path_noisy)
        self._create_set(test_image_name, test_path, test_path_noisy)

    def _create_set(self, names, source_path, target_path):
        if not os.path.isdir(target_path):
            os.mkdir(target_path)

        for name in tqdm(names):
            if not name.endswith('.jpg'):
                continue
            name = name.split('.jpg')[0]
            info = dict()
            source_adr = source_path + name + '.jpg'
            target_adr = target_path + name + '.npy'
            image = cv2.imread(source_adr)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = self._resize_image(image)
            image, image_noisy_full = self._get_noisy_image_full(image)
            image_non_noisy_patched, image_noisy_patched = self._get_noisy_image_patched(image)
            image_non_destroyed_patched, image_destroyed_patched = self._get_destroyed_image_patched(image)
            info['image'] = image
            info['full_noisy'] = image_noisy_full
            info['patched_non_noisy'] = image_non_noisy_patched
            info['patched_noisy'] = image_noisy_patched
            info['patched_non_destroyed'] = image_non_destroyed_patched
            info['patched_destroyed'] = image_destroyed_patched
            np.save(target_adr, info)

    def _resize_image(self, image):
        src_height = image.shape[0]
        src_width = image.shape[1]
        tar_height = Variables.image_input_size_height
        tar_width = Variables.image_input_size_width
        if (src_height != tar_height) or (src_width != tar_width):
            image = cv2.resize(image, (tar_width, tar_height))
        return image

    def _get_noisy_image_full(self, image):
        noisy_image = self._make_whole_image_noisy(image)
        return image, noisy_image

    def _get_noisy_image_patched(self, image):
        patch_height = Variables.patch_height
        patch_width = Variables.patch_width

        center = (patch_height // 2, patch_width // 2)

        noisy_area_x_start = center[0] - round((Variables.noisy_area / 2) * patch_height)
        noisy_area_x_end = center[0] + round((Variables.noisy_area / 2) * patch_height) + 1
        noisy_area_y_start = center[1] - round((Variables.noisy_area / 2) * patch_width)
        noisy_area_y_end = center[1] + round((Variables.noisy_area / 2) * patch_width) + 1

        noisy_area_height = noisy_area_x_end - noisy_area_x_start
        noisy_area_width = noisy_area_y_end - noisy_area_y_start

        central_image = image[noisy_area_x_start:-patch_height+noisy_area_x_end,
                        noisy_area_y_start:-patch_width+noisy_area_y_end].copy()
        central_image = self._make_whole_image_noisy(central_image)

        num_ver_patches = central_image.shape[0] // noisy_area_height
        num_hor_patches = central_image.shape[1] // noisy_area_width

        noisy_image_patch_list = []
        clear_image_patch_list = []
        x1_start = -noisy_area_height
        x1_end = 0
        x2_start = -noisy_area_height
        x2_end = patch_height - noisy_area_height
        for i in range(num_ver_patches):
            x1_start += noisy_area_height
            x1_end += noisy_area_height
            x2_start += noisy_area_height
            x2_end += noisy_area_height

            y1_start = -noisy_area_width
            y1_end = 0
            y2_start = -noisy_area_width
            y2_end = patch_width - noisy_area_width
            for j in range(num_hor_patches):
                y1_start += noisy_area_width
                y1_end += noisy_area_width
                y2_start += noisy_area_width
                y2_end += noisy_area_width

                central_image_patch = central_image[x1_start:x1_end, y1_start:y1_end].copy()
                clear_image_patch = image[x2_start:x2_end, y2_start:y2_end].copy()
                noisy_image_patch = clear_image_patch.copy()
                noisy_image_patch[noisy_area_x_start:noisy_area_x_end, noisy_area_y_start:noisy_area_y_end] = central_image_patch.copy()
                clear_image_patch_list.append(clear_image_patch)
                noisy_image_patch_list.append(noisy_image_patch)

        return clear_image_patch_list, noisy_image_patch_list

    def _get_destroyed_image_patched(self, image):
        patch_height = Variables.patch_height
        patch_width = Variables.patch_width

        center = (patch_height // 2, patch_width // 2)

        destroyed_area_x_start = center[0] - round((Variables.destroyed_area / 2) * patch_height)
        destroyed_area_x_end = center[0] + round((Variables.destroyed_area / 2) * patch_height) + 1
        destroyed_area_y_start = center[1] - round((Variables.destroyed_area / 2) * patch_width)
        destroyed_area_y_end = center[1] + round((Variables.destroyed_area / 2) * patch_width) + 1

        destroyed_area_height = destroyed_area_x_end - destroyed_area_x_start
        destroyed_area_width = destroyed_area_y_end - destroyed_area_y_start

        central_image = image[destroyed_area_x_start:-patch_height+destroyed_area_x_end,
                        destroyed_area_y_start:-patch_width+destroyed_area_y_end].copy()
        central_image = np.zeros(central_image.shape)

        num_ver_patches = central_image.shape[0] // destroyed_area_height
        num_hor_patches = central_image.shape[1] // destroyed_area_width

        destroyed_image_patch_list = []
        clear_image_patch_list = []
        x1_start = -destroyed_area_height
        x1_end = 0
        x2_start = -destroyed_area_height
        x2_end = patch_height - destroyed_area_height
        for i in range(num_ver_patches):
            x1_start += destroyed_area_height
            x1_end += destroyed_area_height
            x2_start += destroyed_area_height
            x2_end += destroyed_area_height

            y1_start = -destroyed_area_width
            y1_end = 0
            y2_start = -destroyed_area_width
            y2_end = patch_width - destroyed_area_width
            for j in range(num_hor_patches):
                y1_start += destroyed_area_width
                y1_end += destroyed_area_width
                y2_start += destroyed_area_width
                y2_end += destroyed_area_width

                central_image_patch = central_image[x1_start:x1_end, y1_start:y1_end].copy()
                clear_image_patch = image[x2_start:x2_end, y2_start:y2_end].copy()
                destroyed_image_patch = clear_image_patch.copy()
                destroyed_image_patch[destroyed_area_x_start:destroyed_area_x_end, destroyed_area_y_start:destroyed_area_y_end] = central_image_patch.copy()
                clear_image_patch_list.append(clear_image_patch)
                destroyed_image_patch_list.append(destroyed_image_patch)

        return clear_image_patch_list, destroyed_image_patch_list

    def _make_whole_image_noisy(self, image):
        noise_type = Variables.noise_type
        if noise_type == "gaussian":
            mean = 0
            var = 0.1
            sigma = np.sqrt(var)
            if image.ndim == 2:
                gauss = np.random.normal(mean, sigma, (image.shape[0], image.shape[1]))
            elif image.ndim == 3:
                gauss = np.random.normal(mean, sigma, (image.shape[0], image.shape[1], image.shape[2]))
            domain = np.random.rand() * 64  # 0 < domain < 64
            noise = gauss * domain
            noisy_image = image + noise
        elif noise_type == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy_image = np.random.poisson(image * vals) / float(vals)

        noisy_image[noisy_image < 0] = 0
        noisy_image[noisy_image > 255] = 255
        noisy_image = np.round(noisy_image).astype('uint8')
        return noisy_image

    def create_source_and_target(self, path):
        source_images = []
        target_images = []

        files = os.listdir(path)
        for file in files:
            info = np.load(path + '/' + file)
            info = info.item()


