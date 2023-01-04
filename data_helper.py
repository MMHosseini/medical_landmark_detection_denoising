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

    def depatch_image(self, patches):
        border, uncovered_area = self._depatch_border(patches)
        center, mask = self._depatch_center(patches)
        reconstructed = (center * mask) + (border * (1 - mask))
        reconstructed = np.round(reconstructed)
        return reconstructed, uncovered_area

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

    def _depatch_border(self, patches):
        patches = patches[:, :, :, 0]
        patch_height = Variables.patch_height
        patch_width = Variables.patch_width

        center = (patch_height // 2, patch_width // 2)

        if Variables.model_name == 'central_denoising':
            manipulated_area = Variables.noisy_area
        elif Variables.model_name == 'central_reconstruction':
            manipulated_area = Variables.destroyed_area

        manipulated_area_x_start = center[0] - round((manipulated_area / 2) * patch_height)
        manipulated_area_x_end = center[0] + round((manipulated_area / 2) * patch_height) + 1
        manipulated_area_y_start = center[1] - round((manipulated_area / 2) * patch_width)
        manipulated_area_y_end = center[1] + round((manipulated_area / 2) * patch_width) + 1

        manipulated_area_height = manipulated_area_x_end - manipulated_area_x_start
        manipulated_area_width = manipulated_area_y_end - manipulated_area_y_start

        image_width = Variables.image_input_size_width
        image_height = Variables.image_input_size_height
        image = np.zeros((image_height, image_width))
        mask = np.zeros((image_height, image_width))

        num_hor_patches = (image_width-(patch_width-manipulated_area_width)) // manipulated_area_width
        num_ver_patches = (image_height-(patch_height-manipulated_area_height)) // manipulated_area_height

        patch_mask = np.ones((patch_width, patch_height))

        counter = 0
        x_start = -manipulated_area_height
        x_end = x_start + patch_height
        for i in range(num_ver_patches):
            x_start += manipulated_area_height
            x_end += manipulated_area_height
            y_start = -manipulated_area_width
            y_end = y_start + patch_width
            for j in range(num_hor_patches):
                y_start += manipulated_area_width
                y_end += manipulated_area_width
                image[x_start:x_end, y_start:y_end] += patches[counter]
                mask[x_start:x_end, y_start:y_end] += patch_mask
                counter += 1

        uncovered_area = mask.copy()
        uncovered_area[uncovered_area == 0] = 255
        uncovered_area[uncovered_area < 255] = 0
        uncovered_area[uncovered_area == 255] = 1

        mask[mask == 0] = 1  # avoid divide by zero
        border = np.divide(image, mask)
        # border[manipulated_area_x_start:image_height-(patch_height - manipulated_area_x_end),
        #     manipulated_area_y_start:image_width-(patch_width - manipulated_area_y_end)] = 0
        return border, uncovered_area

    def _depatch_center(self, patches):
        patches = patches[:, :, :, 0]
        patch_height = Variables.patch_height
        patch_width = Variables.patch_width

        center = (patch_height // 2, patch_width // 2)

        if Variables.model_name == 'central_denoising':
            manipulated_area = Variables.noisy_area
        elif Variables.model_name == 'central_reconstruction':
            manipulated_area = Variables.destroyed_area
        else:
            print('Error in DataHelper --> depatch_image: This is not a patch-based model')
            return

        manipulated_area_x_start = center[0] - round((manipulated_area / 2) * patch_height)
        manipulated_area_x_end = center[0] + round((manipulated_area / 2) * patch_height) + 1
        manipulated_area_y_start = center[1] - round((manipulated_area / 2) * patch_width)
        manipulated_area_y_end = center[1] + round((manipulated_area / 2) * patch_width) + 1

        manipulated_area_height = manipulated_area_x_end - manipulated_area_x_start
        manipulated_area_width = manipulated_area_y_end - manipulated_area_y_start

        image_width = Variables.image_input_size_width
        image_height = Variables.image_input_size_height
        image = np.zeros((image_height, image_width))
        mask = np.zeros((image_height, image_width))

        num_hor_patches = (image_width-(patch_width-manipulated_area_width)) // manipulated_area_width
        num_ver_patches = (image_height-(patch_height-manipulated_area_height)) // manipulated_area_height

        patch_mask = np.ones((patch_width, patch_height))

        counter = 0
        x_start = manipulated_area_x_start - manipulated_area_height
        x_end = x_start + manipulated_area_height
        for i in range(num_ver_patches):
            x_start += manipulated_area_height
            x_end += manipulated_area_height
            y_start = manipulated_area_y_start - manipulated_area_width
            y_end = y_start + manipulated_area_width
            for j in range(num_hor_patches):
                y_start += manipulated_area_width
                y_end += manipulated_area_width
                image[x_start:x_end, y_start:y_end] += patches[counter, manipulated_area_x_start:manipulated_area_x_end,
                                                       manipulated_area_y_start:manipulated_area_y_end]
                mask[x_start:x_end, y_start:y_end] += patch_mask[manipulated_area_x_start:manipulated_area_x_end,
                                                       manipulated_area_y_start:manipulated_area_y_end]
                counter += 1

        return image, mask
