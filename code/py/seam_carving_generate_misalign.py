import cv2
import numpy as np
import random
import time
import torch
import torch.nn.functional as F


TIME = True


def time_log(func):
    def warp_func(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        if TIME:
            print("%s execute in %.4f s" % (func.__name__, time.time()-t0))
        return result
    return warp_func


class SeamCarving(object):
    def __init__(self, energy_func='gradient_canny',
                 horizontal_change_range=(5, 8),
                 vertical_change_range=(5, 8)):

        def gradient_L1(x):
            x = np.copy(x)
            x = cv2.GaussianBlur(x, (3, 3), 0)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
            return np.abs(sobel_x) + np.abs(sobel_y)

        def gradient_canny(x):
            img = np.copy(x)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            canny = cv2.Canny(img, 50, 150)
            return canny

        self.energy_func_dict = {'gradient_L1': gradient_L1,
                                 'gradient_canny': gradient_canny}

        if energy_func in self.energy_func_dict.keys():
            self.energy_func = self.energy_func_dict[energy_func]
        else:
            raise ValueError

        self.horizontal_change_range = horizontal_change_range
        self.vertical_change_range = vertical_change_range
        self.count = 0

    def search_path(self, energy, min_num=3, check_overlap=True, sample_factor=2):
        # path: '0', left down
        #       '1', down
        #       '2', right down
        H, W = energy.shape
        energy_map = np.copy(energy) + np.random.randn(H, W).clip(-1)+1
        dp_array = np.zeros_like(energy_map)
        path_array = np.zeros_like(energy_map)

        H, W = energy_map.shape

        @time_log
        def min_sum_path(dp_array, path_array):
            self.count = 0

            def min_func(x):
                index = x.argmin()
                val = x[index]
                #indexs = np.where(x == val)[0]
                #index = indexs[np.random.randint(0, indexs.shape[0])]
                #index = indexs[indexs.shape[0]//2]
                #index = indexs[0]
                #x += np.random.randn(x.shape[0])
                #index = x.argmin()
                #index = indexs[self.count%indexs.shape[0]]
                #self.count += 1
                return val, index

            dp_array = energy_map

            for h in range(1, H):
                for w in range(W):
                    if w == 0:
                        val, index = min_func(dp_array[h-1, w:w+2])
                        dp_array[h, w] += val
                        path_array[h, w] = index+1
                    elif w == W-1:
                        val, index = min_func(dp_array[h-1, w-1:w+1])
                        dp_array[h, w] += val
                        path_array[h, w] = index
                    else:
                        val, index = min_func(dp_array[h-1, w-1:w+2])
                        dp_array[h, w] += val
                        path_array[h, w] = index
            return dp_array, path_array

        @time_log
        def min_sum_path_torch(dp_array, path_array, MAX_VAL=99999999):
            dp_array = torch.from_numpy(energy_map)
            path_array = torch.from_numpy(path_array)
            for h in range(1, H):
                unfold = F.unfold(dp_array[h-1].view(1,1,1,-1), kernel_size=(1,3), stride=1, padding=0)
                unfold = unfold.squeeze()
                val, index = unfold.min(dim=0)
                dp_array[h, 1:W-1] += val
                path_array[h, 1:W-1] = index
            dp_array[-1, 0] = MAX_VAL
            dp_array[-1, -1] = MAX_VAL
            return dp_array.numpy(), path_array.numpy()

        def get_seam(idx):
            result = []
            result.append(idx)
            cur_idx = idx
            for h in range(energy_map.shape[0]-1, 0, -1):
                if path_array[h, cur_idx] == 0:
                    cur_idx -= 1
                elif path_array[h, cur_idx] == 1:
                    cur_idx = cur_idx
                elif path_array[h, cur_idx] == 2:
                    cur_idx += 1
                result.append(cur_idx)
            result.reverse()
            return result

        def repeat_check(src_list):
            L = len(src_list[0])
            dst_list = []
            dst_list.append(src_list[0])
            for i in range(1, len(src_list)):
                src_begin = src_list[i][0]
                src_mid = src_list[i][L//2]
                src_end = src_list[i][L-1]
                check_flag = True
                for item in dst_list:
                    dst_begin = item[0]
                    dst_mid = item[L//2]
                    dst_end = item[L-1]
                    if (src_begin-dst_begin)*(src_mid-dst_mid) <= 0:
                        check_flag = False
                        break
                    if (src_mid-dst_mid)*(src_end-dst_end) <= 0:
                        check_flag = False
                        break
                if check_flag:
                    dst_list.append(src_list[i])

            return dst_list

        dp_array, path_array = min_sum_path_torch(dp_array, path_array)

        sample_num = (int)(min_num*sample_factor)
        divide_range = W // sample_num - 1
        min_num_index = []
        for i in range(sample_num):
            min_num_index.append(dp_array[-1, i*divide_range:(i+1)*divide_range].argmin()+i*divide_range)
        # min_num_index = np.argpartition(dp_array[-1], -sample_num)[0:sample_num]
        min_num_index = np.array(min_num_index)
        min_num_index = min_num_index[random.sample(range(sample_num), min_num)]

        seams = []
        for i in range(min_num):
            seams.append(get_seam(min_num_index[i]))

        if check_overlap:
            seams = repeat_check(seams)
        seams = sorted(seams)
        return seams

    def visualize_seam(self, origin_img, seams, direction=None):
        assert(isinstance(origin_img, np.ndarray))
        img = np.copy(origin_img)
        H, W, C = img.shape
        if direction == 'V':
            for j in range(len(seams)):
                for i in range(H):
                    img[i, seams[j][i], 0] = 0
                    img[i, seams[j][i], 1] = 0
                    img[i, seams[j][i], 2] = 255
        elif direction == 'H':
            for j in range(len(seams)):
                for i in range(W):
                    img[seams[j][i], i, 0] = 0
                    img[seams[j][i], i, 1] = 0
                    img[seams[j][i], i, 2] = 255
        else:
            raise ValueError
        cv2.imwrite('seam.png', img)

    @time_log
    def generate_seams(self, img, vis_direction=None, DEBUG=False, **kwargs):
        energy_map = self.energy_func(img.astype(np.uint8)).astype(np.float32)
        seams = self.search_path(energy_map, **kwargs)
        if DEBUG:
            assert(vis_direction is not None)
            self.visualize_seam(img, seams, vis_direction)
        return seams

    @time_log
    def delete_seams(self, img, seams, flag_constant=0, DEBUG=True):
        H, W, C = img.shape
        for item in seams:
            for h in range(H):
                if img[h, item[h], 0] == flag_constant:
                    offset = 0
                    while(img[h, item[h]+offset, 0] == flag_constant):
                        offset += 1
                    img[h, item[h]+offset] = flag_constant
                else:
                    img[h, item[h]] = flag_constant
        delete_idx = np.where(img != flag_constant)
        img = img[delete_idx].reshape(H, W-len(seams), 3)
        if DEBUG:
            print(img.shape)
        return img

    @time_log
    def add_seams(self, img, seams, DEBUG=True):
        H, W, C = img.shape
        L = len(seams)
        img_expand = np.zeros((H, W+L, 3))
        for h in range(H):
            idx = 0
            repeat_count = 0
            for i in range(L):
                val = seams[i][h]
                img_expand[h, idx+i:val+i+1] = img[h, idx:val+1]
                if idx == val:
                    img_expand[h, val+i+1+repeat_count] = np.average(img[h, val:val+2])
                    repeat_count += 1
                else:
                    repeat_count = 0
                    img_expand[h, val+i+1] = np.average(img[h, val:val+2])
                idx = val
            img_expand[h, idx+L+1:] = img[h, idx+1:]
        if DEBUG:
            print(img_expand.shape)
        return img_expand

    @time_log
    def __call__(self, img, check_overlap=False):

        assert(isinstance(img, np.ndarray))
        H, W, C = img.shape

        misalign = np.copy(img).astype(np.float32) + 0.01
        vertical_change = np.random.randint(self.vertical_change_range[0], self.vertical_change_range[1])
        horizontal_change = np.random.randint(self.horizontal_change_range[0], self.horizontal_change_range[1])

        # forward(reduce)
        vertical_seams = self.generate_seams(misalign, 'V', min_num=vertical_change, check_overlap=check_overlap)
        misalign = self.delete_seams(misalign, vertical_seams)

        misalign = misalign.transpose(1, 0, 2)
        horizontal_seams = self.generate_seams(misalign, 'H', min_num=horizontal_change, check_overlap=check_overlap)
        misalign = self.delete_seams(misalign, horizontal_seams)
        misalign = misalign.transpose(1, 0, 2)

        # backward(expand)
        vertical_seams = self.generate_seams(misalign, 'V', min_num=vertical_change, check_overlap=False)
        misalign = self.add_seams(misalign, vertical_seams)

        misalign = misalign.transpose(1, 0, 2)
        horizontal_seams = self.generate_seams(misalign, 'H', min_num=horizontal_change, check_overlap=False)
        misalign = self.add_seams(misalign, horizontal_seams)
        misalign = misalign.transpose(1, 0, 2)

        return misalign


def seam_carving_interface(input_path, output_path, DEBUG=True):
    img = cv2.imread(input_path)
    handle_seam_carving = SeamCarving()
    output = handle_seam_carving(img)
    cv2.imwrite(output_path, output)
    if DEBUG:
        sub = img - output
        sub = cv2.cvtColor(sub.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        sub_path = output_path.split('.')[0] + '_sub.' + output_path.split('.')[1]
        cv2.imwrite(sub_path, sub)


if __name__ == "__main__":
    #input_path = '/Users/nyz/code/github/seam_carving/data/image_input.png'
    input_path = 'jun1.jpg'
    output_path = input_path.split('.')[0]+'_misalign.'+input_path.split('.')[1]
    seam_carving_interface(input_path, output_path)
