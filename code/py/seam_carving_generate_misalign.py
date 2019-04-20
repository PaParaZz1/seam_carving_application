import cv2
import numpy as np
import random
import time


def time_log(func):
    def warp_func(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        print("%s execute in %.4f s" % (func.__name__, time.time()-t0))
        return result
    return warp_func


class SeamCarving(object):
    def __init__(self, energy_func='gradient_canny',
                 horizontal_change_range=(3, 6),
                 vertical_change_range=(3, 6)):

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
            cv2.imwrite('canny.png', canny)
            return canny

        self.energy_func_dict = {'gradient_L1': gradient_L1,
                                 'gradient_canny': gradient_canny}

        if energy_func in self.energy_func_dict.keys():
            self.energy_func = self.energy_func_dict[energy_func]
        else:
            raise ValueError

        self.horizontal_change_range = horizontal_change_range
        self.vertical_change_range = vertical_change_range

    def search_path(self, energy, min_num=3, check_overlap=True, sample_factor=2):
        # path: '0', left down
        #       '1', down
        #       '2', right down

        energy_map = np.copy(energy)
        dp_array = np.zeros_like(energy_map)
        path_array = np.zeros_like(energy_map)

        H, W = energy_map.shape

        @time_log
        def min_sum_path(dp_array, path_array):

            def min_func(x):
                val = x.min()
                indexs = np.where(x == val)[0]
                index = indexs[np.random.randint(0, indexs.shape[0])]
                #index = indexs[indexs.shape[0]//2]
                #index = indexs[0]
                #x += np.random.randn(x.shape[0])
                #index = x.argmin()
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

        dp_array, path_array = min_sum_path(dp_array, path_array)

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

    def __call__(self, img, check_overlap=False):

        assert(isinstance(img, np.ndarray))
        H, W, C = img.shape

        misalign = np.copy(img).astype(np.float32) + 0.01

        # forward(reduce)
        energy_map = self.energy_func(misalign.astype(np.uint8)).astype(np.float32)
        vertical_change = np.random.randint(self.vertical_change_range[0], self.vertical_change_range[1])
        vertical_seams = self.search_path(energy_map, min_num=vertical_change, check_overlap=check_overlap)
        self.visualize_seam(misalign, vertical_seams, 'V')

        for item in vertical_seams:
            for h in range(H):
                if misalign[h, item[h], 0] == 0:
                    offset = 0
                    while(misalign[h, item[h]+offset, 0] == 0):
                        offset += 1
                    misalign[h, item[h]+offset] = 0
                else:
                    misalign[h, item[h]] = 0
        misalign_h_idx = np.where(misalign != 0)
        misalign = misalign[misalign_h_idx].reshape(H, W-len(vertical_seams), 3)
        print(misalign.shape)
        W -= len(vertical_seams)

        misalign = misalign.transpose(1, 0, 2)
        energy_map = self.energy_func(misalign.astype(np.uint8)).astype(np.float32)
        horizontal_change = np.random.randint(self.horizontal_change_range[0], self.horizontal_change_range[1])
        horizontal_seams = self.search_path(energy_map, min_num=horizontal_change, check_overlap=check_overlap)

        for item in horizontal_seams:
            for w in range(W):
                if misalign[w, item[w], 0] == 0:
                    offset = 0
                    while(misalign[w, item[w]+offset, 0] == 0):
                        offset += 1
                    misalign[w, item[w]+offset] = 0
                else:
                    misalign[w, item[w]] = 0
        misalign_w_idx = np.where(misalign != 0)
        misalign = misalign[misalign_w_idx].reshape(W, H-len(horizontal_seams), 3)
        misalign = misalign.transpose(1, 0, 2)
        print(misalign.shape)
        H -= len(horizontal_seams)

        # backward(expand)

        energy_map = self.energy_func(misalign.astype(np.uint8)).astype(np.float32)
        vertical_seams = self.search_path(energy_map, min_num=vertical_change, check_overlap=False)

        misalign_expand = np.zeros((H, W+vertical_change, 3))
        for h in range(H):
            idx = 0
            for i in range(len(vertical_seams)):
                val = vertical_seams[i][h]
                misalign_expand[h, idx+i:val+i] = misalign[h, idx:val]
                misalign_expand[h, val+i] = misalign[h, val]
                idx = val
            misalign_expand[h, idx+len(vertical_seams):] = misalign[h, idx:]
        W += len(vertical_seams)

        print(misalign_expand.shape)

        misalign = misalign_expand
        misalign = misalign.transpose(1, 0, 2)

        energy_map = self.energy_func(misalign.astype(np.uint8)).astype(np.float32)
        horizontal_seams = self.search_path(energy_map, min_num=horizontal_change, check_overlap=False)

        misalign_expand = np.zeros((W, H+horizontal_change, 3))
        for w in range(W):
            idx = 0
            for i in range(len(horizontal_seams)):
                val = horizontal_seams[i][w]
                misalign_expand[w, idx+i:val+i] = misalign[w, idx:val]
                misalign_expand[w, val+i] = misalign[w, val]
                idx = val
            misalign_expand[w, idx+len(horizontal_seams):] = misalign[w, idx:]
        H += len(horizontal_seams)
        misalign = misalign_expand.transpose(1, 0, 2)
        print(misalign.shape)

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
    #input_path = '../../data/image_input.png'
    input_path = 'jun1.jpg'
    output_path = input_path.split('.')[0]+'_misalign.'+input_path.split('.')[1]
    print(output_path)
    seam_carving_interface(input_path, output_path)
