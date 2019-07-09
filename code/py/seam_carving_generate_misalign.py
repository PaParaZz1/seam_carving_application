import cv2
import numpy as np
import random
import time
import torch
import torch.nn.functional as F


global viz_save_count
viz_save_count = 0
TIME = False


def time_log(func):
    def warp_func(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        if TIME:
            print("%s execute in %.4f s" % (func.__name__, time.time()-t0))
        return result
    return warp_func


class SeamCarving(object):
    def __init__(self, energy_func='gradient_L1',
                 horizontal_change_range=(1, 2),
                 vertical_change_range=(1, 2)):

        self.energy_func_dict = {'gradient_L1': self.gradient_L1,
                                 'gradient_canny': self.gradient_canny}

        if energy_func in self.energy_func_dict.keys():
            self.energy_func = self.energy_func_dict[energy_func]
        else:
            raise ValueError

        self.horizontal_change_range = horizontal_change_range
        self.vertical_change_range = vertical_change_range
        self.count = 0

    def gradient_L1(self, x):
        x = np.copy(x)
        x = cv2.GaussianBlur(x, (7, 7), 0)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return np.abs(sobel_x) + np.abs(sobel_y)

    def gradient_canny(self, x):
        img = np.copy(x)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        canny = cv2.Canny(img, 25, 150)
        cv2.imwrite('canny.png', canny)
        return canny

    @time_log
    def search_path(self, energy, min_num=3, check_overlap=True, sample_factor=2):
        # path: '0', left down
        #       '1', down
        #       '2', right down
        if min_num == 0:
            return []
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
                dp_array[h, 0] = MAX_VAL
                dp_array[h, W-1] = MAX_VAL
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
        #dp_array, path_array = min_sum_path(dp_array, path_array)

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

    @time_log
    def search_path_torch(self, energy, min_num=3, check_overlap=True, sample_factor=2):
        # path: '0', left down
        #       '1', down
        #       '2', right down
        if min_num == 0:
            return []
        H, W = energy.shape
        energy_map = torch.from_numpy(energy) + torch.randn(H, W).clamp(-1) + 1
        dp_array = energy_map
        path_array = torch.zeros_like(energy_map)


        @time_log
        def min_sum_path_torch(dp_array, path_array, MAX_VAL=99999999):
            for h in range(1, H):
                unfold = F.unfold(dp_array[h-1].view(1,1,1,-1), kernel_size=(1,3), stride=1, padding=0)
                unfold = unfold.squeeze()
                val, index = unfold.min(dim=0)
                dp_array[h, 1:W-1] += val
                path_array[h, 1:W-1] = index
                dp_array[h, 0] = MAX_VAL
                dp_array[h, W-1] = MAX_VAL
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
        global viz_save_count
        viz_save_count += 1
        img = np.copy(origin_img)
        H, W, C = img.shape
        for j in range(len(seams)):
            for i in range(H):
                img[i, seams[j][i], 0] = 0
                img[i, seams[j][i], 1] = 0
                img[i, seams[j][i], 2] = 255
        cv2.imwrite('seam{}.png'.format(viz_save_count), img)

    @time_log
    def generate_seams(self, img, vis_direction=None, DEBUG=True, **kwargs):
        energy_map = self.energy_func(img.astype(np.uint8)).astype(np.float32)
        cv2.imwrite('/Users/nyz/Desktop/energy_map.png', energy_map)
        print(energy_map.shape)
        seams = self.search_path_torch(energy_map, **kwargs)
        #seams = self.search_path(energy_map, **kwargs)
        if DEBUG:
            assert(vis_direction is not None)
            self.visualize_seam(img, seams, vis_direction)
        return seams

    @time_log
    def delete_seams(self, img, seams, flag_constant=0, DEBUG=True):
        def local_search(arr, idx):
            if arr[idx].any() == flag_constant:
                L = len(arr)
                offset = 1
                while True:
                    if idx+offset < L:
                        if arr[idx+offset].all() != flag_constant:
                            return idx+offset
                    if idx-offset >= 0:
                        if arr[idx-offset].all() != flag_constant:
                            return idx-offset
                    offset += 1
            else:
                return idx

        if len(img.shape) == 3:
            H, W, C = img.shape
        elif len(img.shape) == 2:
            H, W = img.shape
        for item in seams:
            for h in range(H):
                idx = local_search(img[h, :], item[h])
                img[h, idx] = flag_constant
        delete_idx = np.where(img != flag_constant)
        if len(img.shape) == 3:
            img = img[delete_idx].reshape(H, W-len(seams), 3)
        elif len(img.shape) == 2:
            img = img[delete_idx].reshape(H, W-len(seams))

        if DEBUG:
            print(img.shape)
        return img

    @time_log
    def add_seams(self, img, seams, DEBUG=True):
        L = len(seams)
        if len(img.shape) == 3:
            H, W, C = img.shape
            img_expand = np.zeros((H, W+L, 3))
        elif len(img.shape) == 2:
            H, W = img.shape
            img_expand = np.zeros((H, W+L))
        for h in range(H):
            idx = 0
            for i in range(L):
                val = seams[i][h]
                if idx != val:
                    img_expand[h, idx+i:val+i+1] = img[h, idx:val+1]
                img_expand[h, val+i+1] = (img[h, val] + img[h, val+1])/2.0
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


class SeamCarvingRaw(SeamCarving):
    def __init__(self, *args, **kwargs):
        super(SeamCarvingRaw, self).__init__(*args, **kwargs)

    @time_log
    def delete_seams_raw(self, img, seams, flag_constant=0, DEBUG=True):
        H, W = img.shape
        for item in seams:
            for h in range(H//2):
                index = item[h]
                if img[2*h, 2*index] == flag_constant:
                    offset = 0
                    while(img[2*h, 2*(index+offset)] == flag_constant):
                        offset += 1
                    img[2*h, 2*(index+offset)] = flag_constant
                    img[2*h, 2*(index+offset)+1] = flag_constant
                else:
                    img[2*h, 2*index] = flag_constant
                    img[2*h, 2*index+1] = flag_constant
                if img[2*h+1, 2*index] == flag_constant:
                    offset = 0
                    while(img[2*h+1, 2*(index+offset)] == flag_constant):
                        offset += 1
                    img[2*h+1, 2*(index+offset)] = flag_constant
                    img[2*h+1, 2*(index+offset)+1] = flag_constant
                else:
                    img[2*h+1, 2*index] = flag_constant
                    img[2*h+1, 2*index+1] = flag_constant
        delete_idx = np.where(img != flag_constant)
        img = img[delete_idx].reshape(H, W-2*len(seams))
        if DEBUG:
            print(img.shape)
        return img

    @time_log
    def add_seams_raw(self, img, seams, DEBUG=True):
        H, W = img.shape
        L = len(seams)
        img_expand = np.zeros((H, W+2*L))
        for h in range(H//2):
            idx = 0
            repeat_count = 0
            for i in range(L):
                try:
                    val = 2*seams[i][h]
                except IndexError:
                    print('h', h)
                    print('H', H)
                    raise IndexError
                img_expand[2*h, idx+2*i:val+2*i+1] = img[2*h, idx:val+1]
                img_expand[2*h+1, idx+2*i:val+2*i+1] = img[2*h+1, idx:val+1]
                if idx == val:
                    img_expand[2*h, val+2*(i+repeat_count)+1] = (img[2*h, val-1] + img[2*h, val+1])/2.0
                    img_expand[2*h+1, val+2*(i+repeat_count)+1] = (img[2*h+1, val-1] + img[2*h+1, val+1])/2.0
                    img_expand[2*h, val+2*(i+repeat_count)+2] = (img[2*h, val] + img[2*h, val+2])/2.0
                    img_expand[2*h+1, val+2*(i+repeat_count)+2] = (img[2*h+1, val] + img[2*h+1, val+2])/2.0
                    repeat_count += 1
                else:
                    repeat_count = 0
                    img_expand[2*h, val+2*i+1] = (img[2*h, val-1] + img[2*h, val+1])/2.0
                    img_expand[2*h+1, val+2*i+1] = (img[2*h+1, val-1] + img[2*h+1, val+1])/2.0
                    img_expand[2*h, val+2*i+2] = (img[2*h, val] + img[2*h, val+2])/2.0
                    img_expand[2*h+1, val+2*i+2] = (img[2*h+1, val] + img[2*h+1, val+2])/2.0
                idx = val
            img_expand[2*h, idx+2*L+1:] = img[2*h, idx+1:]
            img_expand[2*h+1, idx+2*L+1:] = img[2*h+1, idx+1:]
        if DEBUG:
            print(img_expand.shape)
        return img_expand

    @time_log
    def __call__(self, raw, max_value=1024, check_overlap=False):
        assert(isinstance(raw, np.ndarray))
        raw = raw / max_value * 255 + 0.01
        H, W = raw.shape
        img = torch.from_numpy(raw).view(1,1,H,W)
        gray_scale = F.avg_pool2d(img, kernel_size=2, stride=2).squeeze().numpy()
        img = img.numpy().squeeze()

        misalign = np.copy(gray_scale).astype(np.float32) + 0.01
        vertical_change = np.random.randint(self.vertical_change_range[0], self.vertical_change_range[1])
        horizontal_change = np.random.randint(self.horizontal_change_range[0], self.horizontal_change_range[1])

        # forward(reduce)
        vertical_seams = self.generate_seams(misalign, 'V', min_num=vertical_change, check_overlap=check_overlap)
        misalign = self.delete_seams(misalign, vertical_seams)
        img = self.delete_seams_raw(img, vertical_seams)

        misalign = misalign.transpose(1, 0)
        img = img.transpose(1, 0)
        horizontal_seams = self.generate_seams(misalign, 'H', min_num=horizontal_change, check_overlap=check_overlap)
        misalign = self.delete_seams(misalign, horizontal_seams)
        img = self.delete_seams_raw(img, horizontal_seams)
        img = img.transpose(1, 0)
        misalign = misalign.transpose(1, 0)

        # backward(expand)
        vertical_seams = self.generate_seams(misalign, 'V', min_num=vertical_change, check_overlap=False)
        misalign = self.add_seams(misalign, vertical_seams)
        img = self.add_seams_raw(img, vertical_seams)

        misalign = misalign.transpose(1, 0)
        img = img.transpose(1, 0)
        horizontal_seams = self.generate_seams(misalign, 'H', min_num=horizontal_change, check_overlap=False)
        misalign = self.add_seams(misalign, horizontal_seams)
        img = self.add_seams_raw(img, horizontal_seams)
        img = img.transpose(1, 0)
        misalign = misalign.transpose(1, 0)

        return img / 255 * max_value



def seam_carving_interface(input_path, output_path, DEBUG=True):
    img = cv2.imread(input_path)
    handle_seam_carving = SeamCarving()
    print(img.shape)
    output = handle_seam_carving(img)
    cv2.imwrite(output_path, output)
    if DEBUG:
        sub = img - output
        sub = cv2.cvtColor(sub.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        sub_path = output_path.split('.')[0] + '_sub.' + output_path.split('.')[1]
        cv2.imwrite(sub_path, sub)

def seam_carving_raw_interface(input_np):
    handle_seam_carving = SeamCarvingRaw()
    output_np = handle_seam_carving(input_np)
    return output_np


def test_raw():
    root = '1_raw.npy'
    inputs = np.load(root)
    inputs = inputs[0]
    name = root.split('.')[0]
    N, H, W = inputs.shape
    results = []
    for i in range(inputs.shape[0]):
        gray = F.avg_pool2d(torch.from_numpy(inputs[i]).view(1,1,H,W), kernel_size=2, stride=2).squeeze().numpy()
        print('ori mean:{}'.format(gray.mean()))
        cv2.imwrite(name+'_ori_{}.jpg'.format(i), gray)
        output_np = seam_carving_raw_interface(inputs[i])
        gray = F.avg_pool2d(torch.from_numpy(output_np).view(1,1,H,W), kernel_size=2, stride=2).squeeze().numpy()
        print('ans mean:{}'.format(gray.mean()))
        cv2.imwrite(name+'_ans_{}.jpg'.format(i), gray)
        results.append(output_np)
        print('finish:{}'.format(i))
    output_np = np.stack(results, axis=0)
    print('output_np', output_np.shape)
    np.save(name+"_misalign.npy", output_np)


if __name__ == "__main__":
    #input_path = '/Users/nyz/code/github/seam_carving/data/image_input.png'
    #input_path = 'jun2.jpg'
    #output_path = input_path.split('.')[0]+'_misalign.'+input_path.split('.')[1]
    #seam_carving_interface(input_path, output_path)
    #seam_carving_interface(input_path, output_path)
    #test_raw()
    #input_path = ['/Users/nyz/Desktop/texture_1.jpg', '/Users/nyz/Desktop/texture_2.jpg']
    #output_path = ['/Users/nyz/Desktop/texture_1_output.jpg', '/Users/nyz/Desktop/texture_2_output.jpg']
    input_path = ['/Users/nyz/code/github/burst-deghost-deblur/code/utils/img/real_same/1.jpeg']
    output_path = ['/Users/nyz/code/github/burst-deghost-deblur/code/utils/img/real_same/1_seam1.jpeg']
    for i in range(len(input_path)):
        seam_carving_interface(input_path[i], output_path[i])
        print('finish ', i)
        break
