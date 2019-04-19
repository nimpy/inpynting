from data_structures import Patch, Image2BInpainted, coordinates_to_position, UP, DOWN, LEFT, RIGHT, get_half_patch_from_patch
from eeo import generate_smooth_filter, generate_blend_mask

import numpy as np
from scipy import signal

def testing_prune_labels():
    patch = Patch(0, False, False, 0, 0, 0, [1, 2, 3, 4, 5], None, {1: 7, 2: 9, 3: 6, 4: 8, 5: 10}, False) # 3 1 4 2 5
    patch.prune_labels(1)
    print(patch.pruned_labels)

def testing_get_neighbor_position():
    patch = Patch(31, False, False, 12, 0)
    image = Image2BInpainted(np.zeros((16, 16, 3)), np.zeros((16, 16, 3)))
    print(patch.get_down_neighbor_position(image, 4, 1))


def testing_coordinates_to_position():
    x = 1
    y = 1
    image_width = 16
    patch_size = 4
    stride = 1
    pos = coordinates_to_position(x, y, image_width, patch_size, stride)
    print(pos)

def change_patch(patch):
    patch.differences[1] = 0

def testing_change_patch():
    patch = Patch(0, False, False, 0, 0, 0, [1, 2, 3, 4, 5], None, {1: 7, 2: 9, 3: 6, 4: 8, 5: 10}, False) # 3 1 4 2 5
    change_patch(patch)
    print(patch.differences[1])

def testing_smooth_filter_blend_mask_convolve():
    filter_size = 4  # should be > 1
    patch_size = 8

    smooth_filter = generate_smooth_filter(filter_size)
    blend_mask = generate_blend_mask(patch_size)

    print(smooth_filter.shape)
    print(blend_mask.shape)

    blend_mask = signal.convolve2d(blend_mask, smooth_filter, boundary='symm', mode='same')


    print(blend_mask)

def testing_OOP():
    patches_list = []

    for i in range(5):
        p = Patch(i, True, True, i, i)
        patches_list.append(p)

    for i in range(5):
        print(patches_list[i].patch_id)


def testing_half_patch():
    patch_size = 16
    temp = np.arange(0, patch_size**2).reshape((patch_size, patch_size))
    print(temp)
    temp3ch = np.repeat(temp, 3, axis=1).reshape((patch_size, patch_size, 3))
    print()
    print(get_half_patch_from_patch(temp3ch, RIGHT)[:,:,0])

def main():
    # testing_prune_labels()
    print("---")
    # testing_get_neighbor_position()
    print("---")
    # testing_coordinates_to_position()
    print("---")
    # testing_change_patch()
    print("---")
    # generate_smooth_filter(4)
    print("---")
    # testing_smooth_filter_blend_mask_convolve()
    print("---")
    # testing_OOP()
    print("---")
    testing_half_patch()

if __name__ == "__main__":
    main()