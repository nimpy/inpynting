import numpy as np
import warnings

from data_structures import UP, DOWN, LEFT, RIGHT, opposite_side, get_half_patch_from_patch

def patch_diff(img1, img2):
    """ UPDATED Root mean squared differences as it is normalised to patchsize AND has a physical meaning"""
    """ Deprecated: Computing the sum of squared differences (SSD) between two images."""
    assert img1.shape == img2.shape, "Images don't have the same shape."
    
    return rmse(img1, img2)


def patch_diff_img(image, x1, y1, x2, y2):
    warnings.warn("deprecated function", DeprecationWarning)
    
    patch1_rgb = image.rgb[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
    patch2_rgb = image.rgb[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]

    return patch_diff(patch1_rgb, patch2_rgb)


def patch_diff_ir(image, x1, y1, x2, y2):
    warnings.warn("deprecated function", DeprecationWarning)

    patch1_ir = image.ir[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
    patch2_ir = image.ir[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]

    patch1_descr = max_pool(patch1_ir, image.patch_size)
    patch2_descr = max_pool(patch2_ir, image.patch_size)

    return patch_diff(patch1_descr, patch2_descr)


def patch_diff1(image, x1, y1, x2, y2):
    warnings.warn("deprecated function", DeprecationWarning)
    if image.ir is not None:
        patch1_ir = image.ir[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
        patch2_ir = image.ir[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]

        patch1_descr = max_pool(patch1_ir)
        patch2_descr = max_pool(patch2_ir)

        return patch_diff(patch1_descr, patch2_descr)

    else:
        patch1_rgb = image.rgb[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
        patch2_rgb = image.rgb[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]

        return patch_diff(patch1_rgb, patch2_rgb)


def non_masked_patch_diff(image, x, y, x_compare, y_compare):

    # compare just the masked part, which will be on the first patch
    mask = image.mask[x: x + image.patch_size, y: y + image.patch_size]

    if image.ir is not None:
        patch_ir = image.ir[x: x + image.patch_size, y: y + image.patch_size, :]
        patch_compare_ir = image.ir[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size, :]

        nr_channels = patch_ir.shape[2]
        mask_more_ch = np.repeat(mask, nr_channels, axis=1).reshape((image.patch_size, image.patch_size, nr_channels))
        patch_ir = patch_ir * (1 - mask_more_ch)
        patch_compare_ir = patch_compare_ir * (1 - mask_more_ch)

        patch_descr = max_pool(patch_ir)
        patch_compare_descr = max_pool(patch_compare_ir)

        return patch_diff(patch_descr, patch_compare_descr)

    else:

        patch_rgb = image.rgb[x: x + image.patch_size, y: y + image.patch_size, :]
        patch_compare_rgb = image.rgb[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size, :]

        mask_3ch = np.repeat(mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))
        patch_rgb = patch_rgb * (1 - mask_3ch)
        patch_compare_rgb = patch_compare_rgb * (1 - mask_3ch)

        return patch_diff(patch_rgb, patch_compare_rgb)
    

def half_patch_diff(image, x1, y1, x2, y2, side):
    if image.ir is not None:
        patch1_ir = image.ir[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
        patch2_ir = image.ir[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]

        patch1_ir_half = get_half_patch_from_patch(patch1_ir, image.stride, side)
        patch2_ir_half = get_half_patch_from_patch(patch2_ir, image.stride, opposite_side(side))

        patch1_descr = max_pool(patch1_ir_half)
        patch2_descr = max_pool(patch2_ir_half)

        return patch_diff(patch1_descr, patch2_descr)

    else:
        patch1_rgb = image.rgb[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
        patch2_rgb = image.rgb[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]

        patch1_rgb_half = get_half_patch_from_patch(patch1_rgb, image.stride, side)
        patch2_rgb_half = get_half_patch_from_patch(patch2_rgb, image.stride, opposite_side(side))

        return patch_diff(patch1_rgb_half, patch2_rgb_half)


def max_pool(patch_ir, pool_size=8):

    height, width, nr_channels = patch_ir.shape

    patch_ir_reshaped = patch_ir.reshape(height // pool_size, pool_size,
                           width // pool_size, pool_size, nr_channels)
    patch_descr = patch_ir_reshaped.max(axis=1).max(axis=2)

    return patch_descr


def max_pool_pad_full_process(patch_ir, pool_size=8):

    height, width, nr_channels = patch_ir.shape

    padding_height_total = pool_size - (height % pool_size)
    padding_width_total = pool_size - (width % pool_size)

    padding_height_left = padding_height_total // 2
    padding_height_right = padding_height_total - padding_height_left

    padding_width_left = padding_width_total // 2
    padding_width_right = padding_width_total - padding_width_left

    patch_ir = np.pad(patch_ir, ((padding_height_left, padding_height_right), (padding_width_left, padding_width_right),
                                 (0, 0)), mode='constant')

    height += padding_height_total
    width += padding_width_total

    patch_ir_reshaped = patch_ir.reshape(height // pool_size, pool_size,
                           width // pool_size, pool_size, nr_channels)

    patch_descr = patch_ir_reshaped.max(axis=1).max(axis=2)

    return patch_descr


def max_pool_padding(patch_ir, padding_height_left, padding_height_right, padding_width_left, padding_width_right, pool_size=8):

    height, width, nr_channels = patch_ir.shape

    patch_ir = np.pad(patch_ir, ((padding_height_left, padding_height_right), (padding_width_left, padding_width_right),
                                 (0, 0)), mode='constant')

    height += padding_height_left + padding_height_right
    width += padding_width_left + padding_width_right

    patch_ir_reshaped = patch_ir.reshape(height // pool_size, pool_size,
                                         width // pool_size, pool_size, nr_channels)

    patch_descr = patch_ir_reshaped.max(axis=1).max(axis=2)

    return patch_descr


def rmse(a, b):
    # Normalised and has physical meaning
    return np.sqrt(np.mean(np.subtract(a, b, dtype=float)**2))
