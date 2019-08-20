import numpy as np

from .data_structures import UP, DOWN, LEFT, RIGHT, opposite_side, get_half_patch_from_patch

def patch_diff(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return

    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)


# def patch_diff_img(image, x1, y1, x2, y2):
#     patch1_rgb = image.rgb[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
#     patch2_rgb = image.rgb[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]
#
#     return np.sum((np.array(patch1_rgb, dtype=np.float32) - np.array(patch2_rgb, dtype=np.float32)) ** 2)
#
#
# def patch_diff_ir(image, x1, y1, x2, y2):
#     patch1_ir = image.ir[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
#     patch2_ir = image.ir[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]
#
#     patch1_descr = max_pool(patch1_ir, image.patch_size)
#     patch2_descr = max_pool(patch2_ir, image.patch_size)
#
#     return np.sum((np.array(patch1_descr, dtype=np.float32) - np.array(patch2_descr, dtype=np.float32)) ** 2)


# def patch_diff1(image, x1, y1, x2, y2):
#     if image.ir is not None:
#         patch1_ir = image.ir[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
#         patch2_ir = image.ir[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]
#
#         patch1_descr = max_pool(patch1_ir)
#         patch2_descr = max_pool(patch2_ir)
#
#         return np.sum((np.array(patch1_descr, dtype=np.float32) - np.array(patch2_descr, dtype=np.float32)) ** 2)
#
#     else:
#         patch1_rgb = image.rgb[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
#         patch2_rgb = image.rgb[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]
#
#         return np.sum((np.array(patch1_rgb, dtype=np.float32) - np.array(patch2_rgb, dtype=np.float32)) ** 2)


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

        return np.sum((np.array(patch_descr, dtype=np.float32) - np.array(patch_compare_descr, dtype=np.float32)) ** 2)

    else:

        patch_rgb = image.rgb[x: x + image.patch_size, y: y + image.patch_size, :]
        patch_compare_rgb = image.rgb[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size, :]

        mask_3ch = np.repeat(mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))
        patch_rgb = patch_rgb * (1 - mask_3ch)
        patch_compare_rgb = patch_compare_rgb * (1 - mask_3ch)

        return np.sum((np.array(patch_rgb, dtype=np.float32) - np.array(patch_compare_rgb, dtype=np.float32)) ** 2)


def half_patch_diff(image, x1, y1, x2, y2, side):
    if image.ir is not None:
        patch1_ir = image.ir[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
        patch2_ir = image.ir[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]

        patch1_ir_half = get_half_patch_from_patch(patch1_ir, image.stride, side)
        patch2_ir_half = get_half_patch_from_patch(patch2_ir, image.stride, opposite_side(side))

        patch1_descr = max_pool(patch1_ir_half)
        patch2_descr = max_pool(patch2_ir_half)

        return np.sum((np.array(patch1_descr, dtype=np.float32) - np.array(patch2_descr, dtype=np.float32)) ** 2)

    else:
        patch1_rgb = image.rgb[x1: x1 + image.patch_size, y1: y1 + image.patch_size, :]
        patch2_rgb = image.rgb[x2: x2 + image.patch_size, y2: y2 + image.patch_size, :]

        patch1_rgb_half = get_half_patch_from_patch(patch1_rgb, image.stride, side)
        patch2_rgb_half = get_half_patch_from_patch(patch2_rgb, image.stride, opposite_side(side))

        return np.sum((np.array(patch1_rgb_half, dtype=np.float32) - np.array(patch2_rgb_half, dtype=np.float32)) ** 2)


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
