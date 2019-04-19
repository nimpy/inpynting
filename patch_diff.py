import numpy as np

def patch_diff(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return

    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)


# TODO take into account the mask for comparing
# TODO use codes instead of the pixel values
def non_masked_patch_diff(image, patch_size, x, y, x_compare, y_compare):

    patch_rgb = image.rgb[x: x + patch_size, y: y + patch_size, :]
    patch_compare_rgb = image.rgb[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :]

    # compare just the masked part, which will be on the first patch
    # TODO is this a good way to do this? maybe :D
    mask = image.mask[x: x + patch_size, y: y + patch_size]
    mask_3ch = np.repeat(mask, 3, axis=1).reshape((patch_size, patch_size, 3))
    patch_rgb = patch_rgb * (1 - mask_3ch)
    patch_compare_rgb = patch_compare_rgb * (1 - mask_3ch)

    return np.sum((np.array(patch_rgb, dtype=np.float32) - np.array(patch_compare_rgb, dtype=np.float32)) ** 2)

# TODO?
def half_patch_diff(patch1, patch2, ):
    return -1