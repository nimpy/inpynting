from data_structures import Image2BInpainted, Patch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import datetime

import efficient_energy_optimization
import eeo
from label_pruning import label_pruning
# import random
# import sys

image = None

patch_size = 0
gap = 0
output_filename = None

THRESHOLD_UNCERTAINTY = 6755360 #6755360 # TODO to be adjusted
MAX_NB_LABELS = 10
MAX_ITERATION_NR = 10

def loading_data():
    global image
    global patch_size
    global gap
    global output_filename

    # inputs
    folder_path = '/home/niaki/Code/inpynting_images/Lenna'
    image_filename = 'Lenna.png'
    mask_filename = 'Mask512.jpg'
    # # mask_filename = 'Mask512_3.png'

    # folder_path = '/home/niaki/Downloads'
    # image_filename = 'building64.jpg'
    # mask_filename = 'girl64_mask.png'

    # folder_path = '/home/niaki/Code/inpynting_images/building'
    # image_filename = 'building128.jpeg'
    # mask_filename = 'mask128.jpg'

    image_inpainted_name, _ = os.path.splitext(image_filename)
    image_inpainted_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_threshUncert" + str(THRESHOLD_UNCERTAINTY)

    # settings
    np.set_printoptions(threshold=np.nan)
    patch_size = 16
    gap = 8

    # loading the image and the mask
    image_rgb = imageio.imread(folder_path + '/' + image_filename)

    mask = imageio.imread(folder_path + '/' + mask_filename)
    mask = mask[:, :, 0]
    mask = mask / 255
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i,j] = round(mask[i,j])

    # on the image: set everything that's under the mask to white
    for i in range(image_rgb.shape[0]):
        for j in range(image_rgb.shape[1]):
            for k in range(image_rgb.shape[2]):
                if mask[i,j] == 1:
                    image_rgb[i, j, k] = 0


    image = Image2BInpainted(image_rgb, mask)

    output_filename = folder_path + '/' + image_inpainted_name + '_' + image_inpainted_version + '.jpg'


def main():

    global image
    global patch_size
    global gap
    global output_filename

    loading_data()

    plt.imshow(image.rgb, interpolation='nearest')
    plt.show()
    plt.imshow(image.mask, cmap='gray')
    plt.show()

    print("Number of pixels to be inpainted: " + str(np.count_nonzero(image.mask)))

    print()
    print("... Initialization ...")
    eeo.initialization(image, patch_size, gap, THRESHOLD_UNCERTAINTY)

    print()
    print("... Label pruning ...")
    eeo.label_pruning(image, patch_size, gap, THRESHOLD_UNCERTAINTY, MAX_NB_LABELS)

    print()
    print("... Computing pairwise potential matrix ...")
    eeo.compute_pairwise_potential_matrix(image, patch_size, gap, MAX_NB_LABELS)

    print()
    print("... Computing label cost ...")
    eeo.compute_label_cost(image, patch_size, MAX_NB_LABELS)

    print()
    print("... Neighborhood consensus message passing ...")
    eeo.neighborhood_consensus_message_passing(image, patch_size, gap, MAX_NB_LABELS, MAX_ITERATION_NR)

    print()
    print("... Generating inpainted image ...")
    eeo.generate_inpainted_image(image, patch_size)

    imageio.imwrite(output_filename, image.inpainted)
    plt.imshow(image.inpainted, interpolation='nearest')
    plt.show()




if __name__ == "__main__":
    main()
