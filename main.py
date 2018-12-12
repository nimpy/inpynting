from data_structures import Image2BInpainted, Patch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import efficient_energy_optimization
import eeo
from label_pruning import label_pruning
# import random
# import sys

image = None

patch_size = 0
gap = 0
output_filename = None

THRESHOLD_UNCERTAINTY = 80000 # TODO to be adjusted
MAX_NB_LABELS = 10
MAX_ITERATION_NR = 10

def loading_data():
    global image
    global patch_size
    global gap
    global output_filename

    # inputs
    # folder_path = '/home/niaki/Code/Inpainting_Tijana/images'
    # image_filename = 'Lenna.png'
    # mask_filename = 'Mask512.jpg'

    folder_path = '/home/niaki/Downloads'
    image_filename = 'girl64.jpg'
    mask_filename = 'girl64_mask.png'

    image_inpainted_name = 'Lenna'
    image_inpainted_version = '1st_try'
    patch_size = 16
    gap = 8

    # settings
    np.set_printoptions(threshold=np.nan)

    # loading the image and the mask
    image_rgb = imageio.imread(folder_path + '/' + image_filename)
    mask = imageio.imread(folder_path + '/' + mask_filename)
    mask = mask[:, :, 0]

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


    #mask = mask // 255



    patches = eeo.initialization(image, patch_size, gap, THRESHOLD_UNCERTAINTY)

    eeo.label_pruning(image, patch_size, gap, THRESHOLD_UNCERTAINTY, MAX_NB_LABELS)

    eeo.compute_pairwise_potential_matrix(image, patch_size, gap, MAX_NB_LABELS)

    eeo.compute_label_cost(image, patch_size, MAX_NB_LABELS)

    print('NCSP')
    eeo.neighborhood_consensus_message_passing(image, patch_size, gap, MAX_NB_LABELS, MAX_ITERATION_NR)

    print('generate output')
    eeo.generate_output(image, patch_size)

    plt.imshow(image.inpainted, interpolation='nearest')
    imageio.imwrite(output_filename, image.inpainted)


if __name__ == "__main__":
    main()
