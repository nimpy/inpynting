import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import datetime
# import random
# import sys

from data_structures import Image2BInpainted
import eeo





def loading_data(folder_path, image_filename, mask_filename):

    image_inpainted_name, _ = os.path.splitext(image_filename)
    image_inpainted_name = image_inpainted_name + '_'

    # settings
    # np.set_printoptions(threshold=np.nan)

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

    return image, image_inpainted_name


def inpaint_image(folder_path, image_filename, mask_filename, thresh_uncertainty, max_nr_labels, max_nr_iterations):


    image, image_inpainted_name = loading_data(folder_path, image_filename, mask_filename)
    image_inpainted_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_threshUncert" + str(
        thresh_uncertainty)

    print(image.patch_size)
    print(image.stride)

    
    plt.imshow(image.rgb, interpolation='nearest')
    plt.show()
    plt.imshow(image.mask, cmap='gray')
    plt.show()

    print("Number of pixels to be inpainted: " + str(np.count_nonzero(image.mask)))

    # already done and pickled - start

    print()
    print("... Initialization ...")
    eeo.initialization(image, thresh_uncertainty)

    eeo.pickle_global_vars(image_inpainted_name + eeo.initialization.__name__)

    print()
    print("... Label pruning ...")
    eeo.label_pruning(image, thresh_uncertainty, max_nr_labels)

    eeo.pickle_global_vars(image_inpainted_name + eeo.label_pruning.__name__)

    print()
    print("... Computing pairwise potential matrix ...")
    eeo.compute_pairwise_potential_matrix(image, max_nr_labels)

    eeo.pickle_global_vars(image_inpainted_name + eeo.compute_pairwise_potential_matrix.__name__)

    print()
    print("... Computing label cost ...")
    eeo.compute_label_cost(image, max_nr_labels)

    eeo.pickle_global_vars(image_inpainted_name + eeo.compute_label_cost.__name__)

    print()
    print("... Neighborhood consensus message passing ...")
    eeo.neighborhood_consensus_message_passing(image, max_nr_labels, max_nr_iterations)

    eeo.pickle_global_vars(image_inpainted_name + eeo.neighborhood_consensus_message_passing.__name__)

    # already done and pickled - end

    # eeo.unpickle_global_vars(image_inpainted_name + eeo.neighborhood_consensus_message_passing.__name__)

    print()
    print("... Generating inpainted image ...")
    eeo.generate_inpainted_image(image)

    print()
    print("... Generating order image ...")
    eeo.generate_order_image(image)

    filename_inpainted = folder_path + '/' + image_inpainted_name + image_inpainted_version + '.jpg'
    filename_order_image = folder_path + '/' + image_inpainted_name + 'orderimg_' + image_inpainted_version + '.jpg'
    

    imageio.imwrite(filename_inpainted, image.inpainted)
    plt.imshow(image.inpainted, interpolation='nearest')
    plt.show()

    imageio.imwrite(filename_order_image, image.order_image)
    plt.imshow(image.order_image, cmap='gray')
    plt.show()

    print(image.patch_size)
    print(image.stride)


def main():
    
    # inputs

    thresh_uncertainty = 155360  # 100000 #155360 #255360 #6755360 # TODO to be adjusted
    max_nr_labels = 10
    max_nr_iterations = 10
    
    # folder_path = '/home/niaki/Code/inpynting_images/Lenna'
    # image_filename = 'Lenna.png'
    # mask_filename = 'Mask512.jpg'
    # mask_filename = 'Mask512_3.png'

    # folder_path = '/home/niaki/Code/inpynting_images/Greenland'
    # image_filename = 'Greenland.jpg'
    # mask_filename = 'Mask512.jpg'

    # folder_path = '/home/niaki/Downloads'
    # image_filename = 'building64.jpg'
    # mask_filename = 'girl64_mask.png'

    folder_path = '/home/niaki/Code/inpynting_images/building'
    image_filename = 'building128.jpeg'
    mask_filename = 'mask128.jpg'

    
    
    inpaint_image(folder_path, image_filename, mask_filename, thresh_uncertainty, max_nr_labels, max_nr_iterations)


if __name__ == "__main__":
    main()
