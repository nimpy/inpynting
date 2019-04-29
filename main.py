import numpy as np
# import matplotlib.pyplot as plt
import imageio
import os
import datetime
# import random
# import sys

from data_structures import Image2BInpainted
import eeo





def loading_data(folder_path, image_filename, mask_filename, patch_size, stride):

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


    image = Image2BInpainted(image_rgb, mask, patch_size=patch_size, stride=stride)

    return image, image_inpainted_name


def inpaint_image(folder_path, image_filename, mask_filename, patch_size, stride, thresh_uncertainty, max_nr_labels, max_nr_iterations):

    image, image_inpainted_name = loading_data(folder_path, image_filename, mask_filename, patch_size, stride)
    image_inpainted_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(patch_size) + "_" + str(stride) + "_" + str(thresh_uncertainty) + "_" + str(max_nr_labels) + "_" + str(max_nr_iterations)

    # plt.imshow(image.rgb, interpolation='nearest')
    # plt.show()
    # plt.imshow(image.mask, cmap='gray')
    # plt.show()

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
    # plt.imshow(image.inpainted, interpolation='nearest')
    # plt.show()

    imageio.imwrite(filename_order_image, image.order_image)
    # plt.imshow(image.order_image, cmap='gray')
    # plt.show()


def main():

    # TODO thresh_uncertainty should maybe be related to the patch size relative to the image size
    # inputs
    patch_size = 10
    stride = patch_size // 2 #TODO fix problem when stride isn't exactly half of patch size!
    thresh_uncertainty = 150000 #155360 # 6755360  #155360  # 100000 #155360 #255360 #6755360 # TODO to be adjusted
    max_nr_labels = 10
    max_nr_iterations = 10
    
    folder_path = '/home/niaki/Code/inpynting_images/Lenna'
    image_filename = 'Lenna.png'
    mask_filename = 'Mask512.jpg'
    # mask_filename = 'Mask512_3.png'

    # folder_path = '/home/niaki/Code/inpynting_images/Greenland'
    # image_filename = 'Greenland.jpg'
    # mask_filename = 'Mask512.jpg'

    # folder_path = '/home/niaki/Downloads'
    # image_filename = 'building64.jpg'
    # mask_filename = 'girl64_mask.png'

    # folder_path = '/home/niaki/Code/inpynting_images/building'
    # image_filename = 'building128.jpeg'
    # mask_filename = 'mask128.jpg'

    jian_number = '10'
    folder_path = '/home/niaki/Code/inpynting_images/Tijana/Jian' + jian_number #+ '_square'
    image_filename = 'Jian' + jian_number + '_degra.png'
    mask_filename = 'Jian' + jian_number + 'Mask_inverted.png'

    
    inpaint_image(folder_path, image_filename, mask_filename, patch_size, stride, thresh_uncertainty, max_nr_labels, max_nr_iterations)


    #####

    folder_path_origin = '/home/niaki/Code/inpynting_images'
    folder_path_subfolders = ['Lenna', 'Greenland']#, 'Waterfall']
    image_filenames = ['Lenna.png', 'Greenland.jpg']#, 'Waterfall.jpg']

    mask_filenames = ['Mask512_1.jpg', 'Mask512_2.png', 'Mask512_3.png']

    patch_size_values = [10, 16, 20]
    # stride_values = [4, 6, 8, 10]
    thresh_uncertainty_values = [150000, 3450000, 6750000, 13500000]
    max_nr_labels_values = [10]
    max_nr_iterations_values = [10]

    # counter = 0
    # for i, folder_path_subfolder in enumerate(folder_path_subfolders):
    #     folder_path = folder_path_origin + '/' + folder_path_subfolder
    #     image_filename = image_filenames[i]
    #
    #     for mask_filename in mask_filenames:
    #         for patch_size in patch_size_values:
    #             stride = patch_size // 2
    #             for thresh_uncertainty in thresh_uncertainty_values:
    #                 for max_nr_labels in max_nr_labels_values:
    #                     for max_nr_iterations in max_nr_iterations_values:
    #                         print('*****************************************************')
    #                         print(folder_path)
    #                         print(image_filename)
    #                         print(mask_filename)
    #                         print(patch_size)
    #                         print(stride)
    #                         print(thresh_uncertainty)
    #                         print(max_nr_labels)
    #                         print(max_nr_iterations)
    #                         print(counter)
    #                         try:
    #                             eeo.patches = []
    #                             eeo.nodes_count = 0
    #                             eeo.nodes_order = []
    #                             inpaint_image(folder_path, image_filename, mask_filename, patch_size, stride,
    #                                       thresh_uncertainty, max_nr_labels, max_nr_iterations)
    #                         except Exception as e:
    #                             print("Problem!!", str(e))
    #
    #                         print(counter)
    #                         counter += 1
    #
    # print(counter)


if __name__ == "__main__":
    main()
