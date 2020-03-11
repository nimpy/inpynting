import numpy as np
import sys
import matplotlib.pyplot as plt
import imageio
import os
import datetime
# import random
# import sys
import ae_descriptor

from data_structures import Image2BInpainted, coordinates_to_position
import eeo


def loading_data(folder_path, image_filename, mask_filename, patch_size, stride, use_descriptors,
                 store_descriptors_halves, store_descriptors_cube,
                 mask_thresh=254, b_debug=False):
    """

    :param mask_thresh: value between 0 and 255 where a high values means the mask has to be extremely certain before it is inpainted.
    :return:
    """

    image_inpainted_name, _ = os.path.splitext(image_filename)
    image_inpainted_name = image_inpainted_name + '_'

    # settings
    # np.set_printoptions(threshold=np.nan)

    # loading the image and the mask
    image_rgb = imageio.imread(folder_path + '/' + image_filename)

    mask = imageio.imread(folder_path + '/' + mask_filename)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask = np.greater_equal(mask, mask_thresh).astype(np.uint8)

    # on the image: set everything that's under the mask to cyan (for debugging purposes
    cyan = [0, 0, 0]  # [0, 255, 255]  # [127, 127, 127]  # [0, 255, 255]
    image_rgb[mask.astype(bool), :] = cyan

    if b_debug:
        import matplotlib.pyplot as plt
        plt.imshow(image_rgb)
        plt.show()

    edges = imageio.imread(folder_path + '/' + image_filename[:image_filename.find('.')] + '_edges.png')

    image = Image2BInpainted(image_rgb, mask, edges, patch_size=patch_size, stride=stride)

    image.inpainting_approach = Image2BInpainted.USING_RBG_VALUES

    if use_descriptors:

        if store_descriptors_cube:

            image.inpainting_approach = Image2BInpainted.USING_STORED_DESCRIPTORS_CUBE
            encoder_whole_patch = ae_descriptor.init_descr_128(image.patch_size, image.patch_size)
            encoder_landscape_half_patch = ae_descriptor.init_descr_128(image.patch_size // 2, image.patch_size)
            encoder_portrait_half_patch = ae_descriptor.init_descr_128(image.patch_size, image.patch_size // 2)

            # calculating the dimensionalities of the descriptors in order to initialise the cubes
            patch_0 = image.rgb[0: 0 + image.patch_size, 0: 0 + image.patch_size, :]
            patch_half_landscape_0 = image.rgb[0: 0 + image.patch_size // 2, 0: 0 + image.patch_size, :]
            patch_half_portrait_0 = image.rgb[0: 0 + image.patch_size, 0: 0 + image.patch_size // 2, :]
            dimensionality_patch_descr = ae_descriptor.compute_descriptor(patch_0, encoder_whole_patch).flatten().shape[0]
            dimensionality_patch_descr_half_landscape = ae_descriptor.compute_descriptor(patch_half_landscape_0, encoder_landscape_half_patch).flatten().shape[0]
            dimensionality_patch_descr_half_portrait = ae_descriptor.compute_descriptor(patch_half_portrait_0, encoder_portrait_half_patch).flatten().shape[0]

            image.descriptor_cube = np.zeros((image.height - image.patch_size + 1, image.width - image.patch_size + 1, dimensionality_patch_descr), dtype=np.float32)
            image.half_patch_landscape_descriptor_cube = np.zeros((image.height - image.stride + 1, image.width - image.patch_size + 1, dimensionality_patch_descr_half_landscape), dtype=np.float32)
            image.half_patch_portrait_descriptor_cube = np.zeros((image.height - image.patch_size + 1, image.width - image.stride + 1, dimensionality_patch_descr_half_portrait), dtype=np.float32)

            count = 0
            total_count = len(range(0, image.width - image.patch_size + 1)) * len(range(0, image.height - image.patch_size + 1)) + \
                          len(range(0, image.width - image.patch_size + 1)) * len(range(0, image.height - image.stride + 1)) + \
                          len(range(0, image.width - image.stride + 1)) * len(range(0, image.height - image.patch_size + 1))

            print(len(range(0, image.width - image.patch_size + 1)) * len(range(0, image.height - image.patch_size + 1)))
            print(len(range(0, image.width - image.patch_size + 1)) * len(range(0, image.height - image.stride + 1)))
            print(len(range(0, image.width - image.stride + 1)) * len(range(0, image.height - image.patch_size + 1)))

            raise NotImplementedError(
                "The other parts of the inpainting are not (yet) implemented to work with descriptors cube.")

            # # compute descriptors for the whole patches
            # for y in range(0, image.width - image.patch_size + 1):
            #     for x in range(0, image.height - image.patch_size + 1):
            #         sys.stdout.write("\rComputing descriptor " + str(count + 1) + "/" + str(total_count))
            #         count += 1
            #
            #         patch = image.rgb[x: x + image.patch_size, y: y + image.patch_size, :]
            #
            #         patch_descr = ae_descriptor.compute_descriptor(patch, encoder_whole_patch)
            #         patch_descr = patch_descr.flatten()
            #
            #         image.descriptor_cube[x, y, :] = patch_descr
            #
            # # compute descriptors for the half patches (landscape)
            # for y in range(0, image.width - image.patch_size + 1):
            #     for x in range(0, image.height - image.stride + 1):
            #         sys.stdout.write("\rComputing descriptor " + str(count + 1) + "/" + str(total_count))
            #         count += 1
            #
            #         patch_half_landscape = image.rgb[x: x + image.patch_size // 2, y: y + image.patch_size, :]
            #
            #         patch_descr_half_landscape = ae_descriptor.compute_descriptor(patch_half_landscape, encoder_landscape_half_patch)
            #         patch_descr_half_landscape = patch_descr_half_landscape.flatten()
            #
            #         image.half_patch_landscape_descriptor_cube[x, y, :] = patch_descr_half_landscape
            #
            # # compute descriptors for the half patches (portrait)
            # for y in range(0, image.width - image.stride + 1):
            #     for x in range(0, image.height - image.patch_size + 1):
            #         sys.stdout.write("\rComputing descriptor " + str(count + 1) + "/" + str(total_count))
            #         count += 1
            #
            #         patch_half_portrait = image.rgb[x: x + image.patch_size, y: y + image.patch_size // 2, :]
            #
            #         patch_descr_half_portrait = ae_descriptor.compute_descriptor(patch_half_portrait, encoder_portrait_half_patch)
            #         patch_descr_half_portrait = patch_descr_half_portrait.flatten()
            #
            #         image.half_patch_portrait_descriptor_cube[x, y, :] = patch_descr_half_portrait
            #
            # sys.stdout.write("\rComputing descriptor " + str(total_count) + "/" + str(total_count) + " ... Done! \n")

        else:

            image.inpainting_approach = Image2BInpainted.USING_IR

            # compute the intermediate representation, from which descriptors for a single patch can be easily computed

            # encoder_ir, _ = ae_descriptor.init_IR_128(image.height, image.width, image.patch_size)
            encoder_ir, _ = ae_descriptor.init_IR(image.height, image.width, image.patch_size,
                            model_version='128', nr_feature_maps_layer1=32,
                            nr_feature_maps_layer23=32) #128 or 16_alex_layer1finetuned_2_finetuned_3conv3mp_test_images_inpainting
            # TODO check if the image is normalised (divided by 255), and check if data types are causing problems
            ir = ae_descriptor.compute_IR(image.rgb / 255, encoder_ir)
            image.ir = ir
            image.inverted_mask_Nch = 1 - np.repeat(mask, image.ir.shape[2], axis=1).reshape((image.height, image.width, image.ir.shape[2]))

            if store_descriptors_halves:

                print()
                print("... Computing descriptors ...")

                image.inpainting_approach = Image2BInpainted.USING_STORED_DESCRIPTORS_HALVES

                # compute a descriptor for all the half-patches and store it in image object

                encoder_landscape_half_patch = ae_descriptor.init_descr_128(image.patch_size // 2, image.patch_size)
                encoder_portrait_half_patch = ae_descriptor.init_descr_128(image.patch_size, image.patch_size // 2)
                image.half_patch_landscape_descriptors = {}
                image.half_patch_portrait_descriptors = {}

                count = 0
                total_count = len(range(0, image.width - image.patch_size + 1)) * len(range(0, image.height - image.stride + 1)) + \
                              len(range(0, image.width - image.stride + 1)) * len(range(0, image.height - image.patch_size + 1))

                print(len(range(0, image.width - image.patch_size + 1)) * len(range(0, image.height - image.stride + 1)))
                print(len(range(0, image.width - image.stride + 1)) * len(range(0, image.height - image.patch_size + 1)))

                for y in range(0, image.width - image.patch_size + 1):
                    for x in range(0, image.height - image.stride + 1):

                        sys.stdout.write("\rComputing descriptor " + str(count + 1) + "/" + str(total_count))
                        count += 1

                        patch_half_landscape = image.rgb[x: x + image.patch_size // 2, y: y + image.patch_size, :]

                        patch_descr_half_landscape = ae_descriptor.compute_descriptor(patch_half_landscape, encoder_landscape_half_patch)

                        position = coordinates_to_position(x, y, image.height, image.stride)
                        image.half_patch_landscape_descriptors[position] = patch_descr_half_landscape

                for y in range(0, image.width - image.stride + 1):
                    for x in range(0, image.height - image.patch_size + 1):

                        sys.stdout.write("\rComputing descriptor " + str(count + 1) + "/" + str(total_count))
                        count += 1

                        patch_half_portrait = image.rgb[x: x + image.patch_size, y: y + image.patch_size // 2, :]

                        patch_descr_half_portrait = ae_descriptor.compute_descriptor(patch_half_portrait, encoder_portrait_half_patch)

                        position = coordinates_to_position(x, y, image.height, image.patch_size)
                        image.half_patch_portrait_descriptors[position] = patch_descr_half_portrait

                sys.stdout.write("\rComputing descriptor " + str(total_count) + "/" + str(total_count) + " ... Done! \n")

    return image, image_inpainted_name


def inpaint_image(folder_path, image_filename, mask_filename, patch_size, stride, thresh_uncertainty:int,
                  max_nr_labels, max_nr_iterations, use_descriptors, store_descriptors_halves, store_descriptors_cube,
                  mask_thresh:int=254, b_debug=False):

    """

    :param: thresh_uncertainty: thresh for distance function. Value in [0.; 1.] or [0;255]
    :param: mask_thresh: value between 0 and 255 where a high values means the mask has to be extremely certain before it is inpainted.
    :return:
    """

    assert thresh_uncertainty <= 255, f'thresh_uncertainty = {thresh_uncertainty}. Should be in [0.; 1.] or [0;255]'
    if thresh_uncertainty <= 1:     # Transform to working in int8 domain
        thresh_uncertainty = thresh_uncertainty*255

    image, image_inpainted_name = loading_data(folder_path, image_filename, mask_filename, patch_size, stride,
                                               use_descriptors, store_descriptors_halves, store_descriptors_cube,
                                               mask_thresh=mask_thresh, b_debug=b_debug)
    image_inpainted_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(patch_size) + "_" + str(stride) + "_" + str(thresh_uncertainty) + "_" + str(max_nr_labels) + "_" + str(max_nr_iterations)
    if use_descriptors:
        image_inpainted_version += '_descr'
        if store_descriptors_halves:
            image_inpainted_version += '_storedhalves'
        if store_descriptors_cube:
            image_inpainted_version += '_storedcube'

    # plt.imshow(image.rgb, interpolation='nearest')
    # plt.show()
    # plt.imshow(image.mask, cmap='gray')
    # plt.show()

    print()
    print("Number of pixels to be inpainted: " + str(np.count_nonzero(image.mask)))


    # already done and pickled - start

    print()
    print("... Initialization ...")
    # eeo.initialization(image, thresh_uncertainty, max_nr_labels)

    # eeo.pickle_global_vars(image_inpainted_name + eeo.initialization.__name__)
    eeo.unpickle_global_vars(image_inpainted_name + eeo.initialization.__name__)

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

    filename_inpainted = folder_path + '/' + image_inpainted_name + image_inpainted_version + '.png'
    filename_order_image = folder_path + '/' + image_inpainted_name + 'orderimg_' + image_inpainted_version + '.png'


    imageio.imwrite(filename_inpainted, image.inpainted)
    plt.imshow(image.inpainted, interpolation='nearest')
    plt.show()

    imageio.imwrite(filename_order_image, image.order_image)
    # plt.imshow(image.order_image, cmap='gray')
    # plt.show()


def main():
    # TODO also taking into account whether the descripotrs are used
    # inputs
    patch_size = 16  # needs to be an even number
    stride = patch_size // 2 #TODO fix problem when stride isn't exactly half of patch size!
    thresh_uncertainty = 120 #40 #5555360 #35360 #85360 #155360 # 6755360  #155360  # 100000 #155360 #255360 #6755360
    max_nr_labels = 10
    max_nr_iterations = 10

    # valid states of these variables:
    #  if use_descriptors is False, then other two should be False
    #  if use_descriptors is True, then at most one other can be True (and at least zero :D)
    use_descriptors = True
    store_descriptors_halves = True
    store_descriptors_cube = False  # TODO implemented this

    folder_path = '/home/niaki/Code/inpynting_images/Lenna'
    image_filename = 'Lenna.png'
    mask_filename = 'Mask512.png'
    # mask_filename = 'Mask512_3.png'

    # folder_path = '/home/niaki/Code/inpynting_images/Greenland'
    # image_filename = 'Greenland.png'
    # mask_filename = 'Mask512.png'

    # folder_path = '/home/niaki/Code/inpynting_images/Gradient'
    # image_filename = 'Gradient.png'
    # mask_filename = 'Mask256_2.png'

    # folder_path = '/home/niaki/Downloads'
    # image_filename = 'building64.png'
    # mask_filename = 'girl64_mask.png'

    folder_path = '/home/niaki/Code/inpynting_images/building'
    image_filename = 'building128.jpeg'
    mask_filename = 'mask128.jpg' # 'mask128.png' 'mask128_ULcorner.png'

    folder_path = '/home/niaki/Code/inpynting_images/Colours'
    image_filename = 'beach_small_degra1.bmp'
    mask_filename = 'beach_small_mask1.bmp'  # 'mask128.png' 'mask128_ULcorner.png'

    jian_number = '11'
    folder_path = '/home/niaki/Code/inpynting_images/Tijana/Jian' + jian_number + '_uint8_downscaled'
    image_filename = 'Jian_small_' + jian_number + '_degra2.png'
    mask_filename = 'Jian_small_' + jian_number + 'Mask_inverted2.png'


    jian_number = '8'
    folder_path = '/home/niaki/Code/inpynting_images/Tijana/Jian' + jian_number + '_uint8'
    image_filename = 'Jian' + jian_number + '_degra.png'
    mask_filename = 'Jian' + jian_number + 'Mask_inverted.png'

    # folder_path = '/scratch/data/hand'  # don't forget to also change the descriptor
    # image_filename = 'clean.tif'
    # mask_filename = 'pred.tif'

    # folder_path = '/scratch/data/panel13'  # don't forget to also change the descriptor
    # image_filename = 'panel13_cropped1.png'
    # mask_filename = 'panel13_mask_cropped1.png'


    inpaint_image(folder_path, image_filename, mask_filename, patch_size, stride, thresh_uncertainty,
                  max_nr_labels, max_nr_iterations, use_descriptors, store_descriptors_halves, store_descriptors_cube)

    #####

    # folder_path = '/scratch/data/panel13/crops1'
    #
    # crop_height = 389
    # crop_width = 406
    #
    # for i in range(0, 1945, crop_height):
    #     for j in range(0, 1218, crop_width):
    #         print("=================================================")
    #         print(i, j)
    #         image_filename = "image_" + str(i) + "_" + str(j) + ".tif"
    #         mask_filename = "mask_" + str(i) + "_" + str(j) + ".png"
    #
    #         inpaint_image(folder_path, image_filename, mask_filename, patch_size, stride, thresh_uncertainty,
    #                       max_nr_labels, max_nr_iterations, use_descriptors, store_descriptors_halves)
    #
    #         print()
    #         print()



    #####

    # folder_path_origin = '/home/niaki/Code/inpynting_images'
    # folder_path_subfolders = ['Lenna', 'Greenland']  # , 'Waterfall']
    # image_filenames = ['Lenna.png', 'Greenland.png']  # , 'Waterfall.png']
    #
    # mask_filenames = ['Mask512_1.png', 'Mask512_2.png', 'Mask512_3.png']
    #
    # patch_size_values = [10, 16, 20]
    # # stride_values = [4, 6, 8, 10]
    # thresh_uncertainty_values = [150000, 3450000, 6750000, 13500000]
    # max_nr_labels_values = [10]
    # max_nr_iterations_values = [10]
    #
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



    #####
    #
    # folder_path_origin = '/home/niaki/Code/inpynting_images/Tijana'
    # image_filename_versions = ['3', '8', '9', '10']
    #
    # patch_size_values = [16] #[6, 10, 14]
    # thresh_uncertainty_values = [10360] #[70000, 150000, 6750000]
    # max_nr_labels_values = [10]
    # max_nr_iterations_values = [10]
    # use_descriptors_values = [False, True]
    #
    # counter = 0
    # for i, image_filename_version in enumerate(image_filename_versions):
    #     folder_path = folder_path_origin + '/Jian' + image_filename_version
    #     image_filename = 'Jian' + image_filename_version + '_degra.png'
    #     mask_filename = 'Jian' + image_filename_version + 'Mask_inverted.png'
    #     for patch_size in patch_size_values:
    #         stride = patch_size // 2
    #         for thresh_uncertainty in thresh_uncertainty_values:
    #             for max_nr_labels in max_nr_labels_values:
    #                 for max_nr_iterations in max_nr_iterations_values:
    #                     for use_descriptors in use_descriptors_values:
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
    #                                       thresh_uncertainty, max_nr_labels, max_nr_iterations, use_descriptors)
    #                         except Exception as e:
    #                             print("Problem!!", str(e))
    #
    #                         print(counter)
    #                         print()
    #                         counter += 1
    #
    # print(counter)


if __name__ == "__main__":
    main()
