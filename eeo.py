import numpy as np
import sys
import pickle
import math
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio

from data_structures import Image2BInpainted
from data_structures import Node, coordinates_to_position, position_to_coordinates
from data_structures import UP, DOWN, LEFT, RIGHT
from data_structures import get_half_patch_from_patch, opposite_side
# from patch_diff import non_masked_patch_diff, half_patch_diff # TODO should work without these

from patch_diff import max_pool, max_pool_padding, rmse

POOL_SIZE = 8


# -- 1st phase --
# initialization
# (assigning priorities to MRF nodes to be used for determining the visiting order in the 2nd phase)
def initialization_slow(image, thresh_uncertainty):

    global nodes
    global nodes_count

    # for all the patches in an image with stride $stride
    for y in range(0, image.width - image.patch_size + 1, image.stride):
        for x in range(0, image.height - image.patch_size + 1, image.stride):

            patch_mask_overlap = image.mask[x: x + image.patch_size, y: y + image.patch_size]
            patch_mask_overlap_nonzero_elements = np.count_nonzero(patch_mask_overlap)

            # determine with which regions is the patch overlapping
            if patch_mask_overlap_nonzero_elements == 0:
                patch_overlap_source_region = True
                patch_overlap_target_region = False
            elif patch_mask_overlap_nonzero_elements == image.patch_size**2:
                patch_overlap_source_region = False
                patch_overlap_target_region = True
            else:
                patch_overlap_source_region = True
                patch_overlap_target_region = True

            if patch_overlap_target_region:
                patch_position = coordinates_to_position(x, y, image.height, image.patch_size)
                node = Node(patch_position, patch_overlap_source_region, x, y)
                nodes[patch_position] = node
                nodes_count += 1

    for i, node in enumerate(nodes.values()):

        sys.stdout.write("\rInitialising node " + str(i + 1) + "/" + str(nodes_count))

        if node.overlap_source_region:

            # compare the node patch to all patches that are completely in the source region
            for y_compare in range(0, image.width - image.patch_size + 1):
                for x_compare in range(0, image.height - image.patch_size + 1):

                    patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size]
                    patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                    if patch_compare_mask_overlap_nonzero_elements == 0:
                        patch_difference = non_masked_patch_diff(image, node.x_coord, node.y_coord, x_compare, y_compare)
                        
                        patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height, image.patch_size)
                        node.differences[patch_compare_position] = patch_difference
                        node.labels.append(patch_compare_position)

            temp_min_diff = min(list(node.differences.values()))
            temp = [value - temp_min_diff for value in list(node.differences.values())]
            #TODO change thresh_uncertainty such that only patches which are completely in the target region
            #     get assigned the priority value 1.0 (but keep in mind it is used elsewhere)
            node_uncertainty = len([val for (i, val) in enumerate(temp) if val < thresh_uncertainty])

        # if the patch is completely in the target region
        else:

            # make all patches that are completely in the source region be the label of the patch
            for y_compare in range(0, image.width - image.patch_size + 1):
                for x_compare in range(0, image.height - image.patch_size + 1):

                    patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size]
                    patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                    if patch_compare_mask_overlap_nonzero_elements == 0:
                        patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height, image.patch_size)
                        node.differences[patch_compare_position] = 0
                        node.labels.append(patch_compare_position)

            node_uncertainty = len(node.labels)

        # the higher priority the higher priority :D
        node.priority = len(node.labels) / max(node_uncertainty, 1)

        # if nodes_count == 7:
        #     break

    print("\nTotal number of patches: ", len(nodes))
    print("Number of patches to be inpainted: ", nodes_count)


def initialization(image, thresh_uncertainty):
    
    global nodes
    global nodes_count
    global nodes_order

    nodes = {}  # the indices in this list patches match the node_id
    nodes_count = 0
    nodes_order = []

    grey_rgb = [127, 127, 127]
    grey_ir = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
               0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
               0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
               0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,]
    use_inverted_masks = True

    # for all the patches in an image with stride $stride
    for y in range(0, image.width - image.patch_size + 1, image.stride):
        for x in range(0, image.height - image.patch_size + 1, image.stride):

            patch_mask_overlap = image.mask[x: x + image.patch_size, y: y + image.patch_size]
            patch_mask_overlap_nonzero_elements = np.count_nonzero(patch_mask_overlap)

            # determine with which regions is the patch overlapping
            if patch_mask_overlap_nonzero_elements == 0:
                patch_overlap_source_region = True
                patch_overlap_target_region = False
            elif patch_mask_overlap_nonzero_elements == image.patch_size**2:
                patch_overlap_source_region = False
                patch_overlap_target_region = True
            else:
                patch_overlap_source_region = True
                patch_overlap_target_region = True

            if patch_overlap_target_region:
                patch_position = coordinates_to_position(x, y, image.height, image.patch_size)
                node_priority = 1 - (patch_mask_overlap_nonzero_elements / image.patch_size**2)
                node = Node(patch_position, patch_overlap_source_region, x, y, priority=node_priority)
                nodes[patch_position] = node
                nodes_count += 1

    labels_diameter = 100

    # using the rgb values of the patches for comparison, as opposed to their descriptors
    if image.inpainting_approach == Image2BInpainted.USING_RBG_VALUES:

        for i, node in enumerate(nodes.values()):

            sys.stdout.write("\rInitialising node " + str(i + 1) + "/" + str(nodes_count))

            if node.overlap_source_region:

                node_rgb = image.rgb[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                # mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
                # mask_3ch = np.repeat(mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))
                # node_rgb = node_rgb * (1 - mask_3ch)
                if use_inverted_masks:
                    mask = image.inverted_mask_3ch[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                    node_rgb = node_rgb * mask
                else:
                    mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
                    node_rgb[mask.astype(bool), :] = grey_rgb

                # compare the node patch to all patches that are completely in the source region
                for y_compare in range(max(node.y_coord - labels_diameter, 0),
                                       min(node.y_coord + labels_diameter, image.width - image.patch_size + 1)):
                    for x_compare in range(max(node.x_coord - labels_diameter, 0),
                                           min(node.x_coord + labels_diameter, image.height - image.patch_size + 1)):

                        patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size]
                        patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                        if patch_compare_mask_overlap_nonzero_elements == 0:
                            patch_compare_rgb = image.rgb[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size, :]
                            if use_inverted_masks:
                                patch_compare_rgb = patch_compare_rgb * mask
                            else:
                                patch_compare_rgb[mask.astype(bool), :] = grey_rgb

                            patch_difference = rmse(node_rgb, patch_compare_rgb)
                            
                            patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height, image.patch_size)
                            node.differences[patch_compare_position] = patch_difference
                            node.labels.append(patch_compare_position)

                temp_min_diff = min(node.differences.values())
                temp = np.array(list(node.differences.values())) - temp_min_diff
                #TODO change thresh_uncertainty such that only patches which are completely in the target region
                #     get assigned the priority value 1.0 (but keep in mind it is used elsewhere)
                node_uncertainty = len(list(filter(lambda x: x < thresh_uncertainty, temp)))
                node.priority *= len(node.labels) / max(node_uncertainty, 1)
                # node_uncertainty_alternative = np.median(sorted(node.differences.values())[:10]) TODO
                # node.priority *= 1 / node_uncertainty_alternative
        
            # if the patch is completely in the target region
            else:

                # make all patches that are completely in the source region be the label of the patch
                for y_compare in range(max(node.y_coord - labels_diameter, 0),
                                       min(node.y_coord + labels_diameter, image.width - image.patch_size + 1)):
                    for x_compare in range(max(node.x_coord - labels_diameter, 0),
                                           min(node.x_coord + labels_diameter, image.height - image.patch_size + 1)):

                        patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size]
                        patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                        if patch_compare_mask_overlap_nonzero_elements == 0:
                            patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height, image.patch_size)
                            node.differences[patch_compare_position] = 0
                            node.labels.append(patch_compare_position)

                node_uncertainty = len(node.labels)
                node.priority = 0.01

            # the higher priority the higher priority :D
            # node.priority = len(node.labels) / max(node_uncertainty, 1)
            print()
            print(node.priority)
            print(" ", end=" ")

    # using the descriptors from the IR or stored descriptors halves
    elif image.inpainting_approach == Image2BInpainted.USING_IR or image.inpainting_approach == Image2BInpainted.USING_STORED_DESCRIPTORS_HALVES:

        # when patch_size is divisible by pooling size, so no need for padding
        if image.patch_size % POOL_SIZE == 0:

            nr_channels = image.ir.shape[2]

            for i, node in enumerate(nodes.values()):

                sys.stdout.write("\rInitialising node " + str(i + 1) + "/" + str(nodes_count))

                if node.overlap_source_region:

                    if (i == 13) or (i == 28) or (i == 43):
                        print('hey!')

                    node_ir = image.ir[node.x_coord: node.x_coord + image.patch_size,
                              node.y_coord: node.y_coord + image.patch_size, :]
                    # mask = image.mask[node.x_coord: node.x_coord + image.patch_size,
                    #        node.y_coord: node.y_coord + image.patch_size]
                    # mask_more_ch = np.repeat(mask, nr_channels, axis=1).reshape(
                    #     (image.patch_size, image.patch_size, nr_channels))
                    # node_ir = node_ir * (1 - mask_more_ch)

                    if use_inverted_masks:
                        mask = image.inverted_mask_Nch[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                        node_ir = node_ir * mask
                    else:
                        mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
                        node_ir[mask.astype(bool), ...] = grey_ir
                    node_descr = max_pool(node_ir)

                    # compare the node patch to all patches that are completely in the source region
                    for y_compare in range(max(node.y_coord - labels_diameter, 0),
                                           min(node.y_coord + labels_diameter, image.width - image.patch_size + 1)):
                        for x_compare in range(max(node.x_coord - labels_diameter, 0),
                                               min(node.x_coord + labels_diameter, image.height - image.patch_size + 1)):

                            patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size,
                                                         y_compare: y_compare + image.patch_size]
                            patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                            if patch_compare_mask_overlap_nonzero_elements == 0:
                                patch_compare_ir = image.ir[x_compare: x_compare + image.patch_size,
                                                   y_compare: y_compare + image.patch_size, :]
                                if use_inverted_masks:
                                    patch_compare_ir = patch_compare_ir * mask
                                else:
                                    patch_compare_ir[mask.astype(bool), ...] = grey_ir
                                patch_compare_descr = max_pool(patch_compare_ir)

                                patch_difference = rmse(node_descr, patch_compare_descr)

                                patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height,
                                                                                 image.patch_size)
                                node.differences[patch_compare_position] = patch_difference
                                node.labels.append(patch_compare_position)
                    if (i == 13) or (i == 28) or (i == 43):
                        print('hey!')
                    temp_node_diffs_values = np.array(list(node.differences.values()))
                    print()
                    print("    mean " + str(np.mean(temp_node_diffs_values)))
                    print("    std " + str(np.std(temp_node_diffs_values)))
                    print("    median " + str(np.median(temp_node_diffs_values)))
                    print("    min " + str(np.min(temp_node_diffs_values)))
                    print("    max " + str(np.max(temp_node_diffs_values)))


                    temp_min_diff = min(node.differences.values())
                    temp = np.array(list(node.differences.values())) - temp_min_diff
                    #TODO change thresh_uncertainty such that only patches which are completely in the target region
                    #     get assigned the priority value 1.0 (but keep in mind it is used elsewhere)
                    node_uncertainty = len(list(filter(lambda x: x < thresh_uncertainty/250., temp)))
                    node_uncertainty_alternative = np.median(sorted(node.differences.values())[:10])
                    # node.priority *= len(node.labels) / max(node_uncertainty, 1)
                    node_uncertainty_alternative = np.median(sorted(node.differences.values())[:10])
                    node.priority *= 1 / node_uncertainty_alternative

                # if the patch is completely in the target region
                else:

                    # make all patches that are completely in the source region be the label of the patch
                    for y_compare in range(max(node.y_coord - labels_diameter, 0),
                                           min(node.y_coord + labels_diameter, image.width - image.patch_size + 1)):
                        for x_compare in range(max(node.x_coord - labels_diameter, 0),
                                               min(node.x_coord + labels_diameter, image.height - image.patch_size + 1)):

                            patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size,
                                                         y_compare: y_compare + image.patch_size]
                            patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                            if patch_compare_mask_overlap_nonzero_elements == 0:
                                patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height,
                                                                                 image.patch_size)
                                node.differences[patch_compare_position] = 0
                                node.labels.append(patch_compare_position)

                    node_uncertainty = len(node.labels)
                    node.priority = 0.01

                # the higher priority the higher priority :D
                # node.priority = len(node.labels) / max(node_uncertainty, 1)
                print()
                print("node.priority: " + str(node.priority))
                print(" ", end=" ")

        # when patch_size is not divisible by pooling size, so padding is needed
        else:

            nr_channels = image.ir.shape[2]

            # calculating padding parameters for the max pooling
            padding_height_total = POOL_SIZE - (image.patch_size % POOL_SIZE)
            padding_width_total = POOL_SIZE - (image.patch_size % POOL_SIZE)
            padding_height_left = padding_height_total // 2
            padding_height_right = padding_height_total - padding_height_left
            padding_width_left = padding_width_total // 2
            padding_width_right = padding_width_total - padding_width_left

            for i, node in enumerate(nodes.values()):

                sys.stdout.write("\rInitialising node " + str(i + 1) + "/" + str(nodes_count))

                if node.overlap_source_region:

                    node_ir = image.ir[node.x_coord: node.x_coord + image.patch_size,
                              node.y_coord: node.y_coord + image.patch_size, :]
                    # mask = image.mask[node.x_coord: node.x_coord + image.patch_size,
                    #        node.y_coord: node.y_coord + image.patch_size]
                    # mask_more_ch = np.repeat(mask, nr_channels, axis=1).reshape(
                    #     (image.patch_size, image.patch_size, nr_channels))
                    # node_ir = node_ir * (1 - mask_more_ch)

                    if use_inverted_masks:
                        mask = image.inverted_mask_Nch[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                        node_ir = node_ir * mask
                    else:
                        mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
                        node_ir[mask.astype(bool), ...] = grey_ir
                    node_descr = max_pool_padding(node_ir, padding_height_left, padding_height_right,
                                                  padding_width_left, padding_width_right)



                    # compare the node patch to all patches that are completely in the source region
                    for y_compare in range(max(node.y_coord - labels_diameter, 0),
                                           min(node.y_coord + labels_diameter, image.width - image.patch_size + 1)):
                        for x_compare in range(max(node.x_coord - labels_diameter, 0),
                                               min(node.x_coord + labels_diameter, image.height - image.patch_size + 1)):

                            patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size,
                                                         y_compare: y_compare + image.patch_size]
                            patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                            if patch_compare_mask_overlap_nonzero_elements == 0:
                                patch_compare_ir = image.ir[x_compare: x_compare + image.patch_size,
                                                   y_compare: y_compare + image.patch_size, :]
                                if use_inverted_masks:
                                    patch_compare_ir = patch_compare_ir * mask
                                else:
                                    patch_compare_ir[mask.astype(bool), ...] = grey_ir
                                patch_compare_descr = max_pool_padding(patch_compare_ir, padding_height_left,
                                                                       padding_height_right, padding_width_left,
                                                                       padding_width_right)
                                
                                patch_difference = rmse(node_descr, patch_compare_descr)

                                patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height,
                                                                                 image.patch_size)
                                node.differences[patch_compare_position] = patch_difference
                                node.labels.append(patch_compare_position)

                    temp_min_diff = min(node.differences.values())
                    temp = np.array(list(node.differences.values())) - temp_min_diff
                    # TODO change thresh_uncertainty such that only patches which are completely in the target region
                    #     get assigned the priority value 1.0 (but keep in mind it is used elsewhere)
                    node_uncertainty = len(list(filter(lambda x: x < thresh_uncertainty, temp)))

                # if the patch is completely in the target region
                else:

                    # make all patches that are completely in the source region be the label of the patch
                    for y_compare in range(max(node.y_coord - labels_diameter, 0),
                                           min(node.y_coord + labels_diameter, image.width - image.patch_size + 1)):
                        for x_compare in range(max(node.x_coord - labels_diameter, 0),
                                               min(node.x_coord + labels_diameter, image.height - image.patch_size + 1)):

                            patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size,
                                                         y_compare: y_compare + image.patch_size]
                            patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                            if patch_compare_mask_overlap_nonzero_elements == 0:
                                patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height,
                                                                                 image.patch_size)
                                node.differences[patch_compare_position] = 0
                                node.labels.append(patch_compare_position)

                    node_uncertainty = len(node.labels)

                # the higher priority the higher priority :D
                node.priority = len(node.labels) / max(node_uncertainty, 1)

    elif image.inpainting_approach == Image2BInpainted.USING_STORED_DESCRIPTORS_CUBE:

        raise NotImplementedError(
            "The other parts of the inpainting are not (yet) implemented to work with descriptors cube.")

        # for i, node in enumerate(nodes.values()):
        #
        #     sys.stdout.write("\rInitialising node " + str(i + 1) + "/" + str(nodes_count))
        #
        #     if node.overlap_source_region:
        #
        #         node_descr = image.descriptor_cube[node.x_coord, node.y_coord, :]
        #
        #         # compare the node patch to all patches that are completely in the source region
        #         for y_compare in range(max(node.y_coord - labels_diameter, 0),
        #                                min(node.y_coord + labels_diameter, image.width - image.patch_size + 1)):
        #             for x_compare in range(max(node.x_coord - labels_diameter, 0),
        #                                    min(node.x_coord + labels_diameter, image.height - image.patch_size + 1)):
        #
        #                 patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size,
        #                                              y_compare: y_compare + image.patch_size]
        #                 patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)
        #
        #                 if patch_compare_mask_overlap_nonzero_elements == 0:
        #                     patch_compare_descr = image.descriptor_cube[x_compare, y_compare, :]
        #
        #                     patch_difference = rmse(node_descr, patch_compare_descr)
        #
        #                     patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height,
        #                                                                      image.patch_size)
        #                     node.differences[patch_compare_position] = patch_difference
        #                     node.labels.append(patch_compare_position)
        #
        #         temp_min_diff = min(node.differences.values())
        #         temp = np.array(list(node.differences.values())) - temp_min_diff
        #         # TODO change thresh_uncertainty such that only patches which are completely in the target region
        #         #     get assigned the priority value 1.0 (but keep in mind it is used elsewhere)
        #         node_uncertainty = len(list(filter(lambda x: x < thresh_uncertainty, temp)))
        #
        #     # if the patch is completely in the target region
        #     else:
        #
        #         # make all patches that are completely in the source region be the label of the patch
        #         for y_compare in range(max(node.y_coord - labels_diameter, 0),
        #                                min(node.y_coord + labels_diameter, image.width - image.patch_size + 1)):
        #             for x_compare in range(max(node.x_coord - labels_diameter, 0),
        #                                    min(node.x_coord + labels_diameter, image.height - image.patch_size + 1)):
        #
        #                 patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size,
        #                                              y_compare: y_compare + image.patch_size]
        #                 patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)
        #
        #                 if patch_compare_mask_overlap_nonzero_elements == 0:
        #                     patch_compare_position = coordinates_to_position(x_compare, y_compare, image.height,
        #                                                                      image.patch_size)
        #                     node.differences[patch_compare_position] = 0
        #                     node.labels.append(patch_compare_position)
        #
        #         node_uncertainty = len(node.labels)
        #
        #     # the higher priority the higher priority :D
        #     node.priority = len(node.labels) / max(node_uncertainty, 1)


    else:
        raise AssertionError("Inpainting approach has not been properly set.")

    print()
    for i, node in enumerate(nodes.values()):
        print(node.priority, end=' ')

    try:
        pickle.dump(nodes, open("/home/niaki/Downloads/nodes.pickle", "wb"))
    except Exception as e:
        print("Problem while trying to pickle: ", str(e))


    print("\nTotal number of patches: ", len(nodes))
    print("Number of patches to be inpainted: ", nodes_count)


# -- 2nd phase --
# label pruning
# (reducing the number of labels at each node to a relatively small number)
def label_pruning(image, thresh_uncertainty, max_nr_labels):
    global nodes
    global nodes_count
    global nodes_order

    # make a copy of the differences which we can edit and use in this method, and afterwards discard
    for node in nodes.values():
        node.additional_differences = node.differences.copy()

    # visualisation of nodes only
    # fig, ax = plt.subplots(1)
    # ax.imshow(image.rgb)

    # for all the patches that have an overlap with the target region (aka nodes)
    for i in range(nodes_count):

        # find the node with the highest priority that hasn't yet been visited
        node_highest_priority = max(filter(lambda node:not node.committed, nodes.values()),
                                    key=lambda node: node.priority,
                                    default=-1)
        if node_highest_priority == -1:
            err_msg = f'Nodes has no non-committed entries. Make sure the global values are Reset when inpainting new image! {nodes.values}'
            raise AssertionError(err_msg)
            
        highest_priority = node_highest_priority.priority
        node_highest_priority_id = node_highest_priority.node_id
        
        node = nodes[node_highest_priority_id]
        node.committed = True

        node.prune_labels(max_nr_labels)

        # visualisation of nodes only
        # rect = patches.Rectangle((node.y_coord, node.x_coord),
        #                          image.patch_size, image.patch_size, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)

        # visualise_nodes_pruned_labels(node, image)

        print('Highest priority node {0:3d}/{1:3d}: ID {2:d}, priority {3:.2f}'.format(i + 1, nodes_count, node_highest_priority_id, node_highest_priority.priority))
        nodes_order.append(node_highest_priority_id)

        # get the neighbors if they exist and have overlap with the target region
        node_neighbor_up, node_neighbor_down, node_neighbor_left, node_neighbor_right = get_neighbor_nodes(
            node, image)

        if image.inpainting_approach == Image2BInpainted.USING_RBG_VALUES:
            update_neighbors_priority_rgb(node, node_neighbor_up, UP, image, thresh_uncertainty)
            update_neighbors_priority_rgb(node, node_neighbor_down, DOWN, image, thresh_uncertainty)
            update_neighbors_priority_rgb(node, node_neighbor_left, LEFT, image, thresh_uncertainty)
            update_neighbors_priority_rgb(node, node_neighbor_right, RIGHT, image, thresh_uncertainty)
        elif image.inpainting_approach == Image2BInpainted.USING_IR:
            if (image.patch_size // 2) % POOL_SIZE == 0:  # div by 2 is because we will be comparing half patches
                update_neighbors_priority_ir(node, node_neighbor_up, UP, image, thresh_uncertainty)
                update_neighbors_priority_ir(node, node_neighbor_down, DOWN, image, thresh_uncertainty)
                update_neighbors_priority_ir(node, node_neighbor_left, LEFT, image, thresh_uncertainty)
                update_neighbors_priority_ir(node, node_neighbor_right, RIGHT, image, thresh_uncertainty)
            else:
                update_neighbors_priority_ir_padded_mp(node, node_neighbor_up, UP, image, thresh_uncertainty)
                update_neighbors_priority_ir_padded_mp(node, node_neighbor_down, DOWN, image, thresh_uncertainty)
                update_neighbors_priority_ir_padded_mp(node, node_neighbor_left, LEFT, image, thresh_uncertainty)
                update_neighbors_priority_ir_padded_mp(node, node_neighbor_right, RIGHT, image, thresh_uncertainty)
        elif image.inpainting_approach == Image2BInpainted.USING_STORED_DESCRIPTORS_HALVES:
            update_neighbors_priority_stored_descrs_halves(node, node_neighbor_up, UP, image, thresh_uncertainty)
            update_neighbors_priority_stored_descrs_halves(node, node_neighbor_down, DOWN, image, thresh_uncertainty)
            update_neighbors_priority_stored_descrs_halves(node, node_neighbor_left, LEFT, image, thresh_uncertainty)
            update_neighbors_priority_stored_descrs_halves(node, node_neighbor_right, RIGHT, image, thresh_uncertainty)
        elif image.inpainting_approach == Image2BInpainted.USING_STORED_DESCRIPTORS_CUBE:
            raise NotImplementedError(
                "The other parts of the inpainting are not (yet) implemented to work with descriptors cube.")
            # update_neighbors_priority_stored_descrs_cube(node, node_neighbor_up, UP, image, thresh_uncertainty)
        else:
            raise AssertionError("Inpainting approach has not been properly set.")

    # visualisation of nodes only
    # plt.show()


def visualise_nodes_pruned_labels(node, image):
    fig, ax = plt.subplots(1)
    ax.imshow(image.rgb)
    rect = patches.Rectangle((node.y_coord, node.x_coord),
                             image.patch_size, image.patch_size, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    colour_string_original = "#00"
    for node_label_count in range(len(node.pruned_labels) - 1, -1, -1):
        node_label_id = node.pruned_labels[node_label_count]
        node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)
        if node_label_count == 0:
            colour_string = "#FFFF00"
        else:
            # going from green (the most similar) to blue ()
            colour_string = colour_string_original + "%0.2X" % ((9 - (node_label_count - 1)) * 28) + "%0.2X" % (
                        (node_label_count - 1) * 28)
        rect = patches.Rectangle((node_label_y_coord, node_label_x_coord),
                                 image.patch_size, image.patch_size, linewidth=1,
                                 edgecolor=colour_string, facecolor='none')
        ax.add_patch(rect)
    plt.show()


def update_neighbors_priority_slow(node, neighbor, side, image, thresh_uncertainty):

    # if neighbor is a node that hasn't been committed yet
    if neighbor is not None and not neighbor.committed:

        min_additional_difference = sys.maxsize
        additional_differences = {}

        for neighbors_label_id in neighbor.labels:

            neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

            # patch_neighbors_label_rgb = image.rgb[
            #                             neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
            #                             neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]
            # patch_neighbors_label_rgb_half = get_half_patch_from_patch(patch_neighbors_label_rgb, image.stride, opposite_side(side))

            for node_label_id in node.pruned_labels:

                node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                # patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size,
                #                    node_label_y_coord: node_label_y_coord + image.patch_size, :]
                # patchs_label_rgb_half = get_half_patch_from_patch(patchs_label_rgb, image.stride, side)
                # difference = patch_diff(patch_neighbors_label_rgb_half, patchs_label_rgb_half)

                difference = half_patch_diff(image, node_label_x_coord, node_label_y_coord, neighbors_label_x_coord, neighbors_label_y_coord, side)

                if difference < min_additional_difference:
                    min_additional_difference = difference

            additional_differences[neighbors_label_id] = min_additional_difference
            min_additional_difference = sys.maxsize

        for key in additional_differences.keys():
            if key in neighbor.additional_differences:
                neighbor.additional_differences[key] += additional_differences[key]
            else:
                neighbor.additional_differences[key] = additional_differences[key]
                print("Will it ever come to this? (2) (TODO delete if unnecessary)")

        temp_min_diff = min(neighbor.additional_differences.values())
        temp = [value - temp_min_diff for value in
                list(neighbor.additional_differences.values())]
        neighbor_uncertainty = [value < thresh_uncertainty for (i, value) in enumerate(temp)].count(True)

        neighbor.priority = len(neighbor.additional_differences) / neighbor_uncertainty #len(patch_neighbor.differences)?


# using the rgb values of the patches for comparison, as opposed to their descriptors
def update_neighbors_priority_rgb(node, neighbor, side, image, thresh_uncertainty):

    # if neighbor is a node that hasn't been committed yet
    if neighbor is not None and not neighbor.committed:

        min_additional_difference = sys.maxsize
        additional_differences = {}

        for neighbors_label_id in neighbor.labels:

            neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

            patch_neighbors_label_rgb = image.rgb[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                        neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]
            patch_neighbors_label_rgb_half = get_half_patch_from_patch(patch_neighbors_label_rgb, image.stride, opposite_side(side))

            for node_label_id in node.pruned_labels:

                node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size,
                                   node_label_y_coord: node_label_y_coord + image.patch_size, :]
                patchs_label_rgb_half = get_half_patch_from_patch(patchs_label_rgb, image.stride, side)

                # Normalised
                difference = rmse(patch_neighbors_label_rgb_half, patchs_label_rgb_half)

                if difference < (min_additional_difference):    # When changing to mse? /patch_neighbors_label_rgb_half.size
                    min_additional_difference = difference

            additional_differences[neighbors_label_id] = min_additional_difference
            min_additional_difference = sys.maxsize

        for key in additional_differences.keys():
            if key in neighbor.additional_differences:
                neighbor.additional_differences[key] += additional_differences[key]
            else:
                neighbor.additional_differences[key] = additional_differences[key]
                print("Will it ever come to this? (2) (TODO delete if unnecessary)")

        temp_min_diff = min(neighbor.additional_differences.values())
        temp = [value - temp_min_diff for value in
                list(neighbor.additional_differences.values())]
        neighbor_uncertainty = [value < (thresh_uncertainty) for (i, value) in enumerate(temp)].count(True)

        neighbor.priority = len(neighbor.additional_differences) / neighbor_uncertainty #len(patch_neighbor.differences)?


# using the descriptors from the IR, when patch_size is divisible by pooling size, so no need for padding
def update_neighbors_priority_ir(node, neighbor, side, image, thresh_uncertainty):

    # if neighbor is a node that hasn't been committed yet
    if neighbor is not None and not neighbor.committed:

        min_additional_difference = sys.maxsize
        additional_differences = {}

        for neighbors_label_id in neighbor.labels:

            neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

            patch_neighbors_label_ir = image.ir[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                        neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]
            patch_neighbors_label_half_ir = get_half_patch_from_patch(patch_neighbors_label_ir, image.stride, opposite_side(side))
            patch_neighbors_label_half_descr = max_pool(patch_neighbors_label_half_ir)

            for node_label_id in node.pruned_labels:

                node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                patchs_label_ir = image.ir[node_label_x_coord: node_label_x_coord + image.patch_size,
                                   node_label_y_coord: node_label_y_coord + image.patch_size, :]
                patchs_label_half_ir = get_half_patch_from_patch(patchs_label_ir, image.stride, side)
                patchs_label_half_descr = max_pool(patchs_label_half_ir)

                # normalised
                difference = rmse(patch_neighbors_label_half_descr, patchs_label_half_descr)

                if difference < min_additional_difference:
                    min_additional_difference = difference

            additional_differences[neighbors_label_id] = min_additional_difference
            min_additional_difference = sys.maxsize

        for key in additional_differences.keys():
            if key in neighbor.additional_differences:
                neighbor.additional_differences[key] += additional_differences[key]
            else:
                neighbor.additional_differences[key] = additional_differences[key]
                print("Will it ever come to this? (2) (TODO delete if unnecessary)")

        temp_min_diff = min(neighbor.additional_differences.values())
        temp = [value - temp_min_diff for value in
                list(neighbor.additional_differences.values())]
        neighbor_uncertainty = [value < thresh_uncertainty for (i, value) in enumerate(temp)].count(True)

        neighbor.priority = len(neighbor.additional_differences) / neighbor_uncertainty #len(patch_neighbor.differences)?


# using the descriptors from the IR, when patch_size is not divisible by pooling size, so padding is needed
def update_neighbors_priority_ir_padded_mp(node, neighbor, side, image, thresh_uncertainty):

    # calculating padding parameters for the max pooling, for the case: left/right
    padding_height_total_lr = POOL_SIZE - (image.patch_size % POOL_SIZE)
    padding_width_total_lr = POOL_SIZE - ((image.patch_size // 2) % POOL_SIZE)
    padding_height_left_lr = padding_height_total_lr // 2
    padding_height_right_lr = padding_height_total_lr - padding_height_left_lr
    padding_width_left_lr = padding_width_total_lr // 2
    padding_width_right_lr = padding_width_total_lr - padding_width_left_lr
    # calculating padding parameters for the max pooling, for the case: up/down
    padding_height_total_ud = POOL_SIZE - ((image.patch_size // 2) % POOL_SIZE)
    padding_width_total_ud = POOL_SIZE - (image.patch_size % POOL_SIZE)
    padding_height_left_ud = padding_height_total_ud // 2
    padding_height_right_ud = padding_height_total_ud - padding_height_left_ud
    padding_width_left_ud = padding_width_total_ud // 2
    padding_width_right_ud = padding_width_total_ud - padding_width_left_ud

    # if neighbor is a node that hasn't been committed yet
    if neighbor is not None and not neighbor.committed:

        min_additional_difference = sys.maxsize
        additional_differences = {}

        for neighbors_label_id in neighbor.labels:

            neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

            patch_neighbors_label_ir = image.ir[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                        neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]
            patch_neighbors_label_half_ir = get_half_patch_from_patch(patch_neighbors_label_ir, image.stride, opposite_side(side))
            if side == LEFT or side == RIGHT:
                patch_neighbors_label_half_descr = max_pool_padding(patch_neighbors_label_half_ir, padding_height_left_lr, padding_height_right_lr, padding_width_left_lr, padding_width_right_lr)
            else:  # up or down
                patch_neighbors_label_half_descr = max_pool_padding(patch_neighbors_label_half_ir, padding_height_left_ud, padding_height_right_ud, padding_width_left_ud, padding_width_right_ud)

            for node_label_id in node.pruned_labels:

                node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                patchs_label_ir = image.ir[node_label_x_coord: node_label_x_coord + image.patch_size,
                                   node_label_y_coord: node_label_y_coord + image.patch_size, :]
                patchs_label_half_ir = get_half_patch_from_patch(patchs_label_ir, image.stride, side)
                if side == LEFT or side == RIGHT:
                    patchs_label_half_descr = max_pool_padding(patchs_label_half_ir, padding_height_left_lr, padding_height_right_lr, padding_width_left_lr, padding_width_right_lr)
                else:  # up or down
                    patchs_label_half_descr = max_pool_padding(patchs_label_half_ir, padding_height_left_ud, padding_height_right_ud, padding_width_left_ud, padding_width_right_ud)

                difference = rmse(patch_neighbors_label_half_descr, patchs_label_half_descr)

                if difference < min_additional_difference:
                    min_additional_difference = difference

            additional_differences[neighbors_label_id] = min_additional_difference
            min_additional_difference = sys.maxsize

        for key in additional_differences.keys():
            if key in neighbor.additional_differences:
                neighbor.additional_differences[key] += additional_differences[key]
            else:
                neighbor.additional_differences[key] = additional_differences[key]
                print("Will it ever come to this? (2) (TODO delete if unnecessary)")

        temp_min_diff = min(neighbor.additional_differences.values())
        temp = [value - temp_min_diff for value in
                list(neighbor.additional_differences.values())]
        neighbor_uncertainty = [value < thresh_uncertainty for (i, value) in enumerate(temp)].count(True)

        neighbor.priority = len(neighbor.additional_differences) / neighbor_uncertainty #len(patch_neighbor.differences)?


# using the stored descriptors halves
def update_neighbors_priority_stored_descrs_halves(node, neighbor, side, image, thresh_uncertainty):

    # if neighbor is a node that hasn't been committed yet
    if neighbor is not None and not neighbor.committed:

        min_additional_difference = sys.maxsize
        additional_differences = {}

        position_shift_down = image.stride
        position_shift_right = image.stride * (image.height - image.patch_size + 1)

        for neighbors_label_id in neighbor.labels:

            if opposite_side(side) == UP:
                neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)
                neighbor_position = coordinates_to_position(neighbors_label_x_coord, neighbors_label_y_coord, image.height, image.stride)
                patch_neighbors_label_half_descr = image.half_patch_landscape_descriptors[neighbor_position]
            elif opposite_side(side) == DOWN:
                neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)
                neighbor_position = coordinates_to_position(neighbors_label_x_coord, neighbors_label_y_coord, image.height, image.stride)
                patch_neighbors_label_half_descr = image.half_patch_landscape_descriptors[neighbor_position + position_shift_down]
            elif opposite_side(side) == LEFT:
                patch_neighbors_label_half_descr = image.half_patch_portrait_descriptors[neighbors_label_id]
            else:
                patch_neighbors_label_half_descr = image.half_patch_portrait_descriptors[neighbors_label_id + position_shift_right]

            for node_label_id in node.pruned_labels:

                if side == UP:
                    node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)
                    node_position = coordinates_to_position(node_label_x_coord, node_label_y_coord, image.height, image.stride)
                    patchs_label_half_descr = image.half_patch_landscape_descriptors[node_position]
                elif side == DOWN:
                    node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)
                    node_position = coordinates_to_position(node_label_x_coord, node_label_y_coord, image.height, image.stride)
                    patchs_label_half_descr = image.half_patch_landscape_descriptors[node_position + position_shift_down]
                elif side == LEFT:
                    patchs_label_half_descr = image.half_patch_portrait_descriptors[node_label_id]
                else:
                    patchs_label_half_descr = image.half_patch_portrait_descriptors[node_label_id + position_shift_right]

                # difference = np.sum(np.subtract(patch_neighbors_label_half_descr, patchs_label_half_descr, dtype=np.float32) ** 2)
                difference = rmse(patch_neighbors_label_half_descr, patchs_label_half_descr)

                if difference < (min_additional_difference):
                    min_additional_difference = difference

            additional_differences[neighbors_label_id] = min_additional_difference
            min_additional_difference = sys.maxsize

        for key in additional_differences.keys():
            if key in neighbor.additional_differences:
                neighbor.additional_differences[key] += additional_differences[key]
            else:
                neighbor.additional_differences[key] = additional_differences[key]
                print("Will it ever come to this? (2) (TODO delete if unnecessary)")

        temp_min_diff = min(neighbor.additional_differences.values())
        temp = [value - temp_min_diff for value in
                list(neighbor.additional_differences.values())]
        neighbor_uncertainty = [value < (thresh_uncertainty/250.) for (i, value) in enumerate(temp)].count(True)

        neighbor.priority = len(neighbor.additional_differences) / neighbor_uncertainty #len(patch_neighbor.differences)?

        # if np.median(sorted(neighbor.additional_differences.values())[:10]) == 0:
        #     print("sta se desava")
        # # neighbor.priority += 1 / np.median(sorted(neighbor.additional_differences.values())[:10])
        #
        # temp_first_non_zero_index = next((i for i, x in enumerate(sorted(neighbor.additional_differences.values())) if x), None)
        # neighbor.priority += 1 / np.median(sorted(neighbor.additional_differences.values())[temp_first_non_zero_index: temp_first_non_zero_index + 10])
        print(neighbor.priority)




# # using the stored descriptors cube
# def update_neighbors_priority_stored_descrs_cube(node, neighbor, side, image, thresh_uncertainty):
#
#     # if neighbor is a node that hasn't been committed yet
#     if neighbor is not None and not neighbor.committed:
#
#         min_additional_difference = sys.maxsize
#         additional_differences = {}
#
#         position_shift_down = image.stride
#         position_shift_right = image.stride * (image.height - image.patch_size + 1)
#
#         for neighbors_label_id in neighbor.labels:
#
#             if opposite_side(side) == UP:
#                 neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)
#                 neighbor_position = coordinates_to_position(neighbors_label_x_coord, neighbors_label_y_coord, image.height, image.stride)
#                 patch_neighbors_label_half_descr = image.half_patch_landscape_descriptors[neighbor_position]
#             elif opposite_side(side) == DOWN:
#                 neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)
#                 neighbor_position = coordinates_to_position(neighbors_label_x_coord, neighbors_label_y_coord, image.height, image.stride)
#                 patch_neighbors_label_half_descr = image.half_patch_landscape_descriptors[neighbor_position + position_shift_down]
#             elif opposite_side(side) == LEFT:
#                 patch_neighbors_label_half_descr = image.half_patch_portrait_descriptors[neighbors_label_id]
#             else:
#                 patch_neighbors_label_half_descr = image.half_patch_portrait_descriptors[neighbors_label_id + position_shift_right]
#
#             for node_label_id in node.pruned_labels:
#
#                 if side == UP:
#                     node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)
#                     node_position = coordinates_to_position(node_label_x_coord, node_label_y_coord, image.height, image.stride)
#                     patchs_label_half_descr = image.half_patch_landscape_descriptors[node_position]
#                 elif side == DOWN:
#                     node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)
#                     node_position = coordinates_to_position(node_label_x_coord, node_label_y_coord, image.height, image.stride)
#                     patchs_label_half_descr = image.half_patch_landscape_descriptors[node_position + position_shift_down]
#                 elif side == LEFT:
#                     patchs_label_half_descr = image.half_patch_portrait_descriptors[node_label_id]
#                 else:
#                     patchs_label_half_descr = image.half_patch_portrait_descriptors[node_label_id + position_shift_right]
#
#                 difference = np.sum(np.subtract(patch_neighbors_label_half_descr, patchs_label_half_descr, dtype=np.float32) ** 2)
#
#                 if difference < (min_additional_difference):
#                     min_additional_difference = difference
#
#             additional_differences[neighbors_label_id] = min_additional_difference
#             min_additional_difference = sys.maxsize
#
#         for key in additional_differences.keys():
#             if key in neighbor.additional_differences:
#                 neighbor.additional_differences[key] += additional_differences[key]
#             else:
#                 neighbor.additional_differences[key] = additional_differences[key]
#                 print("Will it ever come to this? (2) (TODO delete if unnecessary)")
#
#         temp_min_diff = min(neighbor.additional_differences.values())
#         temp = [value - temp_min_diff for value in
#                 list(neighbor.additional_differences.values())]
#         neighbor_uncertainty = [value < thresh_uncertainty for (i, value) in enumerate(temp)].count(True)
#
#         neighbor.priority = len(neighbor.additional_differences) / neighbor_uncertainty #len(patch_neighbor.differences)?
#
#
#



def get_neighbor_nodes(node, image):

    neighbor_up_id = node.get_up_neighbor_position(image)
    if neighbor_up_id is None:
        neighbor_up = None
    else:
        neighbor_up = nodes.get(neighbor_up_id) # this will either get the node or return None

    neighbor_down_id = node.get_down_neighbor_position(image)
    if neighbor_down_id is None:
        neighbor_down = None
    else:
        neighbor_down = nodes.get(neighbor_down_id)

    neighbor_left_id = node.get_left_neighbor_position(image)
    if neighbor_left_id is None:
        neighbor_left = None
    else:
        neighbor_left = nodes.get(neighbor_left_id)

    neighbor_right_id = node.get_right_neighbor_position(image)
    if neighbor_right_id is None:
        neighbor_right = None
    else:
        neighbor_right = nodes.get(neighbor_right_id)

    return neighbor_up, neighbor_down, neighbor_left, neighbor_right


def compute_pairwise_potential_matrix(image, max_nr_labels):

    global nodes
    global nodes_count

    if image.inpainting_approach == Image2BInpainted.USING_RBG_VALUES:

        # calculate pairwise potential matrix for all pairs of nodes (pathces that have overlap with the target region)
        for node in nodes.values():

            # get the neighbors if they exist and have overlap with the target region (i.e. if they're nodes)
            neighbor_up, _, neighbor_left, _ = get_neighbor_nodes(node, image)

            if neighbor_up is not None:

                potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

                for i, node_label_id in enumerate(node.pruned_labels):

                    node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                    patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size,
                                       node_label_y_coord: node_label_y_coord + image.patch_size, :]

                    patchs_label_rgb_up = get_half_patch_from_patch(patchs_label_rgb, image.stride, UP)

                    for j, neighbors_label_id in enumerate(neighbor_up.pruned_labels):

                        neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

                        patchs_neighbors_label_rgb = image.rgb[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                                     neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]

                        patchs_neighbors_label_rgb_down = get_half_patch_from_patch(patchs_neighbors_label_rgb, image.stride, DOWN)

                        potential_matrix[i, j] = rmse(patchs_label_rgb_up, patchs_neighbors_label_rgb_down)


                node.potential_matrix_up = potential_matrix
                neighbor_up.potential_matrix_down = potential_matrix

            if neighbor_left is not None:

                potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

                for i, node_label_id in enumerate(node.pruned_labels):

                    node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                    patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size,
                                       node_label_y_coord: node_label_y_coord + image.patch_size, :]

                    patchs_label_rgb_left = get_half_patch_from_patch(patchs_label_rgb, image.stride, LEFT)

                    for j, neighbors_label_id in enumerate(neighbor_left.pruned_labels):

                        neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

                        patchs_neighbors_label_rgb = image.rgb[
                                                     neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                                     neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size,
                                                     :]

                        patchs_neighbors_label_rgb_right = get_half_patch_from_patch(patchs_neighbors_label_rgb, image.stride,
                                                                                    RIGHT)

                        potential_matrix[i, j] = rmse(patchs_label_rgb_left, patchs_neighbors_label_rgb_right)


                node.potential_matrix_left = potential_matrix
                neighbor_left.potential_matrix_right = potential_matrix

    elif image.inpainting_approach == Image2BInpainted.USING_IR or image.inpainting_approach == Image2BInpainted.USING_STORED_DESCRIPTORS_HALVES:

        if (image.patch_size // 2) % POOL_SIZE == 0:  # div by 2 is because we will be comparing half patches

            # calculate pairwise potential matrix for all pairs of nodes (pathces that have overlap with the target region)
            for node in nodes.values():

                # get the neighbors if they exist and have overlap with the target region (i.e. if they're nodes)
                neighbor_up, _, neighbor_left, _ = get_neighbor_nodes(node, image)

                if neighbor_up is not None:

                    potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

                    for i, node_label_id in enumerate(node.pruned_labels):

                        node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                        patchs_label_ir = image.ir[node_label_x_coord: node_label_x_coord + image.patch_size,
                                           node_label_y_coord: node_label_y_coord + image.patch_size, :]
                        patchs_label_up_ir = get_half_patch_from_patch(patchs_label_ir, image.stride, UP)
                        patchs_label_up_descr = max_pool(patchs_label_up_ir)

                        for j, neighbors_label_id in enumerate(neighbor_up.pruned_labels):
                            neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)
        
                            patchs_neighbors_label_ir = image.ir[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                                         neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]
                            patchs_neighbors_label_down_ir = get_half_patch_from_patch(patchs_neighbors_label_ir, image.stride, DOWN)
                            patchs_neighbors_label_down_descr = max_pool(patchs_neighbors_label_down_ir)

                            potential_matrix[i, j] = rmse(patchs_label_up_descr, patchs_neighbors_label_down_descr)

                    node.potential_matrix_up = potential_matrix
                    neighbor_up.potential_matrix_down = potential_matrix

                if neighbor_left is not None:

                    potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

                    for i, node_label_id in enumerate(node.pruned_labels):

                        node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                        patchs_label_ir = image.ir[node_label_x_coord: node_label_x_coord + image.patch_size,
                                           node_label_y_coord: node_label_y_coord + image.patch_size, :]
                        patchs_label_left_ir = get_half_patch_from_patch(patchs_label_ir, image.stride, LEFT)
                        patchs_label_left_descr = max_pool(patchs_label_left_ir)

                        for j, neighbors_label_id in enumerate(neighbor_left.pruned_labels):
                            neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

                            patchs_neighbors_label_ir = image.ir[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                                         neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]
                            patchs_neighbors_label_right_ir = get_half_patch_from_patch(patchs_neighbors_label_ir, image.stride, RIGHT)
                            patchs_neighbors_label_right_descr = max_pool(patchs_neighbors_label_right_ir)

                            potential_matrix[i, j] = rmse(patchs_label_left_descr, patchs_neighbors_label_right_descr)

                    node.potential_matrix_left = potential_matrix
                    neighbor_left.potential_matrix_right = potential_matrix

        else:  # need to pad before max pooling

            # calculating padding parameters for the max pooling, for the case: left/right
            padding_height_total_lr = POOL_SIZE - (image.patch_size % POOL_SIZE)
            padding_width_total_lr = POOL_SIZE - ((image.patch_size // 2) % POOL_SIZE)
            padding_height_left_lr = padding_height_total_lr // 2
            padding_height_right_lr = padding_height_total_lr - padding_height_left_lr
            padding_width_left_lr = padding_width_total_lr // 2
            padding_width_right_lr = padding_width_total_lr - padding_width_left_lr
            # calculating padding parameters for the max pooling, for the case: up/down
            padding_height_total_ud = POOL_SIZE - ((image.patch_size // 2) % POOL_SIZE)
            padding_width_total_ud = POOL_SIZE - (image.patch_size % POOL_SIZE)
            padding_height_left_ud = padding_height_total_ud // 2
            padding_height_right_ud = padding_height_total_ud - padding_height_left_ud
            padding_width_left_ud = padding_width_total_ud // 2
            padding_width_right_ud = padding_width_total_ud - padding_width_left_ud

            # calculate pairwise potential matrix for all pairs of nodes (pathces that have overlap with the target region)
            for node in nodes.values():

                # get the neighbors if they exist and have overlap with the target region (i.e. if they're nodes)
                neighbor_up, _, neighbor_left, _ = get_neighbor_nodes(node, image)

                if neighbor_up is not None:

                    potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

                    for i, node_label_id in enumerate(node.pruned_labels):

                        node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                        patchs_label_ir = image.ir[node_label_x_coord: node_label_x_coord + image.patch_size,
                                           node_label_y_coord: node_label_y_coord + image.patch_size, :]
                        patchs_label_up_ir = get_half_patch_from_patch(patchs_label_ir, image.stride, UP)
                        patchs_label_up_descr = max_pool_padding(patchs_label_up_ir, padding_height_left_ud, padding_height_right_ud, padding_width_left_ud, padding_width_right_ud)

                        for j, neighbors_label_id in enumerate(neighbor_up.pruned_labels):
                            neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

                            patchs_neighbors_label_ir = image.ir[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                                         neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]
                            patchs_neighbors_label_down_ir = get_half_patch_from_patch(patchs_neighbors_label_ir, image.stride, DOWN)
                            patchs_neighbors_label_down_descr = max_pool_padding(patchs_neighbors_label_down_ir, padding_height_left_ud, padding_height_right_ud, padding_width_left_ud, padding_width_right_ud)

                            potential_matrix[i, j] = rmse(patchs_label_up_descr, patchs_neighbors_label_down_descr)

                    node.potential_matrix_up = potential_matrix
                    neighbor_up.potential_matrix_down = potential_matrix

                if neighbor_left is not None:

                    potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

                    for i, node_label_id in enumerate(node.pruned_labels):

                        node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                        patchs_label_ir = image.ir[node_label_x_coord: node_label_x_coord + image.patch_size,
                                           node_label_y_coord: node_label_y_coord + image.patch_size, :]
                        patchs_label_left_ir = get_half_patch_from_patch(patchs_label_ir, image.stride, LEFT)
                        patchs_label_left_descr = max_pool_padding(patchs_label_left_ir, padding_height_left_lr, padding_height_right_lr, padding_width_left_lr, padding_width_right_lr)

                        for j, neighbors_label_id in enumerate(neighbor_left.pruned_labels):
                            neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

                            patchs_neighbors_label_ir = image.ir[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                                         neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]
                            patchs_neighbors_label_right_ir = get_half_patch_from_patch(patchs_neighbors_label_ir, image.stride, RIGHT)
                            patchs_neighbors_label_right_descr = max_pool_padding(patchs_neighbors_label_right_ir, padding_height_left_lr, padding_height_right_lr, padding_width_left_lr, padding_width_right_lr)

                            potential_matrix[i, j] = rmse(patchs_label_left_descr, patchs_neighbors_label_right_descr)


                    node.potential_matrix_left = potential_matrix
                    neighbor_left.potential_matrix_right = potential_matrix

    else:
        raise AssertionError("Inpainting approach has not been properly set.")


def compute_label_cost(image, max_nr_labels):

    global nodes
    global nodes_count

    if image.inpainting_approach == Image2BInpainted.USING_RBG_VALUES:

        for node in nodes.values():

            node.label_cost = [0 for _ in range(max_nr_labels)]

            if node.overlap_source_region:

                patch_rgb = image.rgb[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                # mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
                # mask_3ch = np.repeat(mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))
                # patch_rgb = patch_rgb * (1 - mask_3ch)
                mask = image.inverted_mask_3ch[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                patch_rgb = patch_rgb * mask

                for i, node_label_id in enumerate(node.pruned_labels):

                    node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)
                    patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size, node_label_y_coord: node_label_y_coord + image.patch_size, :]
                    patchs_label_rgb = patchs_label_rgb * mask

                    node.label_cost[i] = rmse(patch_rgb, patchs_label_rgb)

            node.local_likelihood = [math.exp(-cost * (1/100000)) for cost in node.label_cost]
            node.mask = node.local_likelihood.index(max(node.local_likelihood))

    elif image.inpainting_approach == Image2BInpainted.USING_IR or image.inpainting_approach == Image2BInpainted.USING_STORED_DESCRIPTORS_HALVES:

        if image.patch_size % POOL_SIZE == 0:

            nr_channels = image.ir.shape[2]

            for node in nodes.values():

                node.label_cost = [0 for _ in range(max_nr_labels)]

                if node.overlap_source_region:

                    patch_ir = image.ir[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                    # mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
                    # mask_3ch = np.repeat(mask, nr_channels, axis=1).reshape((image.patch_size, image.patch_size, nr_channels))
                    # patch_ir = patch_ir * (1 - mask_3ch)
                    mask = image.inverted_mask_Nch[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                    patch_ir = patch_ir * mask
                    patch_descr = max_pool(patch_ir)

                    for i, node_label_id in enumerate(node.pruned_labels):
                        node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)
                        patchs_label_ir = image.ir[node_label_x_coord: node_label_x_coord + image.patch_size, node_label_y_coord: node_label_y_coord + image.patch_size, :]
                        patchs_label_ir = patchs_label_ir * mask
                        patchs_label_descr = max_pool(patchs_label_ir)

                        node.label_cost[i] = rmse(patch_descr, patchs_label_descr)

                node.local_likelihood = [math.exp(-cost * (1 / 100000)) for cost in node.label_cost]
                node.mask = node.local_likelihood.index(max(node.local_likelihood))

        else:

            nr_channels = image.ir.shape[2]
            # calculating padding parameters for the max pooling
            padding_height_total = POOL_SIZE - (image.patch_size % POOL_SIZE)
            padding_width_total = POOL_SIZE - (image.patch_size % POOL_SIZE)
            padding_height_left = padding_height_total // 2
            padding_height_right = padding_height_total - padding_height_left
            padding_width_left = padding_width_total // 2
            padding_width_right = padding_width_total - padding_width_left

            for node in nodes.values():

                node.label_cost = [0 for _ in range(max_nr_labels)]

                if node.overlap_source_region:

                    patch_ir = image.ir[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                    # mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
                    # mask_3ch = np.repeat(mask, nr_channels, axis=1).reshape((image.patch_size, image.patch_size, nr_channels))
                    # patch_ir = patch_ir * (1 - mask_3ch)
                    mask = image.inverted_mask_Nch[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
                    patch_ir = patch_ir * mask
                    patch_descr = max_pool_padding(patch_ir, padding_height_left, padding_height_right, padding_width_left, padding_width_right)

                    for i, node_label_id in enumerate(node.pruned_labels):
                        node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)
                        patchs_label_ir = image.ir[node_label_x_coord: node_label_x_coord + image.patch_size, node_label_y_coord: node_label_y_coord + image.patch_size, :]
                        patchs_label_ir = patchs_label_ir * mask
                        patchs_label_descr = max_pool_padding(patchs_label_ir, padding_height_left, padding_height_right, padding_width_left, padding_width_right)

                        node.label_cost[i] = rmse(patch_descr, patchs_label_descr)

                node.local_likelihood = [math.exp(-cost * (1 / 100000)) for cost in node.label_cost]
                node.mask = node.local_likelihood.index(max(node.local_likelihood))

    else:
        raise AssertionError("Inpainting approach has not been properly set.")


#TODO also calculate InitMask after local_likelihood


#TODO check what happens if there's less than max_nr_labels even without pruning


# -- 3rd phase --
# inference
# (???)

def neighborhood_consensus_message_passing(image, max_nr_labels, max_nr_iterations):

    global nodes

    # initialisation of the nodes' beliefs and messages
    for node in nodes.values():

        #node.messages = [1 for i in range(max_nr_labels)]
        node.messages = np.ones(max_nr_labels)

        #node.beliefs = [0 for i in range(max_nr_labels)]
        node.beliefs = np.zeros(max_nr_labels)
        node.beliefs[node.mask] = 1

        node.beliefs_new = node.beliefs.copy()

    # TODO implement convergence check (what's the criteria?) and then change this for-loop to a while-loop

    for i in range(max_nr_iterations):

        for node in nodes.values():

            neighbor_up, neighbor_down, neighbor_left, neighbor_right = get_neighbor_nodes(node, image)

            #TODO big problem! they are overriding each other!!
            # TODO implement an overwrite condition?
            if neighbor_up is not None:
                node.messages = np.matmul(node.potential_matrix_up, neighbor_up.beliefs.reshape((max_nr_labels, 1)))
            if neighbor_down is not None:
                node.messages = np.matmul(node.potential_matrix_down, neighbor_down.beliefs.reshape((max_nr_labels, 1)))
            if neighbor_left is not None:
                node.messages = np.matmul(node.potential_matrix_left, neighbor_left.beliefs.reshape((max_nr_labels, 1)))
            if neighbor_right is not None:
                node.messages = np.matmul(node.potential_matrix_right, neighbor_right.beliefs.reshape((max_nr_labels, 1)))

            node.messages = np.array([math.exp(-message * (1 / 100000)) for message in
                                       node.messages.reshape((max_nr_labels, 1))]).reshape(1, max_nr_labels)

            node.beliefs_new = np.multiply(node.messages, node.local_likelihood)
            node.beliefs_new = node.beliefs_new / node.beliefs_new.sum()  # normalise to sum up to 1

        # update the mask and beliefs for all the nodes
        for node in nodes.values():
            node.mask = node.beliefs_new.argmax()
            node.beliefs = node.beliefs_new


def generate_inpainted_image(image, blend_method=1, mask_type=1):
    """
    
    :param image:
    :param blend_method: Either 0 or 1
    :param mask_type: Either 0 or 1
    :return:
    """

    global nodes
    global nodes_count
    global nodes_order
    
    assert blend_method in [0, 1], 'blend_method should be either 0 or 1'
    assert mask_type in [0, 1], 'mask_type should be either 0 or 1'

    target_region = np.copy(image.mask).astype('bool')
    original_mask = np.copy(image.mask).astype('bool')
    
    cyan = np.reshape([0, 255, 255], (1, 1, 3))
    image.inpainted = np.copy(image.rgb)
    image.inpainted[target_region, :] = cyan    # To make clear in debugging mode

    if mask_type == 0:
        filter_size = max(2, image.patch_size // 2)  # should be > 1
        smooth_filter = generate_smooth_filter(filter_size)
        
        blend_mask = generate_blend_mask(image.patch_size)
        blend_mask = signal.convolve2d(blend_mask, smooth_filter, boundary='symm', mode='same')
    elif mask_type == 1:
        blend_mask = generate_linear_diamond_mask(image.patch_size)
    else:
        blend_mask = None
    
    blend_mask_rgb = np.repeat(blend_mask[..., None], 3, axis=2)
    for i in range(len(nodes_order)):
    # for i in range(len(nodes_order) - 1, -1, -1):

        node_id = nodes_order[i]
        node = nodes[node_id]

        node_mask_patch_x_coord, node_mask_patch_y_coord =  position_to_coordinates(node.pruned_labels[node.mask], image.height, image.patch_size)

        node_rgb = image.inpainted[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]

        node_rgb_new = image.inpainted[node_mask_patch_x_coord: node_mask_patch_x_coord + image.patch_size, node_mask_patch_y_coord: node_mask_patch_y_coord + image.patch_size, :]

        if blend_method == 0:
            image.inpainted[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :] =\
                node_rgb*blend_mask_rgb + node_rgb_new*(1 - blend_mask_rgb)
        
        # Only inpaint/update pixels belonging to mask
        elif blend_method == 1:
            mask_new = target_region[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
            mask_new_orig = original_mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]

            # only inpaint the mask part
            image.inpainted[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :][mask_new]=\
                (node_rgb_new)[mask_new]   # TODO

            # average out with previous values
            mask_prev = np.logical_and(mask_new_orig, np.logical_not(mask_new))
            
            image.inpainted[node.x_coord: node.x_coord + image.patch_size,
                            node.y_coord: node.y_coord + image.patch_size, :][mask_prev] = \
                (node_rgb*blend_mask_rgb + node_rgb_new*(1 - blend_mask_rgb))[mask_prev]
                
        else:
            ValueError(f'Unknown inpainting strategy: {blend_method}')

        # update the mask
        target_region[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size] = False

        # plt.imshow(image.inpainted.astype(np.uint8), interpolation='nearest')
        # plt.show()
        # imageio.imwrite('/home/niaki/Code/inpynting_images/building/ordering2/building_' + str(i).zfill(4) + '.png', image.inpainted)

    image.inpainted = image.inpainted.astype(np.uint8)


# def generate_inpainted_image_inverse_order(image):
#
#         global nodes
#         global nodes_count
#         global nodes_order
#
#         target_region = image.mask
#
#         image.inpainted = np.multiply(image.rgb,
#                                       np.repeat(1 - target_region, 3, axis=1).reshape((image.height, image.width, 3)))
#
#         filter_size = 4  # should be > 1
#         smooth_filter = generate_smooth_filter(filter_size)
#         blend_mask = generate_blend_mask(image.patch_size)
#         blend_mask = signal.convolve2d(blend_mask, smooth_filter, boundary='symm', mode='same')
#         blend_mask_rgb = np.repeat(blend_mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))
#
#         # for i in range(len(nodes_order)):
#         for i in range(len(nodes_order) - 1, -1, -1):
#             patch_id = nodes_order[i]
#             patch = nodes[patch_id]
#
#             patchs_mask_patch = nodes[patch.pruned_labels[patch.mask]]
#
#             patch_rgb = image.inpainted[patch.x_coord: patch.x_coord + image.patch_size,
#                         patch.y_coord: patch.y_coord + image.patch_size, :]
#
#             patch_rgb_new = image.inpainted[patchs_mask_patch.x_coord: patchs_mask_patch.x_coord + image.patch_size,
#                             patchs_mask_patch.y_coord: patchs_mask_patch.y_coord + image.patch_size, :]
#
#             image.inpainted[patch.x_coord: patch.x_coord + image.patch_size,
#             patch.y_coord: patch.y_coord + image.patch_size, :] = np.multiply(patch_rgb, blend_mask_rgb) + \
#                                                                   np.multiply(patch_rgb_new, 1 - blend_mask_rgb)
#
#             target_region[patch.x_coord: patch.x_coord + image.patch_size,
#             patch.y_coord: patch.y_coord + image.patch_size] = 0
#
#             # plt.imshow(image.inpainted.astype(np.uint8), interpolation='nearest')
#             # plt.show()
#             # imageio.imwrite('/home/niaki/Code/inpynting_images/Tijana/Jian10_uint8/ordering_process1/Jian10_' + str(i).zfill(4) + '.png', image.inpainted)
#
#         image.inpainted = image.inpainted.astype(np.uint8)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def generate_inpainted_image_blended(image):
#
#     global nodes
#     global nodes_count
#     global nodes_order
#
#     target_region = image.mask
#
#     image.inpainted = np.multiply(image.rgb,
#                                   np.repeat(1 - target_region, 3, axis=1).reshape((image.height, image.width, 3)))
#
#     filter_size = 2  # should be > 1
#     smooth_filter = generate_smooth_filter(filter_size)
#     blend_mask = generate_blend_mask_diamond(image.patch_size)  # here it's "diamond" blend mask
#     blend_mask = signal.convolve2d(blend_mask, smooth_filter, boundary='symm', mode='same')
#     blend_mask_rgb = np.repeat(blend_mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))
#
#     for i in range(len(nodes_order)):
#
#         patch_id = nodes_order[i]
#         patch = nodes[patch_id]
#
#         patchs_mask_patch = nodes[patch.pruned_labels[patch.mask]]
#
#         patch_rgb = image.inpainted[patch.x_coord: patch.x_coord + image.patch_size,
#                     patch.y_coord: patch.y_coord + image.patch_size, :]
#
#         patch_rgb_new = image.inpainted[patchs_mask_patch.x_coord: patchs_mask_patch.x_coord + image.patch_size,
#                         patchs_mask_patch.y_coord: patchs_mask_patch.y_coord + image.patch_size, :]
#
#         image.inpainted[patch.x_coord: patch.x_coord + image.patch_size,
#             patch.y_coord: patch.y_coord + image.patch_size, :] = np.multiply(patch_rgb, 1 - blend_mask_rgb) + \
#                                                              np.multiply(patch_rgb_new, blend_mask_rgb)
#
#         target_region[patch.x_coord: patch.x_coord + image.patch_size, patch.y_coord: patch.y_coord + image.patch_size] = 0
#
#         # plt.imshow(image.inpainted.astype(np.uint8), interpolation='nearest')
#         # plt.show()
#         # imageio.imwrite('/home/niaki/Code/inpynting_images/Tijana/Jian10_uint8/ordering_process1/Jian10_' + str(i).zfill(4) + '.png', image.inpainted)
#
#     image.inpainted = image.inpainted.astype(np.uint8)
#
#
#
#










def generate_smooth_filter(kernel_size):

    if kernel_size <= 1:
        raise Exception('Kernel size for the smooth filter should be larger than 1, but is {}.'.format(kernel_size))

    kernel_1D = np.array([0.5, 0.5]).transpose()

    for i in range(kernel_size - 2):
        kernel_1D = np.convolve(np.array([0.5, 0.5]).transpose(), kernel_1D)

    kernel_1D = kernel_1D.reshape((kernel_size, 1))

    kernel_2D = np.matmul(kernel_1D, kernel_1D.transpose())

    return kernel_2D


def generate_blend_mask_diamond(patch_size):

    patch_size_half = patch_size // 2

    blend_mask_quarter1 = np.ones((patch_size_half, patch_size_half))
    for i in range(patch_size_half):
        for j in range(patch_size_half):
            if (i + j) < patch_size_half:
                blend_mask_quarter1[i, j] = 0
    blend_mask_quarter4 = 1 - blend_mask_quarter1

    blend_mask_quarter2 = np.ones((patch_size_half, patch_size_half))
    for i in range(patch_size_half):
        for j in range(patch_size_half):
            if i < j:
                blend_mask_quarter2[i, j] = 0
    blend_mask_quarter3 = 1 - blend_mask_quarter2

    blend_mask = np.zeros((patch_size, patch_size))
    blend_mask[: patch_size_half, : patch_size_half] = blend_mask_quarter1
    blend_mask[: patch_size_half, patch_size_half:] = blend_mask_quarter2
    blend_mask[patch_size_half:, : patch_size_half] = blend_mask_quarter3
    blend_mask[patch_size_half:, patch_size_half:] = blend_mask_quarter4

    return blend_mask


#TODO this is just a hacky way to implement something similar to what is needed
def generate_blend_mask(patch_size):

    blend_mask = np.zeros((patch_size, patch_size))
    blend_mask[:patch_size // 3, :] = 1
    blend_mask[:, :patch_size // 3] = 1

    return blend_mask


def generate_linear_diamond_mask(patch_size):
    # Does NOT need post filtering

    blend_mask = np.zeros((patch_size, patch_size))
    
    # Even e.g. 8: from 0 to 3 (and 4 to 7)
    # Uneven e.g 7: from 0 to 3 (and 3 to 6)
    patch_size_half = int(np.ceil(patch_size/2.))
    
    for i in range(patch_size_half):
        for j in range(patch_size_half):
            val = (i + j) / (patch_size_half - 1 + patch_size_half - 1)

            blend_mask[i, j] = val
            blend_mask[patch_size-1-i, j] = val
            blend_mask[i, patch_size-1-j] = val
            blend_mask[patch_size-1-i, patch_size-1-j] = val

    return blend_mask


def generate_order_image(image):
    global nodes
    global nodes_count
    global nodes_order

    order_image = np.multiply(image.rgb,
                                  np.repeat(1 - image.mask, 3, axis=1).reshape((image.height, image.width, 3)))
    nr_nodes = len(nodes_order)
    for i in range(nr_nodes):
        node = nodes[nodes_order[i]]
        pixel_value = math.floor(i * 255 / nr_nodes)
        for j in range(3):
            order_image[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, j] = pixel_value

    order_image = order_image.astype(np.uint8)
    image.order_image = order_image


def pickle_global_vars(file_version):
    global nodes
    global nodes_count
    global nodes_order

    pickle_patches_file_path = '/home/niaki/Code/inpynting_images/pickles/eeo_global_vars_' + file_version + '.pickle'
    try:
        pickle.dump((nodes, nodes_count, nodes_order), open(pickle_patches_file_path, "wb"))
    except Exception as e:
        print("Problem while trying to pickle: ", str(e))


def unpickle_global_vars(file_version):
    global nodes
    global nodes_count
    global nodes_order

    pickle_patches_file_path = '/home/niaki/Code/inpynting_images/pickles/eeo_global_vars_' + file_version + '.pickle'
    try:
        nodes, nodes_count, nodes_order = pickle.load(open(pickle_patches_file_path, "rb"))
    except Exception as e:
        print("Problem while trying to unpickle: ", str(e))


# def visualise_nodes_priorities(image):
#     global nodes
#
#     priority_image = np.multiply(image.rgb,
#                               np.repeat(1 - image.mask, 3, axis=1).reshape((image.height, image.width, 3)))
#
#     max_priority = 0
#     for patch in nodes:
#         if patch.overlap_target_region:
#
#             if patch.priority > max_priority:
#                 max_priority = patch.priority
#
#     for patch in nodes:
#         if patch.overlap_target_region:
#
#             pixel_value = math.floor(patch.priority * 255 / max_priority)
#             for j in range(3):
#                 priority_image[patch.x_coord: patch.x_coord + image.patch_size,
#                 patch.y_coord: patch.y_coord + image.patch_size, j] = pixel_value
#
#     priority_image = priority_image.astype(np.uint8)
#
#     return priority_image
#
