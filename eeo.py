import numpy as np
import sys
import pickle
import math
from scipy import signal
import matplotlib.pyplot as plt
import imageio

from data_structures import Patch
from data_structures import UP, DOWN, LEFT, RIGHT
from patch_diff import non_masked_patch_diff, half_patch_diff


patches = []  # the indices in this list patches match the patch_id
nodes_count = 0
nodes_order = []

temps_NOT = np.zeros((103, 4000))
temps_NOT_last_index = 0
temps_FULLY = np.zeros((99, 4000))
temps_FULLY_last_index = 0

# -- 1st phase --
# initialization
# (assigning priorities to MRF nodes to be used for determining the visiting order in the 2nd phase)
def initialization(image, thresh_uncertainty):

    global patches
    global nodes_count

    patch_id_counter = 0

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

            patch = Patch(patch_id_counter, patch_overlap_source_region, patch_overlap_target_region, x, y)

            patches.append(patch)
            patch_id_counter += 1

    for patch in patches:

        if patch.overlap_target_region:
            # sys.stdout.write("\rInitialising node " + str(nodes_count + 1))

            if patch.overlap_source_region:

                # compare the patch to all patches that are completely in the source region
                for patch_compare in patches:
                    if patch_compare.overlap_source_region and not patch_compare.overlap_target_region:

                        patch_difference = non_masked_patch_diff(image, patch.x_coord, patch.y_coord, patch_compare.x_coord, patch_compare.y_coord)

                        patch.differences[patch_compare.patch_id] = patch_difference

                        patch.labels.append(patch_compare.patch_id)

                temp_min_diff = min(list(patch.differences.values()))
                temp = [value - temp_min_diff for value in list(patch.differences.values())]
                #TODO change thresh_uncertainty such that only patches which are completely in the target region
                #     get assigned the priority value 1.0 (but keep in mind it is used elsewhere)
                patch_uncertainty = len([val for (i, val) in enumerate(temp) if val < thresh_uncertainty])

            # if the patch is completely in the target region
            else:

                # make all patches that are completely in the source region be the label of the patch
                for patch_compare in patches:
                    if patch_compare.overlap_source_region and not patch_compare.overlap_target_region:

                        patch.differences[patch_compare.patch_id] = 0

                        patch.labels.append(patch_compare.patch_id)

                patch_uncertainty = len(patch.labels)

            # the higher priority the higher priority :D
            patch.priority = len(patch.labels) / max(patch_uncertainty, 1)

            print(patch.patch_id, patch_uncertainty, patch.priority)

            nodes_count +=1

            # if nodes_count == 7:
            #     break

    print("\nTotal number of patches: ", len(patches))
    print("Number of patches to be inpainted: ", nodes_count)


# -- 2nd phase --
# label pruning
# (reducing the number of labels at each node to a relatively small number)

def label_pruning(image, thresh_uncertainty, max_nr_labels):
    global patches
    global nodes_count
    global nodes_order

    # make a copy of the differences which we can edit and use in this method, and afterwards discard
    for patch in patches:
        if patch.overlap_target_region:
            patch.additional_differences = patch.differences.copy()


    # for all the patches that have an overlap with the target region (aka nodes)
    for i in range(nodes_count):

        # print()
        # print("Uncomitted nodes' IDs and priorities:")
        # for patch in patches:
        #     if patch.overlap_target_region:
                # print("{:.2f}".format(patch.priority), end=" ")
        # print()

        # find the node with the highest priority that hasn't yet been visited
        highest_priority = -1
        patch_highest_priority_id = -1
        for patch in patches:
            if patch.overlap_target_region and not patch.committed and patch.priority > highest_priority:
                highest_priority = patch.priority
                patch_highest_priority_id = patch.patch_id

        patch = patches[patch_highest_priority_id]
        patch.committed = True

        patch.prune_labels(max_nr_labels)

        print('Highest priority patch {0:3d}/{1:3d}: {2:d}'.format(i + 1, nodes_count, patch_highest_priority_id))
        nodes_order.append(patch_highest_priority_id)

        # get the neighbors if they exist and have overlap with the target region
        patch_neighbor_up, patch_neighbor_down, patch_neighbor_left, patch_neighbor_right = get_patch_neighbor_nodes(
            patch, image)

        update_patchs_neighbors_priority(patch, patch_neighbor_up, UP, image, thresh_uncertainty)
        update_patchs_neighbors_priority(patch, patch_neighbor_down, DOWN, image, thresh_uncertainty)
        update_patchs_neighbors_priority(patch, patch_neighbor_left, LEFT, image, thresh_uncertainty)
        update_patchs_neighbors_priority(patch, patch_neighbor_right, RIGHT, image, thresh_uncertainty)


def update_patchs_neighbors_priority(patch, patch_neighbor, side, image, thresh_uncertainty):
    global temps_NOT_last_index
    global temps_FULLY_last_index

    # thresh_uncertainty = thresh_uncertainty // 5

    if patch_neighbor is not None and not patch_neighbor.committed:

        min_additional_difference = sys.maxsize
        additional_differences = {}

        for patch_neighbors_label_id in patch_neighbor.labels:

            patch_neighbors_label_x_coord = patches[patch_neighbors_label_id].x_coord
            patch_neighbors_label_y_coord = patches[patch_neighbors_label_id].y_coord

            # patch_neighbors_label_rgb = image.rgb[
            #                             patch_neighbors_label_x_coord: patch_neighbors_label_x_coord + image.patch_size,
            #                             patch_neighbors_label_y_coord: patch_neighbors_label_y_coord + image.patch_size, :]
            # patch_neighbors_label_rgb_half = get_half_patch_from_patch(patch_neighbors_label_rgb, image.stride, opposite_side(side))

            for patchs_label_id in patch.pruned_labels:

                patchs_label_x_coord = patches[patchs_label_id].x_coord
                patchs_label_y_coord = patches[patchs_label_id].y_coord

                # patchs_label_rgb = image.rgb[patchs_label_x_coord: patchs_label_x_coord + image.patch_size,
                #                    patchs_label_y_coord: patchs_label_y_coord + image.patch_size, :]
                # patchs_label_rgb_half = get_half_patch_from_patch(patchs_label_rgb, image.stride, side)
                # difference = patch_diff(patch_neighbors_label_rgb_half, patchs_label_rgb_half)

                difference = half_patch_diff(image, patchs_label_x_coord, patchs_label_y_coord, patch_neighbors_label_x_coord, patch_neighbors_label_y_coord, side)

                if difference < min_additional_difference:
                    min_additional_difference = difference

            additional_differences[patch_neighbors_label_id] = min_additional_difference
            min_additional_difference = sys.maxsize

        for key in additional_differences.keys():
            if key in patch_neighbor.additional_differences:
                patch_neighbor.additional_differences[key] += additional_differences[key]
            else:
                patch_neighbor.additional_differences[key] = additional_differences[key]
                print("Will it ever come to this? (2) (TODO delete if unnecessary)")

        temp_min_diff = min(patch_neighbor.additional_differences.values())
        temp = [value - temp_min_diff for value in
                list(patch_neighbor.additional_differences.values())]
        patch_neighbor_uncertainty = [value < thresh_uncertainty for (i, value) in enumerate(temp)].count(True)

        patch_neighbor.priority = len(patch_neighbor.additional_differences) / patch_neighbor_uncertainty #len(patch_neighbor.differences)?

        # print('### patch neighbour:', 'NOT completely under mask,' if patch_neighbor.overlap_source_region else 'FULLY under mask,', 'uncert', patch_neighbor_uncertainty, ', priority', patch_neighbor.priority)
        # print('NOT' if patch_neighbor.overlap_source_region else 'FULLY', end=',', file=open("/home/niaki/Downloads/temps.txt", "a"))
        # print(*temp, sep = ",", file=open("/home/niaki/Downloads/temps.txt", "a"))

        if patch_neighbor.overlap_source_region:
            # print(*temp, sep=",", file=open("/home/niaki/Downloads/temps_NOT.txt", "a"))
            temp1 = np.zeros(4000)
            temp1[: len(temp)] = np.array(temp)

            temps_NOT[temps_NOT_last_index, :] = temp1
            temps_NOT_last_index += 1
        else:
            # print(*temp, sep=",", file=open("/home/niaki/Downloads/temps_FULLY.txt", "a"))
            temp1 = np.zeros(4000)
            temp1[: len(temp)] = np.array(temp)

            temp = np.array(temp)
            temps_FULLY[temps_FULLY_last_index, :] = temp1
            temps_FULLY_last_index +=1

        pass

def get_patch_neighbor_nodes(patch, image):

    patch_neighbor_up_id = patch.get_up_neighbor_position(image)
    if patch_neighbor_up_id is None:
        patch_neighbor_up = None
    else:
        patch_neighbor_up = patches[patch_neighbor_up_id]
        if not patch_neighbor_up.overlap_target_region:
            patch_neighbor_up = None

    patch_neighbor_down_id = patch.get_down_neighbor_position(image)
    if patch_neighbor_down_id is None:
        patch_neighbor_down = None
    else:
        patch_neighbor_down = patches[patch_neighbor_down_id]
        if not patch_neighbor_down.overlap_target_region:
            patch_neighbor_down = None

    patch_neighbor_left_id = patch.get_left_neighbor_position(image)
    if patch_neighbor_left_id is None:
        patch_neighbor_left = None
    else:
        patch_neighbor_left = patches[patch_neighbor_left_id]
        if not patch_neighbor_left.overlap_target_region:
            patch_neighbor_left = None

    patch_neighbor_right_id = patch.get_right_neighbor_position(image)
    if patch_neighbor_right_id is None:
        patch_neighbor_right = None
    else:
        patch_neighbor_right = patches[patch_neighbor_right_id]
        if not patch_neighbor_right.overlap_target_region:
            patch_neighbor_right = None
    return patch_neighbor_up, patch_neighbor_down, patch_neighbor_left, patch_neighbor_right


def compute_pairwise_potential_matrix(image, max_nr_labels):

    pickle.dump((temps_NOT, temps_FULLY), open('/home/niaki/Downloads/temps.pickle', "wb"))


    global patches
    global nodes_count

    # calculate pairwise potential matrix for all pairs of nodes (pathces that have overlap with the target region)
    for patch in patches:
        if patch.overlap_target_region:

            # get the neighbors if they exist and have overlap with the target region (i.e. if they're nodes)
            patch_neighbor_up, _, patch_neighbor_left, _ = get_patch_neighbor_nodes(patch, image)

            # TODO make this a method
            if patch_neighbor_up is not None:

                potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

                for i, patchs_label_id in enumerate(patch.pruned_labels):

                    patchs_label_x_coord = patches[patchs_label_id].x_coord
                    patchs_label_y_coord = patches[patchs_label_id].y_coord

                    # patchs_label_rgb = image.rgb[patchs_label_x_coord: patchs_label_x_coord + image.patch_size,
                    #                    patchs_label_y_coord: patchs_label_y_coord + image.patch_size, :]

                    # patchs_label_rgb_up = get_half_patch_from_patch(patchs_label_rgb, image.stride, UP)

                    for j, patchs_neighbors_label_id in enumerate(patch_neighbor_up.pruned_labels):

                        patchs_neighbors_label_x_coord = patches[patchs_neighbors_label_id].x_coord
                        patchs_neighbors_label_y_coord = patches[patchs_neighbors_label_id].y_coord

                        # patchs_neighbors_label_rgb = image.rgb[patchs_neighbors_label_x_coord: patchs_neighbors_label_x_coord + image.patch_size,
                        #                              patchs_neighbors_label_y_coord: patchs_neighbors_label_y_coord + image.patch_size, :]

                        # patchs_neighbors_label_rgb_down = get_half_patch_from_patch(patchs_neighbors_label_rgb, image.stride, DOWN)

                        # potential_matrix[i, j] = patch_diff(patchs_label_rgb_up, patchs_neighbors_label_rgb_down)
                        potential_matrix[i, j] = half_patch_diff(image, patchs_label_x_coord, patchs_label_y_coord, patchs_neighbors_label_x_coord, patchs_neighbors_label_y_coord, UP)

                patch.potential_matrix_up = potential_matrix
                patch_neighbor_up.potential_matrix_down = potential_matrix
                # print("potential matrix UpDown")
                # print(potential_matrix)

            if patch_neighbor_left is not None:

                potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

                for i, patchs_label_id in enumerate(patch.pruned_labels):

                    patchs_label_x_coord = patches[patchs_label_id].x_coord
                    patchs_label_y_coord = patches[patchs_label_id].y_coord

                    # patchs_label_rgb = image.rgb[patchs_label_x_coord: patchs_label_x_coord + image.patch_size,
                    #                    patchs_label_y_coord: patchs_label_y_coord + image.patch_size, :]

                    # patchs_label_rgb_left = get_half_patch_from_patch(patchs_label_rgb, image.stride, LEFT)

                    for j, patchs_neighbors_label_id in enumerate(patch_neighbor_left.pruned_labels):

                        patchs_neighbors_label_x_coord = patches[patchs_neighbors_label_id].x_coord
                        patchs_neighbors_label_y_coord = patches[patchs_neighbors_label_id].y_coord

                        # patchs_neighbors_label_rgb = image.rgb[
                        #                              patchs_neighbors_label_x_coord: patchs_neighbors_label_x_coord + image.patch_size,
                        #                              patchs_neighbors_label_y_coord: patchs_neighbors_label_y_coord + image.patch_size,
                        #                              :]

                        # patchs_neighbors_label_rgb_right = get_half_patch_from_patch(patchs_neighbors_label_rgb, image.stride,
                        #                                                             RIGHT)

                        # potential_matrix[i, j] = patch_diff(patchs_label_rgb_left, patchs_neighbors_label_rgb_right)
                        potential_matrix[i, j] = half_patch_diff(image, patchs_label_x_coord, patchs_label_y_coord,
                                                                 patchs_neighbors_label_x_coord,
                                                                 patchs_neighbors_label_y_coord, LEFT)

                patch.potential_matrix_left = potential_matrix
                patch_neighbor_left.potential_matrix_right = potential_matrix
                # print("potential matrix LeftRight")
                # print(potential_matrix)


def compute_label_cost(image, max_nr_labels):

    global patches
    global nodes_count

    for patch in patches:
        if patch.overlap_target_region:

            patch.label_cost = [0 for _ in range(max_nr_labels)]

            if patch.overlap_source_region:

                # patch_rgb = image.rgb[patch.x_coord : patch.x_coord + patch_size,
                #             patch.y_coord : patch.y_coord + patch_size, :]

                for i, patchs_label_id in enumerate(patch.pruned_labels):

                    # patchs_label_rgb = image.rgb[patches[patchs_label_id].x_coord: patches[patchs_label_id].x_coord + patch_size,
                    #                    patches[patchs_label_id].y_coord: patches[patchs_label_id].y_coord + patch_size, :]

                    # patch.label_cost[i] = patch_diff(patch_rgb, patchs_label_rgb)
                    patch.label_cost[i] = non_masked_patch_diff(image, patch.x_coord, patch.y_coord,
                                          patches[patchs_label_id].x_coord, patches[patchs_label_id].y_coord)


            patch.local_likelihood = [math.exp(-cost * (1/100000)) for cost in patch.label_cost]
            # print("patch", patch.patch_id, "label cost", patch.label_cost)
            # print("patch", patch.patch_id, "local likelihood", patch.local_likelihood)

            patch.mask = patch.local_likelihood.index(max(patch.local_likelihood))


#TODO also calculate InitMask after local_likelihood


#TODO check what happens if there's less than max_nr_labels even without pruning


# -- 3rd phase --
# inference
# (???)

def neighborhood_consensus_message_passing(image, max_nr_labels, max_nr_iterations):

    global patches

    # initialisation of the nodes' beliefs and messages
    for patch in patches:
        if patch.overlap_target_region:

            #patch.messages = [1 for i in range(max_nr_labels)]
            patch.messages = np.ones(max_nr_labels)

            #patch.beliefs = [0 for i in range(max_nr_labels)]
            patch.beliefs = np.zeros(max_nr_labels)
            patch.beliefs[patch.mask] = 1

            patch.beliefs_new = patch.beliefs.copy()

    # TODO implement convergence check (what's the criteria?) and then change this for-loop to a while-loop

    for i in range(max_nr_iterations):

        for patch in patches:
            if patch.overlap_target_region:

                patch_neighbor_up, patch_neighbor_down, patch_neighbor_left, patch_neighbor_right = get_patch_neighbor_nodes(
                    patch, image)

                #TODO big problem! they are overriding each other!!
                if patch_neighbor_up is not None:
                    patch.messages = np.matmul(patch.potential_matrix_up, patch_neighbor_up.beliefs.reshape((max_nr_labels, 1)))
                if patch_neighbor_down is not None:
                    patch.messages = np.matmul(patch.potential_matrix_down, patch_neighbor_down.beliefs.reshape((max_nr_labels, 1)))
                if patch_neighbor_left is not None:
                    patch.messages = np.matmul(patch.potential_matrix_left, patch_neighbor_left.beliefs.reshape((max_nr_labels, 1)))
                if patch_neighbor_right is not None:
                    patch.messages = np.matmul(patch.potential_matrix_right, patch_neighbor_right.beliefs.reshape((max_nr_labels, 1)))

                patch.messages = np.array([math.exp(-message * (1 / 100000)) for message in
                                           patch.messages.reshape((max_nr_labels, 1))]).reshape(1, max_nr_labels)

                patch.beliefs_new = np.multiply(patch.messages, patch.local_likelihood)
                patch.beliefs_new = patch.beliefs_new / patch.beliefs_new.sum()  # normalise to sum up to 1

        # update the mask and beliefs for all the nodes
        for patch in patches:
            if patch.overlap_target_region:

                patch.mask = patch.beliefs_new.argmax()
                patch.beliefs = patch.beliefs_new


def generate_inpainted_image(image):

    global patches
    global nodes_count
    global nodes_order

    target_region = image.mask

    image.inpainted = np.multiply(image.rgb,
                                  np.repeat(1 - target_region, 3, axis=1).reshape((image.height, image.width, 3)))

    filter_size = 4  # should be > 1
    smooth_filter = generate_smooth_filter(filter_size)
    blend_mask = generate_blend_mask(image.patch_size)
    blend_mask = signal.convolve2d(blend_mask, smooth_filter, boundary='symm', mode='same')
    blend_mask_rgb = np.repeat(blend_mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))

    for i in range(len(nodes_order)):
    # for i in range(len(nodes_order) - 1, -1, -1):

        patch_id = nodes_order[i]
        patch = patches[patch_id]

        patchs_mask_patch = patches[patch.pruned_labels[patch.mask]]

        patch_rgb = image.inpainted[patch.x_coord: patch.x_coord + image.patch_size, patch.y_coord: patch.y_coord + image.patch_size, :]

        patch_rgb_new = image.inpainted[patchs_mask_patch.x_coord: patchs_mask_patch.x_coord + image.patch_size, patchs_mask_patch.y_coord: patchs_mask_patch.y_coord + image.patch_size, :]

        image.inpainted[patch.x_coord: patch.x_coord + image.patch_size, patch.y_coord: patch.y_coord + image.patch_size, :] =\
            np.multiply(patch_rgb, blend_mask_rgb) + np.multiply(patch_rgb_new, 1 - blend_mask_rgb)

        target_region[patch.x_coord: patch.x_coord + image.patch_size, patch.y_coord: patch.y_coord + image.patch_size] = 0

        # plt.imshow(image.inpainted.astype(np.uint8), interpolation='nearest')
        # plt.show()
        # imageio.imwrite('/home/niaki/Code/inpynting_images/Tijana/Jian10_uint8/ordering_process1/Jian10_' + str(i).zfill(4) + '.jpg', image.inpainted)

    image.inpainted = image.inpainted.astype(np.uint8)


def generate_inpainted_image_inverse_order(image):

        global patches
        global nodes_count
        global nodes_order

        target_region = image.mask

        image.inpainted = np.multiply(image.rgb,
                                      np.repeat(1 - target_region, 3, axis=1).reshape((image.height, image.width, 3)))

        filter_size = 4  # should be > 1
        smooth_filter = generate_smooth_filter(filter_size)
        blend_mask = generate_blend_mask(image.patch_size)
        blend_mask = signal.convolve2d(blend_mask, smooth_filter, boundary='symm', mode='same')
        blend_mask_rgb = np.repeat(blend_mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))

        # for i in range(len(nodes_order)):
        for i in range(len(nodes_order) - 1, -1, -1):
            patch_id = nodes_order[i]
            patch = patches[patch_id]

            patchs_mask_patch = patches[patch.pruned_labels[patch.mask]]

            patch_rgb = image.inpainted[patch.x_coord: patch.x_coord + image.patch_size,
                        patch.y_coord: patch.y_coord + image.patch_size, :]

            patch_rgb_new = image.inpainted[patchs_mask_patch.x_coord: patchs_mask_patch.x_coord + image.patch_size,
                            patchs_mask_patch.y_coord: patchs_mask_patch.y_coord + image.patch_size, :]

            image.inpainted[patch.x_coord: patch.x_coord + image.patch_size,
            patch.y_coord: patch.y_coord + image.patch_size, :] = np.multiply(patch_rgb, blend_mask_rgb) + \
                                                                  np.multiply(patch_rgb_new, 1 - blend_mask_rgb)

            target_region[patch.x_coord: patch.x_coord + image.patch_size,
            patch.y_coord: patch.y_coord + image.patch_size] = 0

            # plt.imshow(image.inpainted.astype(np.uint8), interpolation='nearest')
            # plt.show()
            # imageio.imwrite('/home/niaki/Code/inpynting_images/Tijana/Jian10_uint8/ordering_process1/Jian10_' + str(i).zfill(4) + '.jpg', image.inpainted)

        image.inpainted = image.inpainted.astype(np.uint8)
















def generate_inpainted_image_blended(image):

    global patches
    global nodes_count
    global nodes_order

    target_region = image.mask

    image.inpainted = np.multiply(image.rgb,
                                  np.repeat(1 - target_region, 3, axis=1).reshape((image.height, image.width, 3)))

    filter_size = 2  # should be > 1
    smooth_filter = generate_smooth_filter(filter_size)
    blend_mask = generate_blend_mask_diamond(image.patch_size)  # here it's "diamond" blend mask
    blend_mask = signal.convolve2d(blend_mask, smooth_filter, boundary='symm', mode='same')
    blend_mask_rgb = np.repeat(blend_mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))

    for i in range(len(nodes_order)):

        patch_id = nodes_order[i]
        patch = patches[patch_id]

        patchs_mask_patch = patches[patch.pruned_labels[patch.mask]]

        patch_rgb = image.inpainted[patch.x_coord: patch.x_coord + image.patch_size,
                    patch.y_coord: patch.y_coord + image.patch_size, :]

        patch_rgb_new = image.inpainted[patchs_mask_patch.x_coord: patchs_mask_patch.x_coord + image.patch_size,
                        patchs_mask_patch.y_coord: patchs_mask_patch.y_coord + image.patch_size, :]

        image.inpainted[patch.x_coord: patch.x_coord + image.patch_size,
            patch.y_coord: patch.y_coord + image.patch_size, :] = np.multiply(patch_rgb, 1 - blend_mask_rgb) + \
                                                             np.multiply(patch_rgb_new, blend_mask_rgb)

        target_region[patch.x_coord: patch.x_coord + image.patch_size, patch.y_coord: patch.y_coord + image.patch_size] = 0

        # plt.imshow(image.inpainted.astype(np.uint8), interpolation='nearest')
        # plt.show()
        # imageio.imwrite('/home/niaki/Code/inpynting_images/Tijana/Jian10_uint8/ordering_process1/Jian10_' + str(i).zfill(4) + '.jpg', image.inpainted)

    image.inpainted = image.inpainted.astype(np.uint8)














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
    for i in range(patch_size // 3):
        blend_mask[i, :] = 1
        blend_mask[:, i] = 1

    return blend_mask


def generate_order_image(image):
    global patches
    global nodes_count
    global nodes_order

    order_image = np.multiply(image.rgb,
                                  np.repeat(1 - image.mask, 3, axis=1).reshape((image.height, image.width, 3)))
    nr_nodes = len(nodes_order)
    for i in range(nr_nodes):
        patch = patches[nodes_order[i]]
        pixel_value = math.floor(i * 255 / nr_nodes)
        for j in range(3):
            order_image[patch.x_coord: patch.x_coord + image.patch_size, patch.y_coord: patch.y_coord + image.patch_size, j] = pixel_value

    order_image = order_image.astype(np.uint8)
    image.order_image = order_image


def pickle_global_vars(file_version):
    global patches
    global nodes_count
    global nodes_order

    pickle_patches_file_path = '/home/niaki/Code/inpynting_images/pickles/eeo_global_vars_' + file_version + '.pickle'
    try:
        pickle.dump((patches, nodes_count, nodes_order), open(pickle_patches_file_path, "wb"))
    except Exception as e:
        print("Problem while trying to pickle: ", str(e))


def unpickle_global_vars(file_version):
    global patches
    global nodes_count
    global nodes_order

    pickle_patches_file_path = '/home/niaki/Code/inpynting_images/pickles/eeo_global_vars_' + file_version + '.pickle'
    try:
        patches, nodes_count, nodes_order = pickle.load(open(pickle_patches_file_path, "rb"))
    except Exception as e:
        print("Problem while trying to unpickle: ", str(e))


def visualise_nodes_priorities(image):
    global patches

    priority_image = np.multiply(image.rgb,
                              np.repeat(1 - image.mask, 3, axis=1).reshape((image.height, image.width, 3)))

    max_priority = 0
    for patch in patches:
        if patch.overlap_target_region:

            if patch.priority > max_priority:
                max_priority = patch.priority

    for patch in patches:
        if patch.overlap_target_region:

            pixel_value = math.floor(patch.priority * 255 / max_priority)
            for j in range(3):
                priority_image[patch.x_coord: patch.x_coord + image.patch_size,
                patch.y_coord: patch.y_coord + image.patch_size, j] = pixel_value

    priority_image = priority_image.astype(np.uint8)

    return priority_image
