import numpy as np
import sys
import matplotlib.pyplot as plt

from data_structures import Patch
from data_structures import UP, DOWN, LEFT, RIGHT, opposite_side, get_half_patch_from_patch
from patch_diff import patch_diff, non_masked_patch_diff
import math
from scipy import signal

# the indices in this list patches match the patch_id
patches = []
nodes_count = 0
nodes_order = []


# -- 1st phase --
# initialization
# (assigning priorities to MRF nodes to be used for determining the visiting order in the 2nd phase)
def initialization(image, patch_size, gap, THRESHOLD_UNCERTAINTY):

    global patches
    global nodes_count

    patch_id_counter = 0

    # for all the patches in an image (not all, but with $gap stride)
    for y in range(0, image.width - patch_size + 1, gap):
        for x in range(0, image.height - patch_size + 1, gap):

            patch_mask_overlap = image.mask[x: x + patch_size, y: y + patch_size]
            patch_mask_overlap_nonzero_elements = np.count_nonzero(patch_mask_overlap)

            # determine with which regions is the patch overlapping
            if patch_mask_overlap_nonzero_elements == 0:
                patch_overlap_source_region = True
                patch_overlap_target_region = False
            elif patch_mask_overlap_nonzero_elements == patch_size**2:
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

            if patch.overlap_source_region:

                # compare the patch to all patches that are completely in the source region
                for patch_compare in patches:
                    if patch_compare.overlap_source_region and not patch_compare.overlap_target_region:

                        patch_difference = non_masked_patch_diff(image, patch_size, patch.x_coord, patch.y_coord, patch_compare.x_coord, patch_compare.y_coord)

                        patch.differences[patch_compare.patch_id] = patch_difference

                        patch.labels.append(patch_compare.patch_id)

                temp = [value - min(list(patch.differences.values())) for value in list(patch.differences.values())]
                #TODO change THRESHOLD_UNCERTAINTY such that only patches which are completely in the target region
                #     get assigned the priority value 1.0 (but keep in mind it is used elsewhere)
                patch_uncertainty = len([val for (i, val) in enumerate(temp) if val < THRESHOLD_UNCERTAINTY])
                del temp

            # if the patch is completely in the target region
            else:

                # make all patches that are completely in the source region be the label of the patch
                for patch_compare in patches:
                    if patch_compare.overlap_source_region and not patch_compare.overlap_target_region:

                        patch.differences[patch_compare.patch_id] = 0

                        patch.labels.append(patch_compare.patch_id)

                        #TODO set differences to zeros

                patch_uncertainty = len(patch.labels)

                # TODO something mentioned in the other file (find_label_pos)

            # the higher priority the higher priority :D
            patch.priority = len(patch.labels) / max(patch_uncertainty, 1)

            nodes_count +=1

            # if nodes_count == 7:
            #     break

    print("Total number of patches: ", len(patches))
    print("Number of patches to be inpainted: ", nodes_count)


# -- 2nd phase --
# label pruning
# (reducing the number of labels at each node to a relatively small number)

def label_pruning(image, patch_size, gap, THRESHOLD_UNCERTAINTY, MAX_NB_LABELS):

    global patches
    global nodes_count
    global nodes_order

    # for all the patches that have an overlap with the target region (aka nodes)
    for i in range(nodes_count):
        # print()
        # print("Uncomitted nodes' IDs and priorities:")
        # for patch in patches:
        #     if patch.overlap_target_region and not patch.committed:
        #         print(patch.patch_id, patch.priority)
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

        patch.prune_labels(MAX_NB_LABELS)

        print('Highest priority patch {0:3d}/{1:3d}: {2:d}'.format(i + 1, nodes_count, patch_highest_priority_id))
        nodes_order.append(patch_highest_priority_id)

        # get the neighbors if they exist and have overlap with the target region
        patch_neighbor_up, patch_neighbor_down, patch_neighbor_left, patch_neighbor_right = get_patch_neighbor_nodes(
            patch, image, patch_size, gap)

        update_patchs_neighbors_priority(patch, patch_neighbor_up, UP, image, gap, patch_size, THRESHOLD_UNCERTAINTY)
        update_patchs_neighbors_priority(patch, patch_neighbor_down, DOWN, image, gap, patch_size, THRESHOLD_UNCERTAINTY)
        update_patchs_neighbors_priority(patch, patch_neighbor_left, LEFT, image, gap, patch_size, THRESHOLD_UNCERTAINTY)
        update_patchs_neighbors_priority(patch, patch_neighbor_right, RIGHT, image, gap, patch_size, THRESHOLD_UNCERTAINTY)


def get_patch_neighbor_nodes(patch, image, patch_size, gap):

    patch_neighbor_up_id = patch.get_up_neighbor_position(image, patch_size, gap)
    if patch_neighbor_up_id is None:
        patch_neighbor_up = None
    else:
        patch_neighbor_up = patches[patch_neighbor_up_id]
        if not patch_neighbor_up.overlap_target_region:
            patch_neighbor_up = None

    patch_neighbor_down_id = patch.get_down_neighbor_position(image, patch_size, gap)
    if patch_neighbor_down_id is None:
        patch_neighbor_down = None
    else:
        patch_neighbor_down = patches[patch_neighbor_down_id]
        if not patch_neighbor_down.overlap_target_region:
            patch_neighbor_down = None

    patch_neighbor_left_id = patch.get_left_neighbor_position(image, patch_size, gap)
    if patch_neighbor_left_id is None:
        patch_neighbor_left = None
    else:
        patch_neighbor_left = patches[patch_neighbor_left_id]
        if not patch_neighbor_left.overlap_target_region:
            patch_neighbor_left = None

    patch_neighbor_right_id = patch.get_right_neighbor_position(image, patch_size, gap)
    if patch_neighbor_right_id is None:
        patch_neighbor_right = None
    else:
        patch_neighbor_right = patches[patch_neighbor_right_id]
        if not patch_neighbor_right.overlap_target_region:
            patch_neighbor_right = None
    return patch_neighbor_up, patch_neighbor_down, patch_neighbor_left, patch_neighbor_right


def update_patchs_neighbors_priority(patch, patch_neighbor, side, image, gap, patch_size, THRESHOLD_UNCERTAINTY):

    if patch_neighbor is not None and not patch_neighbor.committed:

        if not patch_neighbor.overlap_target_region:
            print("Will it ever come to this? (1) (TODO delete if unnecessary)")

        min_additional_difference = sys.maxsize
        additional_differences = {}

        for patch_neighbors_label_id in patch_neighbor.labels:

            patch_neighbors_label_x_coord = patches[patch_neighbors_label_id].x_coord
            patch_neighbors_label_y_coord = patches[patch_neighbors_label_id].y_coord

            patch_neighbors_label_rgb = image.rgb[
                                        patch_neighbors_label_x_coord: patch_neighbors_label_x_coord + patch_size,
                                        patch_neighbors_label_y_coord: patch_neighbors_label_y_coord + patch_size, :]

            patch_neighbors_label_rgb_half = get_half_patch_from_patch(patch_neighbors_label_rgb, gap, opposite_side(side))

            for patchs_label_id in patch.pruned_labels:

                patchs_label_x_coord = patches[patchs_label_id].x_coord
                patchs_label_y_coord = patches[patchs_label_id].y_coord

                patchs_label_rgb = image.rgb[patchs_label_x_coord: patchs_label_x_coord + patch_size,
                                   patchs_label_y_coord: patchs_label_y_coord + patch_size, :]

                patchs_label_rgb_half = get_half_patch_from_patch(patchs_label_rgb, gap, side)

                difference = patch_diff(patch_neighbors_label_rgb_half, patchs_label_rgb_half)

                if difference < min_additional_difference:
                    min_additional_difference = difference

            additional_differences[patch_neighbors_label_id] = min_additional_difference
            min_additional_difference = sys.maxsize

        for key in additional_differences.keys():
            if key in patch_neighbor.differences:
                additional_differences[key] += patch_neighbor.differences[key]
            else:
                print("Will it ever come to this? (2) (TODO delete if unnecessary)")

        # TODO why isn't this calculated in the same way as above?
        temp = [value - min(additional_differences.values()) for value in
                list(additional_differences.values())]
        patch_neighbor_uncertainty = [value < THRESHOLD_UNCERTAINTY for (i, value) in enumerate(temp)].count(True)
        del temp

        patch_neighbor.priority = len(additional_differences) / patch_neighbor_uncertainty #len(patch_neighbor.differences)?


def compute_pairwise_potential_matrix(image, patch_size, gap, MAX_NB_LABELS):

    global patches
    global nodes_count

    # calculate pairwise potential matrix for all pairs of nodes (pathces that have overlap with the target region)
    for patch in patches:
        if patch.overlap_target_region:

            # get the neighbors if they exist and have overlap with the target region (i.e. if they're nodes)
            patch_neighbor_up, _, patch_neighbor_left, _ = get_patch_neighbor_nodes(patch, image, patch_size, gap)

            # TODO make this a method
            if patch_neighbor_up is not None:

                potential_matrix = np.zeros((MAX_NB_LABELS, MAX_NB_LABELS))

                for i, patchs_label_id in enumerate(patch.pruned_labels):

                    patchs_label_x_coord = patches[patchs_label_id].x_coord
                    patchs_label_y_coord = patches[patchs_label_id].y_coord

                    patchs_label_rgb = image.rgb[patchs_label_x_coord: patchs_label_x_coord + patch_size,
                                       patchs_label_y_coord: patchs_label_y_coord + patch_size, :]

                    patchs_label_rgb_up = get_half_patch_from_patch(patchs_label_rgb, gap, UP)

                    for j, patchs_neighbors_label_id in enumerate(patch_neighbor_up.pruned_labels):

                        patchs_neighbors_label_x_coord = patches[patchs_neighbors_label_id].x_coord
                        patchs_neighbors_label_y_coord = patches[patchs_neighbors_label_id].y_coord

                        patchs_neighbors_label_rgb = image.rgb[patchs_neighbors_label_x_coord: patchs_neighbors_label_x_coord + patch_size,
                                                     patchs_neighbors_label_y_coord: patchs_neighbors_label_y_coord + patch_size, :]

                        patchs_neighbors_label_rgb_down = get_half_patch_from_patch(patchs_neighbors_label_rgb, gap, DOWN)

                        potential_matrix[i, j] = patch_diff(patchs_label_rgb_up, patchs_neighbors_label_rgb_down)

                patch.potential_matrix_up = potential_matrix
                patch_neighbor_up.potential_matrix_down = potential_matrix
                # print("potential matrix UpDown")
                # print(potential_matrix)

            if patch_neighbor_left is not None:

                potential_matrix = np.zeros((MAX_NB_LABELS, MAX_NB_LABELS))

                for i, patchs_label_id in enumerate(patch.pruned_labels):

                    patchs_label_x_coord = patches[patchs_label_id].x_coord
                    patchs_label_y_coord = patches[patchs_label_id].y_coord

                    patchs_label_rgb = image.rgb[patchs_label_x_coord: patchs_label_x_coord + patch_size,
                                       patchs_label_y_coord: patchs_label_y_coord + patch_size, :]

                    patchs_label_rgb_left = get_half_patch_from_patch(patchs_label_rgb, gap, LEFT)

                    for j, patchs_neighbors_label_id in enumerate(patch_neighbor_left.pruned_labels):

                        patchs_neighbors_label_x_coord = patches[patchs_neighbors_label_id].x_coord
                        patchs_neighbors_label_y_coord = patches[patchs_neighbors_label_id].y_coord

                        patchs_neighbors_label_rgb = image.rgb[
                                                     patchs_neighbors_label_x_coord: patchs_neighbors_label_x_coord + patch_size,
                                                     patchs_neighbors_label_y_coord: patchs_neighbors_label_y_coord + patch_size,
                                                     :]

                        patchs_neighbors_label_rgb_right = get_half_patch_from_patch(patchs_neighbors_label_rgb, gap,
                                                                                    RIGHT)

                        potential_matrix[i, j] = patch_diff(patchs_label_rgb_left, patchs_neighbors_label_rgb_right)

                patch.potential_matrix_left = potential_matrix
                patch_neighbor_left.potential_matrix_right = potential_matrix
                # print("potential matrix LeftRight")
                # print(potential_matrix)


def compute_label_cost(image, patch_size, MAX_NB_LABELS):

    global patches
    global nodes_count

    for patch in patches:
        if patch.overlap_target_region:

            patch.label_cost = [0 for _ in range(MAX_NB_LABELS)]

            if patch.overlap_source_region:

                # patch_rgb = image.rgb[patch.x_coord : patch.x_coord + patch_size,
                #             patch.y_coord : patch.y_coord + patch_size, :]

                for i, patchs_label_id in enumerate(patch.pruned_labels):

                    # patchs_label_rgb = image.rgb[patches[patchs_label_id].x_coord: patches[patchs_label_id].x_coord + patch_size,
                    #                    patches[patchs_label_id].y_coord: patches[patchs_label_id].y_coord + patch_size, :]

                    # patch.label_cost[i] = patch_diff(patch_rgb, patchs_label_rgb)
                    patch.label_cost[i] = non_masked_patch_diff(image, patch_size, patch.x_coord, patch.y_coord,
                                          patches[patchs_label_id].x_coord, patches[patchs_label_id].y_coord)


            #TODO maybe this constant needs adjusting?
            patch.local_likelihood = [math.exp(-cost * (1/100000)) for cost in patch.label_cost]
            # print("patch", patch.patch_id, "label cost", patch.label_cost)
            # print("patch", patch.patch_id, "local likelihood", patch.local_likelihood)

            #TODO (for the moment it's just the index in [0..MAX_NB_LABELS)
            # but maybe should be the label of the patch with the highest local likelihood
            #patch.mask = patch.pruned_labels[patch.local_likelihood.index(max(patch.local_likelihood))]
            #TODO patch.mask always seems to be zero, why?
            patch.mask = patch.local_likelihood.index(max(patch.local_likelihood))
            print(patch.mask)

#TODO maybe rename local_likelihood to likelihood?
#TODO also calculate InitMask after local_likelihood


#TODO check what happens if there's less than MAX_NB_LABELS even without pruning


# -- 3rd phase --
# inference
# (???)

def neighborhood_consensus_message_passing(image, patch_size, gap, MAX_NB_LABELS, MAX_ITERATION_NR):

    global patches
    global nodes_count

    # initialisation of the nodes' beliefs and messages
    for patch in patches:
        if patch.overlap_target_region:

            #patch.messages = [1 for i in range(MAX_NB_LABELS)]
            patch.messages = np.ones(MAX_NB_LABELS)

            #patch.beliefs = [0 for i in range(MAX_NB_LABELS)]
            patch.beliefs = np.zeros(MAX_NB_LABELS)
            patch.beliefs[patch.mask] = 1


    converged = False
    iteration_nr = 0

    while not converged and iteration_nr < MAX_ITERATION_NR:

        for patch in patches:
            if patch.overlap_target_region:

                patch_neighbor_up, patch_neighbor_down, patch_neighbor_left, patch_neighbor_right = get_patch_neighbor_nodes(
                    patch, image, patch_size, gap)

                #TODO figure out how to do the matrix multiplication cleaner, without the reshaping
                #TODO big problem! they are overriding each other!!
                if patch_neighbor_up is not None:
                    patch.messages = np.matmul(patch.potential_matrix_up, patch_neighbor_up.beliefs.reshape((MAX_NB_LABELS, 1)))
                if patch_neighbor_down is not None:
                    patch.messages = np.matmul(patch.potential_matrix_down, patch_neighbor_down.beliefs.reshape((MAX_NB_LABELS, 1)))
                if patch_neighbor_left is not None:
                    patch.messages = np.matmul(patch.potential_matrix_left, patch_neighbor_left.beliefs.reshape((MAX_NB_LABELS, 1)))
                if patch_neighbor_right is not None:
                    patch.messages = np.matmul(patch.potential_matrix_right, patch_neighbor_right.beliefs.reshape((MAX_NB_LABELS, 1)))


                patch.messages = np.array([math.exp(-message * (1 / 100000)) for message in patch.messages.reshape((MAX_NB_LABELS,1))]).reshape(1, MAX_NB_LABELS)


                # print(patch.messages)
                # print(patch.local_likelihood)
                # TODO maybe should be new beliefs? (nah, i don't think so)
                patch.beliefs = np.multiply(patch.messages, patch.local_likelihood)


                # print(patch.beliefs)
                # normalise to sum up to 1
                #TODO make sure it's element-wise
                patch.beliefs = patch.beliefs / patch.beliefs.sum()


                #TODO fix this disgraceful code
                patch.mask = patch.beliefs.tolist()[0].index(max(patch.beliefs.tolist()[0]))


        iteration_nr += 1


#TODO rename
def generate_inpainted_image(image, patch_size):

    global patches
    global nodes_count
    global nodes_order

    target_region = image.mask

    image.inpainted = np.multiply(image.rgb,
                                  np.repeat(1 - target_region, 3, axis=1).reshape((image.height, image.width, 3)))

    filter_size = 4  # should be > 1

    smooth_filter = generate_smooth_filter(filter_size)


    for i in range(len(nodes_order)):

        patch_id = nodes_order[i]

        patch = patches[patch_id]

        print(len(patches), len(patch.pruned_labels))
        print(patch.mask)
        print(patch.pruned_labels[patch.mask])
        patchs_mask_patch = patches[patch.pruned_labels[patch.mask]] #TODO IndexError: list index out of range

        patch_rgb = image.inpainted[patch.x_coord: patch.x_coord + patch_size,
                    patch.y_coord: patch.y_coord + patch_size, :]

        patch_rgb_new = image.inpainted[patchs_mask_patch.x_coord: patchs_mask_patch.x_coord + patch_size,
                        patchs_mask_patch.y_coord: patchs_mask_patch.y_coord + patch_size, :]

        # sqaured error is used to calculate the blend mask
        #squared_error_3channels = (patch_rgb - patch_rgb_new)**2
        #squared_error_1channel = np.dot(squared_error_3channels[..., :3], [1/3, 1/3, 1/3])

        blend_mask = generate_blend_mask(patch_size)

        blend_mask = signal.convolve2d(blend_mask, smooth_filter, boundary='symm', mode='same')

        blend_mask_rgb = np.repeat(blend_mask, 3, axis=1).reshape((patch_size, patch_size, 3))

        print("inpainted")
        plt.imshow(image.inpainted[patch.x_coord : patch.x_coord + patch_size,
            patch.y_coord : patch.y_coord + patch_size, :].astype(np.uint8), interpolation='nearest')
        # plt.show()
        plt.imshow((np.multiply(patch_rgb, blend_mask_rgb) + np.multiply(patch_rgb_new, 1 - blend_mask_rgb)).astype(np.uint8), interpolation='nearest')
        # plt.show()

        print(image.inpainted[patch.x_coord : patch.x_coord + patch_size,
            patch.y_coord : patch.y_coord + patch_size, 0])
        print((np.multiply(patch_rgb, blend_mask_rgb) + np.multiply(patch_rgb_new, 1 - blend_mask_rgb))[:,:,0])



        image.inpainted[patch.x_coord : patch.x_coord + patch_size,
            patch.y_coord : patch.y_coord + patch_size, :] = np.multiply(patch_rgb, blend_mask_rgb) + \
                                                             np.multiply(patch_rgb_new, 1 - blend_mask_rgb)



        target_region[patch.x_coord : patch.x_coord + patch_size, patch.y_coord : patch.y_coord + patch_size] = 0


    image.inpainted = image.inpainted.astype(np.uint8)



    # In the Tijana's code, this line makes a difference because the mask is not looked as a binary thing,
    # but as a scale (for some reason). Here, it's binary, and hence this line doesn't do anything.
    # image.inpainted = np.multiply(image.inpainted, np.repeat(1 - target_region, 3, axis=1).reshape((image.height, image.width, 3)))




def generate_smooth_filter(kernel_size):

    if kernel_size <= 1:
        raise Exception('Kernel size for the smooth filter should be larger than 1, but is {}.'.format(kernel_size))

    kernel_1D = np.array([0.5, 0.5]).transpose()

    for i in range(kernel_size - 2):
        kernel_1D = np.convolve(np.array([0.5, 0.5]).transpose(), kernel_1D)

    kernel_1D = kernel_1D.reshape((kernel_size, 1))

    kernel_2D = np.matmul(kernel_1D, kernel_1D.transpose())

    return kernel_2D


#TODO this is just a hacky way to implement something similar to what is needed
def generate_blend_mask(patch_size):

    blend_mask = np.zeros((patch_size, patch_size))
    for i in range(patch_size // 3):
        blend_mask[i, :] = 1
        blend_mask[:, i] = 1

    return blend_mask