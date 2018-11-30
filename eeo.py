import numpy as np
import sys
from data_structures import Patch
from patch_diff import patch_diff


# the indices in the list match the patch_id
patches = []
nodes_count = 0


# -- 1st phase --
# initialization
# (assigning priorities to MRF nodes to be used for determining the visiting order in the 2nd phase)
def initialization(image, patch_size, gap, THRESHOLD_UNCERTAINTY):

    global patches
    global nodes_count

    patch_id_counter = 0

    #TODO taking the patches that are not fully in the image, or not taking some that are? deal with this
    #TODO it should be image.width - patch_size + 1 (I think)
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

            if patch.overlap_target_region:

                patch_rgb = image.rgb[x : x + patch_size, y : y + patch_size, :]


                if patch.overlap_source_region:

                    # compare the patch to all the other patches (that are not completely in the target?)
                    patch_compare_id_counter = 0
                    for y_compare in range(0, image.width - patch_size, gap):
                        for x_compare in range(0, image.height - patch_size, gap):
                            # TODO don't do it twice, but once

                            # TODO take into account the mask for comparing
                            # TODO use codes instead of the pixel values

                            patch_difference = patch_diff(patch_rgb,
                                                          image.rgb[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :])
                            patch.differences[patch_compare_id_counter] = patch_difference

                            patch.labels.append(patch_compare_id_counter)

                            patch_compare_id_counter += 1

                    #temp = list(patch.differences.values()) - min(list(patch.differences.values()))
                    temp = [value - min(list(patch.differences.values())) for value in list(patch.differences.values())]
                    patch_uncertainty = len([val for (i, val) in enumerate(temp) if val < THRESHOLD_UNCERTAINTY])
                    del temp

                else:
                    patch_compare_id_counter = 0
                    for y_compare in range(0, image.width - patch_size, gap):
                        for x_compare in range(0, image.height - patch_size, gap):

                            patch.labels.append(patch_compare_id_counter)

                            patch_compare_id_counter += 1

                    patch_uncertainty = 1

                    # TODO something mentioned in the other file (find_label_pos)

                # the higher priority the higher priority :D
                patch.priority = len(patch.labels) / max(patch_uncertainty, 1)

                nodes_count +=1

            patches.append(patch)
            patch_id_counter += 1

            # if nodes_count == 7:
            #     break


    print("len(patches) = ", len(patches))

# -- 2nd phase --
# label pruning
# (reducing the number of labels at each node to a relatively small number)

def label_pruning(image, patch_size, gap, THRESHOLD_UNCERTAINTY, MAX_NB_LABELS):

    global patches
    global nodes_count

    nodes_visiting_order = np.zeros(nodes_count, dtype=np.int32)

    # for all the patches that have an overlap with the target region (aka nodes)
    for i in range(nodes_count):

        #TODO this can maybe be done faster, by sorting the nodes by priority once,
        #TODO unless the priorities are changing, which might actually be the case :D
        # find the node with the highest priority that hasn't yet been visited
        highest_priority = -1
        patch_highest_priority_id = -1
        for patch in patches:
            if not patch.committed and patch.priority > highest_priority:
                highest_priority = patch.priority
                patch_highest_priority_id = patch.patch_id


        nodes_visiting_order[i] = patch_highest_priority_id
        patch = patches[patch_highest_priority_id]
        patch.committed = True

        patch.prune_labels(MAX_NB_LABELS)

        print(patch_highest_priority_id)

        # get the neighbors if they exist and have overlap with the target region
        patch_neighbor_up, patch_neighbor_down, patch_neighbor_left, patch_neighbor_right = get_patch_neighbor_nodes(
            patch, image, patch_size, gap)

        update_patchs_neighbors_differences_and_priority(patch, patch_neighbor_up, image, patch_size, THRESHOLD_UNCERTAINTY)
        update_patchs_neighbors_differences_and_priority(patch, patch_neighbor_down, image, patch_size, THRESHOLD_UNCERTAINTY)
        update_patchs_neighbors_differences_and_priority(patch, patch_neighbor_left, image, patch_size, THRESHOLD_UNCERTAINTY)
        update_patchs_neighbors_differences_and_priority(patch, patch_neighbor_right, image, patch_size, THRESHOLD_UNCERTAINTY)


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


def update_patchs_neighbors_differences_and_priority(patch, patch_neighbor, image, patch_size, THRESHOLD_UNCERTAINTY):

    if not patch_neighbor is None and patch_neighbor.overlap_target_region and not patch_neighbor.committed:

        min_additional_difference = sys.maxsize
        additional_differences = {}

        for patch_neighbors_label_id in patch_neighbor.labels:

            patch_neighbors_label_x_coord = patches[patch_neighbors_label_id].x_coord
            patch_neighbors_label_y_coord = patches[patch_neighbors_label_id].y_coord

            patch_neighbors_label_rgb = image.rgb[
                                        patch_neighbors_label_x_coord: patch_neighbors_label_x_coord + patch_size,
                                        patch_neighbors_label_y_coord: patch_neighbors_label_y_coord + patch_size, :]

            for patchs_label_id in patch.pruned_labels:

                patchs_label_x_coord = patches[patchs_label_id].x_coord
                patchs_label_y_coord = patches[patchs_label_id].y_coord

                patchs_label_rgb = image.rgb[patchs_label_x_coord: patchs_label_x_coord + patch_size,
                                   patchs_label_y_coord: patchs_label_y_coord + patch_size, :]

                difference = patch_diff(patch_neighbors_label_rgb, patchs_label_rgb)

                if (difference < min_additional_difference):
                    min_additional_difference = difference

            additional_differences[patch_neighbors_label_id] = min_additional_difference
            min_additional_difference = sys.maxsize

        for key in additional_differences.keys():
            if key in patch_neighbor.differences:
                patch_neighbor.differences[key] += additional_differences[key]
            else:
                patch_neighbor.differences[key] = additional_differences[key]

        temp = [value - min(patch_neighbor.differences.values()) for value in
                list(patch_neighbor.differences.values())]
        patch_neighbor_uncertainty = [value < THRESHOLD_UNCERTAINTY for (i, value) in enumerate(temp)].count(True)
        del temp

        patch_neighbor.priority = len(patch_neighbor.differences) / patch_neighbor_uncertainty


def compute_pairwise_potential_matrix(image, patch_size, gap, MAX_NB_LABELS):

    global patches
    global nodes_count

    for patch in patches:
        if patch.overlap_target_region:

            # get the neighbors if they exist and have overlap with the target region
            patch_neighbor_up, _, patch_neighbor_left, _ = get_patch_neighbor_nodes(patch, image, patch_size, gap)




            if not patch_neighbor_up is None:

                potential_matrix = np.zeros((MAX_NB_LABELS, MAX_NB_LABELS))

                for i, patchs_label_id in enumerate(patch.pruned_labels):

                    patchs_label_x_coord = patches[patchs_label_id].x_coord
                    patchs_label_y_coord = patches[patchs_label_id].y_coord

                    patchs_label_upper_rgb = image.rgb[patchs_label_x_coord : patchs_label_x_coord + patch_size - gap,
                                       patchs_label_y_coord : patchs_label_y_coord + patch_size, :]


                    for j, patchs_neighbors_label_id in enumerate(patch_neighbor_up.pruned_labels):

                        patchs_neighbors_label_x_coord = patches[patchs_neighbors_label_id].x_coord
                        patchs_neighbors_label_y_coord = patches[patchs_neighbors_label_id].y_coord

                        patchs_neighbors_label_lower_rgb = image.rgb[patchs_neighbors_label_x_coord + gap : patchs_neighbors_label_x_coord + patch_size,
                                                     patchs_neighbors_label_y_coord : patchs_neighbors_label_y_coord + patch_size, :]

                        potential_matrix[i, j] = patch_diff(patchs_label_upper_rgb, patchs_neighbors_label_lower_rgb)


                patch.potential_matrix_up = potential_matrix
                patch_neighbor_up.potential_matrix_down = potential_matrix



            if not patch_neighbor_left is None:

                potential_matrix = np.zeros((MAX_NB_LABELS, MAX_NB_LABELS))

                for i, patchs_label_id in enumerate(patch.pruned_labels):

                    patchs_label_x_coord = patches[patchs_label_id].x_coord
                    patchs_label_y_coord = patches[patchs_label_id].y_coord

                    patchs_label_left_rgb = image.rgb[patchs_label_x_coord : patchs_label_x_coord + patch_size,
                                       patchs_label_y_coord : patchs_label_y_coord + patch_size - gap, :]


                    for j, patchs_neighbors_label_id in enumerate(patch_neighbor_left.pruned_labels):

                        patchs_neighbors_label_x_coord = patches[patchs_neighbors_label_id].x_coord
                        patchs_neighbors_label_y_coord = patches[patchs_neighbors_label_id].y_coord

                        patchs_neighbors_label_right_rgb = image.rgb[patchs_neighbors_label_x_coord : patchs_neighbors_label_x_coord + patch_size,
                                                     patchs_neighbors_label_y_coord + gap : patchs_neighbors_label_y_coord + patch_size, :]

                        potential_matrix[i, j] = patch_diff(patchs_label_left_rgb, patchs_neighbors_label_right_rgb)


                patch.potential_matrix_left = potential_matrix
                patch_neighbor_left.potential_matrix_right = potential_matrix


def compute_label_cost(image, patch_size, MAX_NB_LABELS):

    global patches
    global nodes_count


    for patch in patches:
        if patch.overlap_target_region:

            patch.label_cost = [0 for i in range(MAX_NB_LABELS)]

            if patch.overlap_source_region:

                patch_rgb = image.rgb[patch.x_coord : patch.x_coord + patch_size,
                            patch.y_coord : patch.y_coord + patch_size, :]

                for i, patchs_label_id in enumerate(patch.pruned_labels):

                    patchs_label_rgb = image.rgb[patches[patchs_label_id].x_coord : patches[patchs_label_id].x_coord + patch_size,
                                       patches[patchs_label_id].y_coord : patches[patchs_label_id].y_coord + patch_size, :]

                    patch.label_cost[i] = patch_diff(patch_rgb, patchs_label_rgb)




#TODO check what happens if there's less than MAX_NB_LABELS even without pruning


# -- 3rd phase --
# inference
# (???)