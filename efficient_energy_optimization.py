import numpy as np
import sys
from patch_diff import patch_diff
from collections import namedtuple
from data_structures import Image2BInpainted

# efficient energy optimization


# -- 1st phase --
# initialization
# (assigning priorities to MRF nodes to be used for determining the visiting order in the 2nd phase)

# TODO don't need patches_count?
# TODO deal with the memory overflow, maybe use smaller types?patches_count


def initialization(image, mask, patch_size, gap, THRESHOLD_UNCERTAINTY):

    max_nb_nodes = 3844  # (image_height - patch_size + 1) * (image_width - patch_size + 1) TODO
    nodes_coords = np.zeros((max_nb_nodes, 2), dtype=np.int32)
    nodes_priority = np.zeros(max_nb_nodes)

    nodes_count = 0
    patches_count = 0
    nodes_labels = []
    nodes_differences = []

    # for all the patches in an image (not all, but with $gap stride)
    for y in range(0, image.width - patch_size, gap):
        for x in range(0, image.height - patch_size, gap):

            patch_mask_overlap = mask[x: x + patch_size, y: y + patch_size]
            patch_mask_overlap_nonzero_elements = np.count_nonzero(patch_mask_overlap)

            # if the patch overlaps with the target region
            if patch_mask_overlap_nonzero_elements > 0:

                print('---')
                print(nodes_count)
                print()

                patch = image[x: x + patch_size, y: y + patch_size, :]

                nodes_coords[nodes_count, 0] = x
                nodes_coords[nodes_count, 1] = y

                nodes_labels.append([])  # nodes_labels[nodes_count].append...

                nodes_differences.append(np.zeros(max_nb_nodes))

                # if the patch also overlaps with the source region
                if patch_mask_overlap_nonzero_elements < patch_size * patch_size:

                    patches_count_compare = 0
                    for y_compare in range(0, image.width - patch_size, gap):
                        for x_compare in range(0, image.height - patch_size, gap):
                            # TODO take into account the mask for comparing...
                            # TODO ... use codes instead of the pixel values

                            nodes_differences[nodes_count][patches_count_compare] = patch_diff(
                                patch, image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :])

                            nodes_labels[nodes_count].append(patches_count_compare)

                            patches_count_compare += 1

                    temp = nodes_differences[nodes_count] - min(nodes_differences[nodes_count])
                    patch_uncertainty = len([val for (i, val) in enumerate(temp) if val < THRESHOLD_UNCERTAINTY])
                    del temp

                # if the patch is completely within the target region
                else:

                    patches_count_compare = 0
                    for y_compare in range(0, image.width - patch_size, gap):
                        for x_compare in range(0, image.height - patch_size, gap):
                            nodes_labels[nodes_count].append(patches_count_compare)

                            patches_count_compare += 1

                    # the labels of blocks assigned in find_label_pos...,
                    # and i think they include only patches that are completely inside the target region, TODO

                # the higher priority the higher priority :D
                nodes_priority[nodes_count] = len(nodes_labels[nodes_count]) / max(patch_uncertainty, 1)

                nodes_count += 1

            patches_count += 1

            # TODO discard this
            if nodes_count == 7:
                break

    nodes_priority = nodes_priority[0:nodes_count]


    nodes = namedtuple("nodes", ["nodes_count", "nodes_priority", "nodes_differences", "nodes_labels", "nodes_coords"])
    return nodes(nodes_count, nodes_priority, nodes_differences, nodes_labels, nodes_coords)


def label_pruning(nodes_count, nodes_priority, nodes_differences, nodes_labels, nodes_coords, image, patch_size, gap, THRESHOLD_UNCERTAINTY, MAX_NB_LABELS):
    nodes_commited = np.zeros(nodes_count, dtype=bool)
    nodes_labels_new = np.zeros((nodes_count, MAX_NB_LABELS), dtype=np.int32)
    nodes_visiting_order = np.zeros(nodes_count, dtype=np.int32)

    # for all the nodes (patches) that are in or intersect with the target region
    for node in range(nodes_count):
        print('-----')
        print(node)
        print()

        # find the node with the highest priority that hasn't yet been visited
        nodes_priority_uncomitted = np.array([val for (i, val) in enumerate(nodes_priority) if not nodes_commited[i]])
        node_position = np.where(nodes_priority == max(nodes_priority_uncomitted))[0][
            0]  # TODO maybe a better way of doing this?

        nodes_commited[node_position] = True

        nodes_visiting_order[node] = node_position

        # TODO faster
        temp = nodes_differences[node_position].argsort()[1: MAX_NB_LABELS + 1]
        nodes_labels_new[node_position] = np.array(nodes_labels[node_position])[temp]
        del temp

        node_neighbour_up = 0
        node_neighbour_down = 0
        node_neighbour_left = 0
        node_neighbour_right = 0

        # for all 4 node's neighbours
        # TODO instead of this for loop, somehow lookup the position (label) of the nodes neighbours
        # TODO everywhere where it's variable node it should actually be node_position
        for i in range(nodes_count):
            if (nodes_coords[i, 0] == nodes_coords[node, 0] - gap) and (nodes_coords[i, 1] == nodes_coords[node, 1]):
                node_neighbour_up = i
            if (nodes_coords[i, 0] == nodes_coords[node, 0] + gap) and (nodes_coords[i, 1] == nodes_coords[node, 1]):
                node_neighbour_down = i
            if (nodes_coords[i, 0] == nodes_coords[node, 0]) and (nodes_coords[i, 1] == nodes_coords[node, 1] - gap):
                node_neighbour_left = i
            if (nodes_coords[i, 0] == nodes_coords[node, 0]) and (nodes_coords[i, 1] == nodes_coords[node, 1] + gap):
                node_neighbour_right = i

            # if upper neighbour and has lower priority than the current node
            if node_neighbour_up and not nodes_commited[node_neighbour_up]:

                neighbour_nb_labels = len(nodes_labels[node_neighbour_up])

                min_diff = sys.maxsize
                differences = np.zeros(neighbour_nb_labels)

                # for all the labels of the upper neighbour
                for j in range(neighbour_nb_labels):

                    neighbours_label = nodes_labels[node_neighbour_up][j]
                    neighbours_label_x_coord = nodes_coords[neighbours_label, 0]  # this isn't right?
                    neighbours_label_y_coord = nodes_coords[neighbours_label, 1]

                    neighbours_label_patch = image[
                                             neighbours_label_x_coord + gap: neighbours_label_x_coord + patch_size,
                                             neighbours_label_y_coord: neighbours_label_y_coord + patch_size, :]

                    for k in range(MAX_NB_LABELS):
                        nodes_label = nodes_labels_new[node, k]
                        nodes_label_x_coord = nodes_coords[nodes_label, 0]
                        nodes_label_y_coord = nodes_coords[nodes_label, 1]

                        nodes_label_patch = image[nodes_label_x_coord: nodes_label_x_coord + patch_size - gap,
                                            nodes_label_y_coord: nodes_label_y_coord + patch_size, :]

                        diff = patch_diff(neighbours_label_patch, nodes_label_patch)

                        if (diff < min_diff):
                            min_diff = diff

                    differences[j] = min_diff
                    min_diff = sys.maxsize

                nodes_differences[node_neighbour_up] += differences
                print(nodes_differences[node_neighbour_up].shape)
                print(differences.shape)

                temp = nodes_differences[node_neighbour_up] - min(nodes_differences[node_neighbour_up])
                uncertainty = [val < THRESHOLD_UNCERTAINTY for (i, val) in enumerate(temp)].count(True)
                del temp
                nodes_priority[node_neighbour_up] = len(nodes_differences[node_neighbour_up]) / uncertainty