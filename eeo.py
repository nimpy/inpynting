import numpy as np
import sys
import pickle
import math
from scipy import signal
import matplotlib.pyplot as plt
import imageio

from data_structures import Node, coordinates_to_position, position_to_coordinates
from data_structures import UP, DOWN, LEFT, RIGHT
from data_structures import get_half_patch_from_patch, opposite_side
from patch_diff import non_masked_patch_diff, half_patch_diff
from patch_diff import max_pool


nodes = {}  # the indices in this list patches match the node_id
nodes_count = 0
nodes_order = []


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




def initialization_rgb(image, thresh_uncertainty):

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

    labels_diametar = 100

    for i, node in enumerate(nodes.values()):

        sys.stdout.write("\rInitialising node " + str(i + 1) + "/" + str(nodes_count))

        if node.overlap_source_region:

            node_rgb = image.rgb[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
            mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
            mask_3ch = np.repeat(mask, 3, axis=1).reshape((image.patch_size, image.patch_size, 3))
            node_rgb = node_rgb * (1 - mask_3ch)

            # compare the node patch to all patches that are completely in the source region
            for y_compare in range(max(node.y_coord - labels_diametar, 0),
                                   min(node.y_coord + labels_diametar, image.width - image.patch_size + 1)):
                for x_compare in range(max(node.x_coord - labels_diametar, 0),
                                       min(node.x_coord + labels_diametar, image.height - image.patch_size + 1)):

            # for y_compare in range(0, image.width - image.patch_size + 1):
            #     for x_compare in range(0, image.height - image.patch_size + 1):

                    patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size]
                    patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                    if patch_compare_mask_overlap_nonzero_elements == 0:
                        patch_compare_rgb = image.rgb[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size, :]
                        patch_compare_rgb = patch_compare_rgb * (1 - mask_3ch)

                        # patch_difference = non_masked_patch_diff(image, node.x_coord, node.y_coord, x_compare, y_compare)
                        patch_difference = np.sum(np.subtract(node_rgb, patch_compare_rgb, dtype=np.int32) ** 2)

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
            for y_compare in range(max(node.y_coord - labels_diametar, 0),
                                   min(node.y_coord + labels_diametar, image.width - image.patch_size + 1)):
                for x_compare in range(max(node.x_coord - labels_diametar, 0),
                                       min(node.x_coord + labels_diametar, image.height - image.patch_size + 1)):
            # for y_compare in range(0, image.width - image.patch_size + 1):
            #     for x_compare in range(0, image.height - image.patch_size + 1):

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


def initialization_ir(image, thresh_uncertainty):

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

    labels_diametar = 100
    nr_channels = image.ir.shape[2]

    for i, node in enumerate(nodes.values()):

        sys.stdout.write("\rInitialising node " + str(i + 1) + "/" + str(nodes_count))

        if node.overlap_source_region:

            node_ir = image.ir[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]
            mask = image.mask[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size]
            mask_more_ch = np.repeat(mask, nr_channels, axis=1).reshape((image.patch_size, image.patch_size, nr_channels))
            node_ir = node_ir * (1 - mask_more_ch)
            node_descr = max_pool(node_ir)

            # compare the node patch to all patches that are completely in the source region
            for y_compare in range(max(node.y_coord - labels_diametar, 0),
                                   min(node.y_coord + labels_diametar, image.width - image.patch_size + 1)):
                for x_compare in range(max(node.x_coord - labels_diametar, 0),
                                       min(node.x_coord + labels_diametar, image.height - image.patch_size + 1)):

            # for y_compare in range(0, image.width - image.patch_size + 1):
            #     for x_compare in range(0, image.height - image.patch_size + 1):

                    patch_compare_mask_overlap = image.mask[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size]
                    patch_compare_mask_overlap_nonzero_elements = np.count_nonzero(patch_compare_mask_overlap)

                    if patch_compare_mask_overlap_nonzero_elements == 0:
                        patch_compare_ir = image.ir[x_compare: x_compare + image.patch_size, y_compare: y_compare + image.patch_size, :]
                        patch_compare_ir = patch_compare_ir * (1 - mask_more_ch)
                        patch_compare_descr = max_pool(patch_compare_ir)

                        # patch_difference = non_masked_patch_diff(image, node.x_coord, node.y_coord, x_compare, y_compare)
                        patch_difference = np.sum(np.subtract(node_descr, patch_compare_descr, dtype=np.int32) ** 2)

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
            for y_compare in range(max(node.y_coord - labels_diametar, 0),
                                   min(node.y_coord + labels_diametar, image.width - image.patch_size + 1)):
                for x_compare in range(max(node.x_coord - labels_diametar, 0),
                                       min(node.x_coord + labels_diametar, image.height - image.patch_size + 1)):
            # for y_compare in range(0, image.width - image.patch_size + 1):
            #     for x_compare in range(0, image.height - image.patch_size + 1):

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




# -- 2nd phase --
# label pruning
# (reducing the number of labels at each node to a relatively small number)

def label_pruning_rgb(image, thresh_uncertainty, max_nr_labels):
    global nodes
    global nodes_count
    global nodes_order

    # make a copy of the differences which we can edit and use in this method, and afterwards discard
    for node in nodes.values():
        node.additional_differences = node.differences.copy()

    # for all the patches that have an overlap with the target region (aka nodes)
    for i in range(nodes_count):

        # find the node with the highest priority that hasn't yet been visited
        highest_priority = -1
        node_highest_priority_id = -1
        for node in nodes.values():
            if not node.committed and node.priority > highest_priority:
                highest_priority = node.priority
                node_highest_priority_id = node.node_id

        node = nodes[node_highest_priority_id]
        node.committed = True

        node.prune_labels(max_nr_labels)

        print('Highest priority node {0:3d}/{1:3d}: {2:d}'.format(i + 1, nodes_count, node_highest_priority_id))
        nodes_order.append(node_highest_priority_id)

        # get the neighbors if they exist and have overlap with the target region
        node_neighbor_up, node_neighbor_down, node_neighbor_left, node_neighbor_right = get_neighbor_nodes(
            node, image)

        update_neighbors_priority_rgb(node, node_neighbor_up, UP, image, thresh_uncertainty)
        update_neighbors_priority_rgb(node, node_neighbor_down, DOWN, image, thresh_uncertainty)
        update_neighbors_priority_rgb(node, node_neighbor_left, LEFT, image, thresh_uncertainty)
        update_neighbors_priority_rgb(node, node_neighbor_right, RIGHT, image, thresh_uncertainty)


def label_pruning_ir(image, thresh_uncertainty, max_nr_labels):
    global nodes
    global nodes_count
    global nodes_order

    # make a copy of the differences which we can edit and use in this method, and afterwards discard
    for node in nodes.values():
        node.additional_differences = node.differences.copy()

    # for all the patches that have an overlap with the target region (aka nodes)
    for i in range(nodes_count):

        # find the node with the highest priority that hasn't yet been visited
        highest_priority = -1
        node_highest_priority_id = -1
        for node in nodes.values():
            if not node.committed and node.priority > highest_priority:
                highest_priority = node.priority
                node_highest_priority_id = node.node_id

        node = nodes[node_highest_priority_id]
        node.committed = True

        node.prune_labels(max_nr_labels)

        print('Highest priority node {0:3d}/{1:3d}: {2:d}'.format(i + 1, nodes_count, node_highest_priority_id))
        nodes_order.append(node_highest_priority_id)

        # get the neighbors if they exist and have overlap with the target region
        node_neighbor_up, node_neighbor_down, node_neighbor_left, node_neighbor_right = get_neighbor_nodes(
            node, image)

        update_neighbors_priority_ir(node, node_neighbor_up, UP, image, thresh_uncertainty)
        update_neighbors_priority_ir(node, node_neighbor_down, DOWN, image, thresh_uncertainty)
        update_neighbors_priority_ir(node, node_neighbor_left, LEFT, image, thresh_uncertainty)
        update_neighbors_priority_ir(node, node_neighbor_right, RIGHT, image, thresh_uncertainty)


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

                difference = np.sum(np.subtract(patch_neighbors_label_rgb_half, patchs_label_rgb_half, dtype=np.int32) ** 2)

                # difference = half_patch_diff(image, node_label_x_coord, node_label_y_coord, neighbors_label_x_coord, neighbors_label_y_coord, side)

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


def update_neighbors_priority_ir(node, neighbor, side, image, thresh_uncertainty):

    # if neighbor is a node that hasn't been committed yet
    if neighbor is not None and not neighbor.committed:

        min_additional_difference = sys.maxsize
        additional_differences = {}

        for neighbors_label_id in neighbor.labels:

            neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

            patch_neighbors_label_ir = image.ir[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                                        neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]
            patch_neighbors_label_ir_half = get_half_patch_from_patch(patch_neighbors_label_ir, image.stride, opposite_side(side))

            for node_label_id in node.pruned_labels:

                node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                patchs_label_ir = image.ir[node_label_x_coord: node_label_x_coord + image.patch_size,
                                   node_label_y_coord: node_label_y_coord + image.patch_size, :]
                patchs_label_ir_half = get_half_patch_from_patch(patchs_label_ir, image.stride, side)

                difference = np.sum(np.subtract(patch_neighbors_label_ir_half, patchs_label_ir_half, dtype=np.int32) ** 2)

                # difference = half_patch_diff(image, node_label_x_coord, node_label_y_coord, neighbors_label_x_coord, neighbors_label_y_coord, side)

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

    # calculate pairwise potential matrix for all pairs of nodes (pathces that have overlap with the target region)
    for node in nodes.values():

        # get the neighbors if they exist and have overlap with the target region (i.e. if they're nodes)
        neighbor_up, _, neighbor_left, _ = get_neighbor_nodes(node, image)

        # TODO make this a method
        if neighbor_up is not None:

            potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

            for i, node_label_id in enumerate(node.pruned_labels):

                node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                # patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size,
                #                    node_label_y_coord: node_label_y_coord + image.patch_size, :]

                # patchs_label_rgb_up = get_half_patch_from_patch(patchs_label_rgb, image.stride, UP)

                for j, neighbors_label_id in enumerate(neighbor_up.pruned_labels):

                    neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

                    # patchs_neighbors_label_rgb = image.rgb[neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                    #                              neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size, :]

                    # patchs_neighbors_label_rgb_down = get_half_patch_from_patch(patchs_neighbors_label_rgb, image.stride, DOWN)

                    # potential_matrix[i, j] = patch_diff(patchs_label_rgb_up, patchs_neighbors_label_rgb_down)
                    potential_matrix[i, j] = half_patch_diff(image, node_label_x_coord, node_label_y_coord, neighbors_label_x_coord, neighbors_label_y_coord, UP)

            node.potential_matrix_up = potential_matrix
            neighbor_up.potential_matrix_down = potential_matrix
            # print("potential matrix UpDown")
            # print(potential_matrix)

        if neighbor_left is not None:

            potential_matrix = np.zeros((max_nr_labels, max_nr_labels))

            for i, node_label_id in enumerate(node.pruned_labels):

                node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)

                # patchs_label_rgb = image.rgb[node_label_x_coord: node_label_x_coord + image.patch_size,
                #                    node_label_y_coord: node_label_y_coord + image.patch_size, :]

                # patchs_label_rgb_left = get_half_patch_from_patch(patchs_label_rgb, image.stride, LEFT)

                for j, neighbors_label_id in enumerate(neighbor_left.pruned_labels):

                    neighbors_label_x_coord, neighbors_label_y_coord = position_to_coordinates(neighbors_label_id, image.height, image.patch_size)

                    # patchs_neighbors_label_rgb = image.rgb[
                    #                              neighbors_label_x_coord: neighbors_label_x_coord + image.patch_size,
                    #                              neighbors_label_y_coord: neighbors_label_y_coord + image.patch_size,
                    #                              :]

                    # patchs_neighbors_label_rgb_right = get_half_patch_from_patch(patchs_neighbors_label_rgb, image.stride,
                    #                                                             RIGHT)

                    # potential_matrix[i, j] = patch_diff(patchs_label_rgb_left, patchs_neighbors_label_rgb_right)
                    potential_matrix[i, j] = half_patch_diff(image, node_label_x_coord, node_label_y_coord,
                                                             neighbors_label_x_coord,
                                                             neighbors_label_y_coord, LEFT)

            node.potential_matrix_left = potential_matrix
            neighbor_left.potential_matrix_right = potential_matrix
            # print("potential matrix LeftRight")
            # print(potential_matrix)


def compute_label_cost(image, max_nr_labels):

    global nodes
    global nodes_count

    for node in nodes.values():

        node.label_cost = [0 for _ in range(max_nr_labels)]

        if node.overlap_source_region:

            # patch_rgb = image.rgb[node.x_coord : node.x_coord + patch_size,
            #             node.y_coord : node.y_coord + patch_size, :]

            for i, node_label_id in enumerate(node.pruned_labels):

                # patchs_label_rgb = image.rgb[patches[node_label_id].x_coord: patches[node_label_id].x_coord + patch_size,
                #                    patches[node_label_id].y_coord: patches[node_label_id].y_coord + patch_size, :]

                # node.label_cost[i] = patch_diff(patch_rgb, patchs_label_rgb)

                node_label_x_coord, node_label_y_coord = position_to_coordinates(node_label_id, image.height, image.patch_size)
                node.label_cost[i] = non_masked_patch_diff(image, node.x_coord, node.y_coord, node_label_x_coord, node_label_y_coord)


        node.local_likelihood = [math.exp(-cost * (1/100000)) for cost in node.label_cost]
        # print("node", node.node_id, "label cost", node.label_cost)
        # print("node", node.node_id, "local likelihood", node.local_likelihood)

        node.mask = node.local_likelihood.index(max(node.local_likelihood))


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


def generate_inpainted_image(image):

    global nodes
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

        node_id = nodes_order[i]
        node = nodes[node_id]

        node_mask_patch_x_coord, node_mask_patch_y_coord =  position_to_coordinates(node.pruned_labels[node.mask], image.height, image.patch_size)

        node_rgb = image.inpainted[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :]

        node_rgb_new = image.inpainted[node_mask_patch_x_coord: node_mask_patch_x_coord + image.patch_size, node_mask_patch_y_coord: node_mask_patch_y_coord + image.patch_size, :]

        image.inpainted[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size, :] =\
            np.multiply(node_rgb, blend_mask_rgb) + np.multiply(node_rgb_new, 1 - blend_mask_rgb)

        target_region[node.x_coord: node.x_coord + image.patch_size, node.y_coord: node.y_coord + image.patch_size] = 0

        # plt.imshow(image.inpainted.astype(np.uint8), interpolation='nearest')
        # plt.show()
        # imageio.imwrite('/home/niaki/Code/inpynting_images/building/ordering2/building_' + str(i).zfill(4) + '.jpg', image.inpainted)

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
#             # imageio.imwrite('/home/niaki/Code/inpynting_images/Tijana/Jian10_uint8/ordering_process1/Jian10_' + str(i).zfill(4) + '.jpg', image.inpainted)
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
#         # imageio.imwrite('/home/niaki/Code/inpynting_images/Tijana/Jian10_uint8/ordering_process1/Jian10_' + str(i).zfill(4) + '.jpg', image.inpainted)
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
    for i in range(patch_size // 3):
        blend_mask[i, :] = 1
        blend_mask[:, i] = 1

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
