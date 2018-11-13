import numpy as np
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
    for y in range(0, image.width - patch_size, gap):
        for x in range(0, image.height - patch_size, gap):

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

                patch_rgb = image.rgb[x: x + patch_size, y: y + patch_size, :]


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

            if nodes_count == 7:
                break


    print(len(patches))
    print(patches[774])


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


        patch = patches[patch_highest_priority_id]
        patch.committed = True
        nodes_visiting_order[i] = patch_highest_priority_id

        # prune the labels of this node
        patch.prune_labels(MAX_NB_LABELS)




















