import numpy as np
from data_structures import Patch
from patch_diff import patch_diff



patches = []



# -- 1st phase --
# initialization
# (assigning priorities to MRF nodes to be used for determining the visiting order in the 2nd phase)





def initialization(image, patch_size, gap, THRESHOLD_UNCERTAINTY):

    patch_id_counter = 0
    temp_nodes_count = 0

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
                            # TODO take into account the mask for comparing...
                            # TODO ... use codes instead of the pixel values

                            patch_difference = patch_diff(patch_rgb,
                                                          image.rgb[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size, :])
                            patch.differences[patch_compare_id_counter] = patch_difference

                            patch.labels.append(patch_compare_id_counter)

                            patch_compare_id_counter += 1

                    temp = list(patch.differences.values()) - min(list(patch.differences.values()))
                    patch_uncertainty = len([val for (i, val) in enumerate(temp) if val < THRESHOLD_UNCERTAINTY])
                    del temp

                else:
                    patch_compare_id_counter = 0
                    for y_compare in range(0, image.width - patch_size, gap):
                        for x_compare in range(0, image.height - patch_size, gap):

                            patch.labels.append(patch_compare_id_counter)

                            patch_compare_id_counter += 1

                    # TODO something mentioned in the other file (find_label_pos)

                patch.priority = len(patch.labels) / max(patch_uncertainty, 1)

                temp_nodes_count +=1

            patches.append(patch)

            patch_id_counter += 1

            if temp_nodes_count == 7:
                break


