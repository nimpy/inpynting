class Image2BInpainted:

    #TODO maybe discard patch_size and gap
    def __init__(self, rgb, mask):
        self.rgb = rgb
        #self.mask = mask
        self.mask = mask // 255
        self.height = self.rgb.shape[0]
        self.width = self.rgb.shape[1]


class Patch:


    def __init__(self, patch_id, overlap_source_region, overlap_target_region, x_coord, y_coord,
                 priority=0, labels=[], pruned_labels=[], differences={}, committed=False,
                 potential_matrix_up=None, potential_matrix_down=None, potential_matrix_left=None, potential_matrix_right=None,
                 label_cost=None, local_likelihood=None, mask=None,
                 messages=None, beliefs=None):

        # properties of all patches
        self.patch_id = patch_id
        self.overlap_source_region = overlap_source_region
        self.overlap_target_region = overlap_target_region
        self.x_coord = x_coord
        self.y_coord = y_coord

        # properties of patches having an intersection with the target region (i.e. patches to be inpainted)
        self.priority = priority
        self.labels = labels
        self.pruned_labels = pruned_labels
        self.differences = differences
        self.committed = committed

        self.potential_matrix_up = potential_matrix_up
        self.potential_matrix_down = potential_matrix_down
        self.potential_matrix_left = potential_matrix_left
        self.potential_matrix_right = potential_matrix_right

        self.label_cost = label_cost
        self.local_likelihood = local_likelihood

        self.mask = mask

        self.messages = messages
        self. beliefs = beliefs


    def prune_labels(self, MAX_NB_LABELS):

        sorted_differences = sorted(self.differences.items(), key=lambda kv: kv[1])[:MAX_NB_LABELS] #, reverse=True
        self.pruned_labels = [label for (label, diff) in sorted_differences]


    def get_up_neighbor_position(self, image, patch_size, gap):

        if self.x_coord < gap:
            return None

        neighbor_x_coord = self.x_coord - gap
        neighbor_y_coord = self.y_coord

        return coordinates_to_position(neighbor_x_coord, neighbor_y_coord, image.width, patch_size, gap)

    def get_down_neighbor_position(self, image, patch_size, gap):

        if self.x_coord > image.height - (patch_size + gap):
            return None

        neighbor_x_coord = self.x_coord + gap
        neighbor_y_coord = self.y_coord

        return coordinates_to_position(neighbor_x_coord, neighbor_y_coord, image.width, patch_size, gap)

    def get_left_neighbor_position(self, image, patch_size, gap):

        if self.y_coord < gap:
            return None

        neighbor_x_coord = self.x_coord
        neighbor_y_coord = self.y_coord - gap

        return coordinates_to_position(neighbor_x_coord, neighbor_y_coord, image.width, patch_size, gap)

    def get_right_neighbor_position(self, image, patch_size, gap):

        if self.y_coord > image.width - (patch_size + gap):
            return None

        neighbor_x_coord = self.x_coord
        neighbor_y_coord = self.y_coord + gap

        return coordinates_to_position(neighbor_x_coord, neighbor_y_coord, image.width, patch_size, gap)


def coordinates_to_position(x, y, image_width, patch_size, gap):
        return (y // gap) * len(range(0, image_width - patch_size + 1, gap)) + (x // gap)