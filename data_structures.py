class Image2BInpainted:

    #TODO maybe discard patch_size and gap
    def __init__(self, rgb, mask, inpainted=None):
        self.rgb = rgb
        self.mask = mask
        self.height = self.rgb.shape[0]
        self.width = self.rgb.shape[1]
        self.inpainted = inpainted


class Patch:


    def __init__(self, patch_id, overlap_source_region, overlap_target_region, x_coord, y_coord,
                 priority=0, labels=None, pruned_labels=None, differences=None, committed=False,
                 potential_matrix_up=None, potential_matrix_down=None, potential_matrix_left=None, potential_matrix_right=None,
                 label_cost=None, local_likelihood=None, mask=None,
                 messages=None, beliefs=None, beliefs_new=None):

        # properties of all patches
        self.patch_id = patch_id
        self.overlap_source_region = overlap_source_region
        self.overlap_target_region = overlap_target_region
        self.x_coord = x_coord
        self.y_coord = y_coord

        # properties of patches having an intersection with the target region (i.e. patches to be inpainted)
        self.priority = priority
        if labels is None:
            self.labels = []
        else:
            self.labels = labels
        if pruned_labels is None:
            self.pruned_labels = []
        else:
            self.pruned_labels = pruned_labels
        if differences is None:
            self.differences = {}
        else:
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
        self.beliefs_new = beliefs_new



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


UP = 1
DOWN = -1
LEFT = 2
RIGHT = -2


def opposite_side(side):
    return -side


def get_half_patch_from_patch(patch, gap, side):
    patch_size = patch.shape[0]
    if side == UP:
        half_patch = patch[0: gap, :, :]
    elif side == DOWN:
        half_patch = patch[gap: patch_size, :, :]
    elif side == LEFT:
        half_patch = patch[:, 0: gap, :]
    else:
        half_patch = patch[:, gap: patch_size, :]
    return half_patch
