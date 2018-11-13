class Image2BInpainted:

    #TODO maybe discard patch_size and gap
    def __init__(self, rgb, mask):
        self.rgb = rgb
        #self.mask = mask
        self.mask = mask // 255
        self.height = self.rgb.shape[0]
        self.width = self.rgb.shape[1]


class Patch:


    def __init__(self, patch_id, overlap_source_region, overlap_target_region, x_coord, y_coord, priority=None, labels=None, pruned_labels=None, differences=None, committed=False):

        # properties of all patches
        self.patch_id = patch_id
        self.overlap_source_region = overlap_source_region
        self.overlap_target_region = overlap_target_region
        self.x_coord = x_coord
        self.y_coord = y_coord

        # properties of patches having an intersection with the target region (i.e. patches to be inpainted)
        if priority is None:
            self.priority = 0
        else:
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

        if committed is None:
            self.committed = False
        else:
            self.committed = committed


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