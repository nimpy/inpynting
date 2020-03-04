import numpy as np

class Image2BInpainted:

    USING_RBG_VALUES = 0
    USING_IR = 1
    USING_STORED_DESCRIPTORS_HALVES = 2
    USING_STORED_DESCRIPTORS_CUBE = 3

    def __init__(self, rgb, mask, edges, patch_size, stride, inpainting_approach=-1,
                 descriptor_cube=None, half_patch_landscape_descriptor_cube=None, half_patch_portrait_descriptor_cube=None,
                 ir=None, patch_descriptors=None,
                 half_patch_landscape_descriptors=None, half_patch_portrait_descriptors=None,
                 inpainted=None, order_image=None):
        self.rgb = rgb
        self.mask = mask
        self.edges = edges
        self.patch_size = patch_size
        self.stride = stride
        self.height = self.rgb.shape[0]
        self.width = self.rgb.shape[1]
        self.inpainting_approach = inpainting_approach
        self.descriptor_cube = descriptor_cube
        self.half_patch_landscape_descriptor_cube = half_patch_landscape_descriptor_cube
        self.half_patch_portrait_descriptor_cube = half_patch_portrait_descriptor_cube
        self.ir = ir
        self.patch_descriptors = patch_descriptors
        self.half_patch_landscape_descriptors = half_patch_landscape_descriptors
        self.half_patch_portrait_descriptors = half_patch_portrait_descriptors
        self.inpainted = inpainted
        self.order_image = order_image
        # TODO
        self.inverted_mask_3ch = 1 - np.repeat(mask, 3, axis=1).reshape((self.height, self.width, 3))
        self.inverted_mask_Nch = None

# a patch to be inpainted
class Node:

    def __init__(self, node_id, overlap_source_region, x_coord, y_coord,
                 priority=0, labels=None, pruned_labels=None, differences=None, committed=False, additional_differences=None,
                 potential_matrix_up=None, potential_matrix_down=None, potential_matrix_left=None, potential_matrix_right=None,
                 label_cost=None, local_likelihood=None, mask=None,
                 messages=None, beliefs=None, beliefs_new=None):

        self.node_id = node_id
        self.overlap_source_region = overlap_source_region
        self.x_coord = x_coord
        self.y_coord = y_coord

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
        if additional_differences is None:
            self.additional_differences = {}
        else:
            self.additional_differences = additional_differences

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

        sorted_differences = sorted(self.additional_differences.items(), key=lambda kv: kv[1])[:MAX_NB_LABELS] #, reverse=True
        self.pruned_labels = [label for (label, diff) in sorted_differences]


    def get_up_neighbor_position(self, image):

        if self.x_coord < image.stride:
            return None

        neighbor_x_coord = self.x_coord - image.stride
        neighbor_y_coord = self.y_coord

        return coordinates_to_position(neighbor_x_coord, neighbor_y_coord, image.height, image.patch_size)

    def get_down_neighbor_position(self, image):

        if self.x_coord > image.height - (image.patch_size + image.stride):
            return None

        neighbor_x_coord = self.x_coord + image.stride
        neighbor_y_coord = self.y_coord

        return coordinates_to_position(neighbor_x_coord, neighbor_y_coord, image.height, image.patch_size)

    def get_left_neighbor_position(self, image):

        if self.y_coord < image.stride:
            return None

        neighbor_x_coord = self.x_coord
        neighbor_y_coord = self.y_coord - image.stride

        return coordinates_to_position(neighbor_x_coord, neighbor_y_coord, image.height, image.patch_size)

    def get_right_neighbor_position(self, image):

        if self.y_coord > image.width - (image.patch_size + image.stride):
            return None

        neighbor_x_coord = self.x_coord
        neighbor_y_coord = self.y_coord + image.stride

        return coordinates_to_position(neighbor_x_coord, neighbor_y_coord, image.height, image.patch_size)


def coordinates_to_position(x, y, image_height, patch_size):
    return y * len(range(0, image_height - patch_size + 1)) + x


def position_to_coordinates(position, image_height, patch_size):
    x = position % (image_height - patch_size + 1)
    y = position // (image_height - patch_size + 1)
    return x, y


UP = 1
DOWN = -1
LEFT = 2
RIGHT = -2


def opposite_side(side):
    return -side


def get_half_patch_from_patch(patch, stride, side):
    patch_size = patch.shape[0]
    if side == UP:
        half_patch = patch[0: stride, :, :]
    elif side == DOWN:
        half_patch = patch[stride: patch_size, :, :]
    elif side == LEFT:
        half_patch = patch[:, 0: stride, :]
    else:
        half_patch = patch[:, stride: patch_size, :]
    return half_patch
