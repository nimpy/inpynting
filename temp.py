from data_structures import Patch, Image2BInpainted, coordinates_to_position
import numpy as np

def testing_prune_labels():
    patch = Patch(0, False, False, 0, 0, 0, [1, 2, 3, 4, 5], None, {1: 7, 2: 9, 3: 6, 4: 8, 5: 10}, False) # 3 1 4 2 5
    patch.prune_labels(1)
    print(patch.pruned_labels)

def testing_get_neighbor_position():
    patch = Patch(31, False, False, 12, 0)
    image = Image2BInpainted(np.zeros((16, 16, 3)), np.zeros((16, 16, 3)))
    print(patch.get_down_neighbor_position(image, 4, 1))


def testing_coordinates_to_position():
    x = 1
    y = 1
    image_width = 16
    patch_size = 4
    gap = 1
    pos = coordinates_to_position(x, y, image_width, patch_size, gap)
    print(pos)

def change_patch(patch):
    patch.differences[1] = 0

def testing_change_patch():
    patch = Patch(0, False, False, 0, 0, 0, [1, 2, 3, 4, 5], None, {1: 7, 2: 9, 3: 6, 4: 8, 5: 10}, False) # 3 1 4 2 5
    change_patch(patch)
    print(patch.differences[1])


def main():
    # testing_prune_labels()
    print("---")
    # testing_get_neighbor_position()
    print("---")
    # testing_coordinates_to_position()
    print("---")
    testing_change_patch()

if __name__ == "__main__":
    main()