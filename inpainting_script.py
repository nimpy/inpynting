import sys
from main import inpaint_image

print(len(sys.argv))
assert len(sys.argv) == 7, "Usage: python inpainting_script.py patch_size thresh_uncertainty use_descriptors store_descriptors folder_path image_filename mask_filename"

patch_size = int(sys.argv[1])
assert patch_size % 2 == 0, "Patch size needs to be an even number!"
stride = patch_size // 2

thresh_uncertainty = int(sys.argv[2])

max_nr_labels = 10
max_nr_iterations = 10

use_descriptors = bool(sys.argv[3])
store_descriptors = bool(sys.argv[4])

folder_path = sys.argv[5]  # don't forget to also change the descriptor
image_filename = sys.argv[6]
mask_filename = sys.argv[7]

# Don't forget to also change the descriptor!!!

inpaint_image(folder_path, image_filename, mask_filename, patch_size, stride, thresh_uncertainty, max_nr_labels,
              max_nr_iterations, use_descriptors, store_descriptors)

# Example: python inpainting_script.py 16 6755360 1 1 /scratch/data/panel13/crops image_0_406.tif mask_0_406.png

# patch_size = 16  # needs to be an even number
# stride = patch_size // 2 #TODO fix problem when stride isn't exactly half of patch size!
# thresh_uncertainty = 6755360 #10360 #5555360 #35360 #85360 #155360 # 6755360  #155360  # 100000 #155360 #255360 #6755360
# max_nr_labels = 10
# max_nr_iterations = 10
# use_descriptors = True
# store_descriptors = True
#
# folder_path = '/scratch/data/panel13/crops'  # don't forget to also change the descriptor
# image_filename = 'image_0_406.tif'
# mask_filename = 'mask_0_406.png'