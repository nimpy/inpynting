def patch_diff(patch_1, patch_2):
    if patch_1.shape != patch_2.shape:
        print("Patches don't have the same shape.")
        return
    height = patch_1.shape[0]
    width = patch_1.shape[1]
    nr_channels = patch_1.shape[2]
    ssd = 0
    for i in range(height):
        for j in range (width):
            for k in range(nr_channels):
                #diff = patch_1[i][j][k] - patch_2[i][j][k]
                diff = patch_1[i,j,k] - patch_2[i,j,k]
                ssd += diff * diff
    return ssd