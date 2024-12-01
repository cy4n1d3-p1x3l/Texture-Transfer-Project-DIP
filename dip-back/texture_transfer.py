import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt
import os
import random
from typing import List, Union
from scipy.ndimage import convolve

# Function to calculate similarity between the template and sample image by SSD
def patch_similarity(template, overlap_mask, sample_img):

    # using float for accurate calculations
    template = template.astype(np.float64)
    overlap_mask = overlap_mask.astype(np.float64)
    sample_img = sample_img.astype(np.float64)
    
    # function for ssd calculation of a given channel
    def channel_ssd(channel):
        masked_template = np.zeros_like(template[:, :, channel])
        
        for i in range(template.shape[0]):  
            for j in range(template.shape[1]): 
                masked_template[i, j] = overlap_mask[i, j, channel] * template[i, j, channel]
        
        masked_ss = 0
        for i in range(masked_template.shape[0]):  
            for j in range(masked_template.shape[1]):  
                masked_ss += masked_template[i, j] ** 2
        
        convolved_1 = convolve(sample_img[:, :, channel], masked_template)
        convolved_2 = convolve(sample_img[:, :, channel] ** 2, overlap_mask[:, :, channel])
        result = masked_ss - 2 * convolved_1 + convolved_2
        
        return result
    
    # Compute total SSD by summing over all channels
    result = 0
    for i in range(template.shape[2]):
        result += channel_ssd(i)
    
    return result   

def customized_cut(err_patch):
    error_patch_shape = err_patch.shape
    h = error_patch_shape[0]
    w = error_patch_shape[1]
    grid = []  # This will store the cells directly as dictionaries
    for i in range(h):
        grid.append({"i": i, "j": 0, "cost": err_patch[i, 0], "before": None})

    for j in range(1, w):
        new_column = []
        for i in range(h):
            if i - 1 < 0:
                before = [
                    cell for cell in grid if cell["j"] == j - 1 and cell["i"] in [i, i + 1]
                ]
            elif i + 1 >= h:
                before = [
                    cell for cell in grid if cell["j"] == j - 1 and cell["i"] in [i - 1, i]
                ]
            else:
                before = [
                    cell for cell in grid if cell["j"] == j - 1 and cell["i"] in [i - 1, i, i + 1]
                ]

            # Using the inbuilt python function to find the minimum path across the matrix
            min_before = min(before, key=lambda x: x["cost"])
            current_cell = {
                "i": i,
                "j": j,
                "cost": err_patch[i, j] + min_before["cost"],
                "before": min_before,
            }
            new_column.append(current_cell)

        grid.extend(new_column)
    
    last_column = [cell for cell in grid if cell["j"] == w - 1]
    min_cell = min(last_column, key=lambda x: x["cost"])

    mask = np.zeros(err_patch.shape)
    best_path = []
    cell = min_cell

    # constructing the best path
    for _ in range(w, 0, -1):
        best_path.append((cell["i"], cell["j"]))
        mask[:cell["i"], cell["j"]] = 1
        cell = cell["before"]

    #return the best path along with the mask
    return mask, best_path    

def multiply_color(matrix_a, matrix_b):

    # Multiply the color values of the two matrices
    for i in range(matrix_a.shape[2]):
        matrix_a[:, :, i] *= matrix_b
    return matrix_a

# Main function where all the computation is done
def texture_transfer(input_image_file, target_image_file):
    input_image = io.imread(input_image_file)  # Reading the input image
    target_image = io.imread(target_image_file)  # Reading the target image
    h, w, _ = target_image.shape

    output = np.zeros_like(target_image)  # This creates an empty array for the output image

    # The following values can be changed according to need
    patch_size = 20     # The size of the patch to be picked
    overlap = 12        # The overlap between two neighboring patches
    tol = 50            # The tolerance that is acceptable
    iterations = 3
    reduction = 0.7     # The reduction in patch size after every iteration to focus on fine-tuning the results

    for n in range(iterations):
        synthesized_output = output.copy()
        output = np.zeros_like(target_image)  # Reset the output for this iteration
        alpha = 0.1 + 0.8 * n / (iterations - 1)
        offset = patch_size - overlap

        for i in range(0, h - offset, offset):
            for j in range(0, w - offset, offset):
                print(i, "/", h - offset, " ", j, "/", w - offset)  # Keeps us informed of the current situation of the processing

                template = output[i:i + patch_size, j:j + patch_size, :].copy()
                target_patch = target_image[i:i + patch_size, j:j + patch_size, :].copy()

                # Mask creation for overlap
                mask = np.zeros_like(template)
                if i > 0:
                    mask[:overlap, :, :] = 1
                if j > 0:
                    mask[:, :overlap, :] = 1

                # Calculate SSDs (Sum of Squared Differences)
                ssd_overlap = calculate_ssd(template, mask, input_image)
                ssd_target = calculate_ssd(target_patch, np.ones_like(mask), input_image)
                ssd_prev = 0
                if n > 0:
                    prev_patch = synthesized_output[i:i + patch_size, j:j + patch_size, :]
                    ssd_prev = calculate_ssd(prev_patch, np.ones_like(mask), input_image)

                # Combine the SSDs using alpha for blending overlap and target similarity
                ssd = alpha * (ssd_overlap + ssd_prev) + (1 - alpha) * ssd_target

                # Randomly select a patch within the tolerance to avoid monotony
                indices = np.argpartition(ssd.ravel(), tol - 1)[:tol]
                choices = np.column_stack(np.unravel_index(indices, ssd.shape))
                x, y = random.choice(choices)

                # Extract the selected patch
                patch = input_image[x:x + patch_size, y:y + patch_size, :]

                # Blend the selected patch with the template
                if i > 0: 
                    vertical_diff = (template[:overlap, :, :] - patch[:overlap, :, :]) ** 2
                    vertical_cut, _ = customized_cut(np.sum(vertical_diff, axis=2))
                    mask[:overlap, :, :] = vertical_cut[:, :, None]

                if j > 0:  
                    horizontal_diff = (template[:, :overlap, :] - patch[:, :overlap, :]) ** 2
                    horizontal_cut, _ = customized_cut(np.sum(horizontal_diff, axis=2).T)
                    mask[:, :overlap, :] = horizontal_cut.T[:, :, None]

                # Combine the selected patch and the existing template
                patch = patch * mask + template * (1 - mask)
                output[i:i + patch_size, j:j + patch_size, :] = patch

        # Reduce the patch size and overlap size for finer details
        patch_size = int(patch_size * reduction)
        overlap = int(overlap * reduction)

    return output.astype(np.uint8)
