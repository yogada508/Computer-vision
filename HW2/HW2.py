import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_rgb, img_gray

def SIFT(gray_img):
    '''
        Use SIFT to get the keypoint and descriptor of the keypoint

        Args:
            gray_img: SIFT need gray scale image as input
        
        Returns:
            keypoint: The keypoints tuple 
            des: The 128-dim of each keypoint

    '''
    sift_detector = cv2.SIFT_create()
    keypoint, des = sift_detector.detectAndCompute(gray_img, None)
    return keypoint, des

def match_keypoint(kp_left, kp_right, des_left, des_right):
    match = []
    K = 2
    ratio = 0.7

    for i in range(des_left.shape[0]):
        distance = np.linalg.norm(des_left[i] - des_right, axis=1)
        nearest_neighbor_ids = distance.argsort()[:K]
        if (distance[nearest_neighbor_ids[0]] < ratio * distance[nearest_neighbor_ids[1]]):
            match.append([kp_left[i].pt, kp_right[nearest_neighbor_ids[0]].pt])
    
    # The result will have duplicate keypoint but with different descriptor, 
    # but we only interest at keypoiny coordinate to image stitching, so remove duplicate ones.
    remove_dupliacte = []
    remove_dupliacte = sorted(match)
    remove_dupliacte = list(remove_dupliacte for remove_dupliacte, _ in itertools.groupby(remove_dupliacte))
    
    return remove_dupliacte

def homography(pairs):
    """
        solve p1' = H * p1, (H is homograpy matrix we want to get)

        Args: 
            pairs = [[(x1, y1), (x1', y1')], [(x2,y2), (x2', y2')], ...]
            pairs should have at least 4 pairs.
            
    """
    A = []
    for i in range(len(pairs)):
        p1 = pairs[i][0] #(x1, y1)
        p2 = pairs[i][1] #(x1', y1')

        # [x1, y1, 1, 0, 0, 0, -x1'x1, -x1'y1, -x1']
        A.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]])
        # [0, 0, 0, x1, y1, 1, -y1'x1, -y1'y1, -y1']
        A.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]])

    # use SVD to solve the linear equation A*H = 0
    # find the smallest number in s and H = corresponding vector in vt
    # np.linalg.svd() will return s in descending order, so we can get the last vector in vt to get H
    u, s, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)

    # normalize h22 to 1
    H = H/H[2, 2]
    return H

def RANSAC(match_point, iteration, threshold):
    max_inlier = 0
    best_H = None
    
    for i in range(iteration):
        # get 4 random points at match_point and calculate homography matrix
        random_match = random.sample(match_point, 4)
        H = homography(random_match)
        
        # find the best homography matrix that has the max number of inliners
        inlinr_num = 0
        for j in range(len(match_point)):
            if match_point[j] not in random_match:
                src_point, dst_point = match_point[j]
                src_point = np.hstack((src_point, [1])) # add z-axis

                correspond_point = np.matmul(H, src_point)
                correspond_point = correspond_point / correspond_point[2]
                if (np.linalg.norm(correspond_point[:2] - dst_point) < threshold):
                    inlinr_num += 1
                
        if max_inlier < inlinr_num:
            max_inlier = inlinr_num
            best_H = H
        
        # when inlier/match_count > 80%, end loop
        if max_inlier/len(match_point) > 0.8:
            break

    
    print(f"best inlier/match_count: {max_inlier}/{len(match_point)}")
    return best_H

def warp_img(img1, img2, H):
    # Get the projected points coordinate of the corner of img1
    height_l, width_l, channel_l = img1.shape
    corners = [[0, 0, 1], [height_l, 0, 1], [height_l, width_l, 1], [0, width_l, 1]]
    corners_new = [np.dot(H, corner) for corner in corners]
    corners_new = np.array(corners_new).T 
    x_news = corners_new[0] / corners_new[2]
    y_news = corners_new[1] / corners_new[2]
    y_min = min(y_news)
    x_min = min(x_news)

    # Use affine matrix to translate the disappear part
    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = np.dot(translation_mat, H)
    
    # Get height, width
    height_r, width_r, channel_r = img2.shape
    height_new = int(round(abs(y_min) + height_r))
    width_new = int(round(abs(x_min) + width_r))
    size = (width_new, height_new)

    # left image
    warped_l = cv2.warpPerspective(src=img1, M=H, dsize=size)

    # right image
    warped_r = cv2.warpPerspective(src=img2, M=translation_mat, dsize=size)

    return warped_l, warped_r

def linear_blending(warped_l, warped_r):
    _, mask_left = cv2.threshold(warped_l, 0, 1, cv2.THRESH_BINARY)
    _, mask_right = cv2.threshold(warped_r, 0, 1, cv2.THRESH_BINARY)
    height, width, _ = mask_left.shape

    alpha_mask = np.zeros((height, width))
    beta_mask = np.zeros((height, width))
    overlap_mask = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if np.count_nonzero(mask_left[i, j]) > 0:
                alpha_mask[i][j] = 1
            if np.count_nonzero(mask_right[i, j]) > 0:
                beta_mask[i][j] = 1
            if alpha_mask[i][j] and beta_mask[i][j]:
                overlap_mask[i][j] = 1

    
    for i in range(height):
        overlap_idx = [j for j,value in enumerate(overlap_mask[i]) if value == 1]
        if overlap_idx:
            start_idx = overlap_idx[0]
            end_idx = overlap_idx[-1]
            
            ratio = 1 / (end_idx - start_idx + 1)
            for j in overlap_idx:
                alpha = 1 - (ratio * (j - start_idx))
                beta = 1 - alpha

                alpha_mask[i][j] = alpha
                beta_mask[i][j] = beta

    # add new axis to use numpy broadcasting
    alpha_mask = alpha_mask[:,:,np.newaxis]
    beta_mask = beta_mask[:,:,np.newaxis]
    stitch_result = (warped_l * alpha_mask + warped_r * beta_mask).astype(np.uint8)

    return stitch_result

def normal_stitch(warped_l, warped_r):
    alpha = 0.5
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(warped_l, alpha, warped_r, beta, 0.0)

    _, mask_left = cv2.threshold(warped_l, 0, 1, cv2.THRESH_BINARY)
    _, mask_right = cv2.threshold(warped_r, 0, 1, cv2.THRESH_BINARY)
    overlap = (mask_left + mask_right - 1)

    result = warped_l + warped_r - overlap*dst

    return result

def stitch_two_image(images_path):
    left_img, left_img_gray = read_img(images_path[0])
    right_img, right_img_gray = read_img(images_path[1])

    left_kp, left_des = SIFT(left_img_gray)
    right_kp, right_des = SIFT(right_img_gray)

    print("Find matched keypoint...")
    match_point = match_keypoint(left_kp, right_kp, left_des, right_des)

    print("Find best homography matrix...")
    H = RANSAC(match_point, iteration=3000, threshold=3)

    print("Warp and blend images")
    warp_l, warp_r = warp_img(left_img, right_img, H)
    warp = linear_blending(warp_l, warp_r)

    return warp
    

if __name__ == '__main__':
    print("Stitch m1.jpg and m2.jpg")
    images = ["./test/m1.jpg", "./test/m2.jpg"]
    result = stitch_two_image(images)
    print("Save stitch two images result './test/m12.jpg'\n")
    plt.imsave("./test/m12.jpg", result)

    print("Stitch m12.jpg and m3.jpg")
    images = ["./test/m12.jpg", "./test/m3.jpg"]
    result = stitch_two_image(images)
    print("Save stitch two images result './test/m123.jpg'\n")
    plt.imsave("./test/m123.jpg", result)

    print("Stitch m123.jpg and m4.jpg")
    images = ["./test/m123.jpg", "./test/m4.jpg"]
    result = stitch_two_image(images)
    print("Save stitch two images result './test/m1234.jpg'\n")
    plt.imsave("./test/m1234.jpg", result)