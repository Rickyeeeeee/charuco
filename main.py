import cv2
import cv2.aruco as aruco
import numpy as np
import os

# Load the previously generated Charuco board
squaresX = 12
squaresY = 12
square_length = 0.04  # Length of each square in meters
marker_length = 0.02  # Length of each ArUco marker in meters
# dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
charuco_board = aruco.CharucoBoard((squaresX, squaresY),square_length, marker_length, dictionary)

# Capture images of the Charuco board for calibration
work_folder = '12x12/'
calibration_folder = work_folder + 'Calibration'
input_folder = work_folder + 'Eraser'
output_folder = input_folder + '/out'
image_files = [f for f in os.listdir(calibration_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
h, w, c = cv2.imread(os.path.join(calibration_folder, image_files[0])).shape
img_size = (h, w)
print(img_size)
# Camera calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
all_obj_points = []  # 3D points in real-world coordinates
all_img_points = []  # 2D points in image coordinates
all_corners = []
all_ids = []

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

for image_file in image_files:
    image = cv2.imread(os.path.join(calibration_folder, image_file))
    image_new = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers and corners
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is not None:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, charuco_board)
        if charuco_corners is not None and charuco_ids is not None:
            # all_corners.append(charuco_corners)
            # all_ids.append(charuco_ids)
            obj_points, img_points = charuco_board.matchImagePoints(charuco_corners, charuco_ids)
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)

# Perform camera calibration
_, mtx, dist, _, _ = cv2.calibrateCamera(
    all_obj_points, all_img_points, img_size, None, None)
print(mtx)

print('drawing images...')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
for image_file in image_files:
    image = cv2.imread(os.path.join(input_folder, image_file))
    image_new = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers and corners
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is not None:
        aruco.drawDetectedMarkers(image_new, corners, ids)
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, charuco_board)
        if charuco_corners is not None and charuco_ids is not None:
            aruco.drawDetectedCornersCharuco(image_new, charuco_corners, charuco_ids, (255, 0, 0))
            
            valid, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, mtx, dist, None, None)
            if valid:
                cv2.drawFrameAxes(image_new, mtx, dist, rvec, tvec, 0.1)
            cv2.imwrite(os.path.join(output_folder, image_file), image_new)
# Save calibration results
# np.savez("calibration_results.npz", mtx=mtx, dist=dist)
# print("Calibration successful. Calibration results saved to 'calibration_results.npz'")

# # Load the Charuco board image for pose estimation
# charuco_board_image = cv2.imread("charuco_board.png")

# # Capture a new image with the Charuco board visible
# image_with_board = cv2.imread("charuco_board.png")
# gray = cv2.cvtColor(image_with_board, cv2.COLOR_BGR2GRAY)

# # Detect markers and corners in the image
# corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
# if ids is not None:
#     _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
#         corners, ids, gray, charuco_board)

#     # Estimate the pose of the camera
#     rvec, tvec, _ = cv2.aruco.estimatePoseCharucoBoard(
#         charuco_corners, charuco_ids, charuco_board, mtx, dist, rvec, tvec)

#     # You now have the rotation (rvec) and translation (tvec) of the camera
#     print("Rotation Vector (rvec):", rvec)
#     print("Translation Vector (tvec):", tvec)