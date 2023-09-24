import cv2
import cv2.aruco as aruco

# Define Charuco board parameters
squaresX = 10
squaresY = 10
square_length = 0.04  # Length of each square in meters
marker_length = 0.02  # Length of each ArUco marker in meters
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

# Create Charuco board
charuco_board = aruco.CharucoBoard(
    (squaresX, squaresY), square_length, marker_length, dictionary)

# Generate Charuco board image
charuco_board_image = charuco_board.generateImage((1200, 1200))
cv2.imwrite("charuco_board.png", charuco_board_image)
