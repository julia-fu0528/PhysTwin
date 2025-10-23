import cv2
import numpy as np

markersX = 5;                # Number of markers in X direction
markersY = 7;                # Number of markers in Y direction
markerLength = 60;           # Marker side length (in pixels)
markerSeparation = 15;       # Separation between two consecutive markers in the grid (in pixels)
dictionaryId = '6x6_250';    # dictionary id
margins = markerSeparation;  # Margins size (in pixels)
borderBits = 1;              # Number of bits in marker borders

width  = markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins
height = markersY * (markerLength + markerSeparation) - markerSeparation + 2 * margins
imageSize = (int(width), int(height))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.GridBoard((markersX, markersY), markerLength, markerSeparation, dictionary)

# show created board
boardImage = cv2.aruco.drawPlanarBoard(
    board, 
    outSize=imageSize, 
    marginSize=margins, 
    borderBits=borderBits
)

# cv2.imshow('GridBoard', boardImage)
# cv2.waitKey(0)

# save image
cv2.imwrite('GridBoard.png', boardImage)