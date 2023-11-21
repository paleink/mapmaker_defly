import cv2
import numpy as np
import os


def drawmap(vertices, lines, territories, input_file, image, output_folder=""):

    # Initialize a blank canvas (all gray)
    canvas = np.full((image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)

    # Draw each dot (vertex)
    for cords, id in vertices.items():  # Adjusted to use dictionary
        cv2.circle(canvas, cords, radius=1, color=(200, 200, 200), thickness=-1)  # Light gray dots

    # Draw each line (edge)
    for line in lines:
        dot1_id, dot2_id = line
        dot1_cords = [key for key, value in vertices.items() if value == dot1_id][0]
        dot2_cords = [key for key, value in vertices.items() if value == dot2_id][0]
        cv2.line(canvas, dot1_cords, dot2_cords, color=(150, 150, 150), thickness=1)  # Medium gray lines

    # Fill each territory
    for territory in territories:
        points = np.array([[key for key, value in vertices.items() if value == dot_id][0] for dot_id in territory],
                          np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [points], color=(220, 220, 220))  # Light gray territories

    # Save the image
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file_name = os.path.join(output_folder, f"{base_name}_preview.jpg")

    cv2.imwrite(output_file_name, canvas)
    # Or display the image in a window
    # cv2.imshow('Map Preview', canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
