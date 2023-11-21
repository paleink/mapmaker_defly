import cv2
import numpy as np
import os
from map_to_image import drawmap


def calculate_area(shape):
    return cv2.contourArea(shape)


def eliminate_close_points(shape, threshold=2):
    # Ensure contour is a numpy array with the shape (n, 1, 2)
    shape = np.array(shape, dtype=np.int32).reshape((-1, 1, 2))
    filtered_contour = [shape[0]]

    for point in shape[1:]:
        if np.linalg.norm(point - filtered_contour[-1]) > threshold:
            filtered_contour.append(point)

    return np.array(filtered_contour, dtype=np.int32)


def calculate_centroid(vertices_inbound):
    x_coords = [v[0] for v in vertices_inbound]
    y_coords = [v[1] for v in vertices_inbound]
    _len = len(vertices_inbound)
    centroid_x = sum(x_coords) / _len
    centroid_y = sum(y_coords) / _len
    return centroid_x, centroid_y


def calculate_polygon_area(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2.0
    return area


def resize_image(image):

    # Calculate the total number of pixels
    total_pixels = image.shape[0] * image.shape[1]

    # Check if resizing is needed
    if total_pixels > 800000:
        # Calculate the resizing scale factor
        scale_factor = (800000 / total_pixels) ** 0.5

        # Calculate the new dimensions
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return resized_image
    else:
        return image


def transform(input_file):
    # Load the image
    image = resize_image(cv2.imread(input_file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 100, 200)

    # Apply dilation followed by erosion.
    # Warning! This will severely decrease final map quality, but will cut map file size by at least 1/3
    kernel = np.ones((3, 3), np.uint8)

    dilated = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.erode(dilated, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming `contours` is the list of contours from the OpenCV processing
    vertex_id = 1               # Starting ID
    vertices = {}               # Dictionary to store vertices with their IDs
    lines = []                  # List to store lines
    territories = []            # List to store territories
    min_vertices = 3            # Minimum number of vertices for a polygon
    min_area = 10               # Minimum area for a polygon
    centroid_threshold = 10     # Set a suitable threshold for centroid proximity
    area_threshold = 50         # Set a threshold for area similarity
    filtered_territories = []   # Finalized list of territories

    # Approximate contours and identify vertices (dots)
    for contour in contours:

        # Simplify the contour to eliminate close points
        simplified_contour = eliminate_close_points(contour)

        if len(simplified_contour) >= min_vertices and calculate_area(simplified_contour) >= min_area:
            territory = []
            for vertex in simplified_contour:  # Access the x,y coordinates
                x, y = tuple(vertex[0])
                if (x, y) not in vertices:
                    vertices[(x, y)] = vertex_id
                    vertex_id += 1
                territory.append(vertices[(x, y)])

            # Add lines for each pair of consecutive vertices in the territory
            for i in range(len(territory)):
                line = (territory[i], territory[(i + 1) % len(territory)])  # Loop back to the first vertex
                if line not in lines:
                    lines.append(line)

            territories.append(territory)

    vertices_inv = {v: k for k, v in vertices.items()}

    def polygons_are_similar(poly1, poly2, centroid_threshold, area_threshold):
        # Calculate centroids
        centroid1 = calculate_centroid([vertices_inv[v] for v in poly1])
        centroid2 = calculate_centroid([vertices_inv[v] for v in poly2])

        # Check centroid proximity
        distance = np.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
        if distance > centroid_threshold:
            return False

        # Compare areas
        area1 = calculate_polygon_area([vertices_inv[v] for v in poly1])
        area2 = calculate_polygon_area([vertices_inv[v] for v in poly2])
        return abs(area1 - area2) < area_threshold

    # Attempt to get rid of redundant polygons
    for i, territory1 in enumerate(territories):
        duplicate = False
        for j, territory2 in enumerate(territories):
            if i != j and polygons_are_similar(territory1, territory2, centroid_threshold, area_threshold):
                duplicate = True
                break
        if not duplicate:
            filtered_territories.append(territory1)

    # Output to file
    filename = os.path.splitext(os.path.basename(input_file))[0]

    with open(f'./temp/{filename}/{filename}_map.txt', 'w') as file:
        file.write(f'MAP_WIDTH {image.shape[1]}\n')
        file.write(f'MAP_HEIGHT {image.shape[0]}\n')

        for vertex, v_id in vertices.items():
            file.write(f'\nd {v_id} {vertex[0]} {vertex[1]}')

        for line in lines:
            file.write(f'\nl {line[0]} {line[1]}')

        for territory in filtered_territories:
            file.write('\nz ' + ' '.join(map(str, territory)))

    drawmap(vertices, lines, territories, input_file, image, f'./temp/{filename}')


"""# Display the result
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
