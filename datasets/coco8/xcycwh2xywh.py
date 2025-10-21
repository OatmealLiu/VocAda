def convert_to_pixel_coordinates(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Parameters:
    - x_center_norm: Normalized x center of the bounding box
    - y_center_norm: Normalized y center of the bounding box
    - width_norm: Normalized width of the bounding box
    - height_norm: Normalized height of the bounding box
    - img_width: Width of the image in pixels
    - img_height: Height of the image in pixels

    Returns:
    - (x_min, y_min, x_max, y_max): Tuple of bounding box coordinates in pixels
    """
    # Convert normalized center coordinates to pixel coordinates
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height

    # Convert normalized width and height to pixel dimensions
    box_width = width_norm * img_width
    box_height = height_norm * img_height

    # Calculate top-left (x_min, y_min) and bottom-right (x_max, y_max) coordinates
    x_min = int(x_center - box_width / 2)
    y_min = int(y_center - box_height / 2)
    x_max = int(x_center + box_width / 2)
    y_max = int(y_center + box_height / 2)

    return (x_min, y_min, x_max, y_max)


def xcycwh2xywh(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    """
    Convert normalized center coordinates to top-left corner format with width and height in pixels.

    Parameters:
    - x_center_norm: Normalized x center of the bounding box
    - y_center_norm: Normalized y center of the bounding box
    - width_norm: Normalized width of the bounding box
    - height_norm: Normalized height of the bounding box
    - img_width: Width of the image in pixels
    - img_height: Height of the image in pixels

    Returns:
    - (x_min, y_min, w, h): Tuple of bounding box coordinates with top-left corner and size in pixels
    """
    # Convert normalized center coordinates to pixel coordinates
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height

    # Convert normalized dimensions to pixel dimensions
    box_width = width_norm * img_width
    box_height = height_norm * img_height

    # Calculate top-left corner
    x_min = int(x_center - box_width / 2)
    y_min = int(y_center - box_height / 2)

    # Width and height remain the same, just converted to pixel dimensions
    w = int(box_width)
    h = int(box_height)

    return (x_min, y_min, w, h)
