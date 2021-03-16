import os
from io import BytesIO
from typing import Dict, List

from PIL import Image

import utility


def get_images(directory: str) -> List[str]:
    files = os.listdir(directory)
    return utility.filter_files_by_extension(files, ('.jpg', '.jpeg', '.png'))


def get_image_directories(base_directory: str) -> Dict[str, List[str]]:
    image_directories: Dict[str, List[str]] = {}

    for dir_info in os.walk(base_directory):
        images = get_images(dir_info[2])
        if len(images) > 0:
            image_directories[dir_info[0]] = images

    return image_directories


def create_thumbnail(bytestream: BytesIO) -> bytes:
    img = Image.open(bytestream)
    img.thumbnail((160, 160))
    stream = BytesIO()
    img.save(stream, format='JPEG')
    stream.seek(0)
    return stream.getvalue()


def get_35mm_sensor_object_height(object_height: int, image_height: int) -> float:
    return (24 * object_height) / image_height

# Based on https://www.scantips.com/lights/fieldofviewmath.html
# https://en.wikipedia.org/wiki/Image_sensor_format (https://en.wikipedia.org/wiki/File:SensorSizes.svg)
# https://exiftool.org/TagNames/EXIF.html (FocalLengthIn35mmFormat)


def get_35mm_distance_to_object(real_object_height: int,
                                pixel_object_height: int,
                                image_height: int,
                                focal_length: int) -> float:
    """Calculates an estimated distance to any given object

    :param real_object_height: The reallife object height in mm
    :param pixel_object_height: The image object height in pixels
    :param image_height: The total height of the image in pixels
    :param focal_length: The 35 mm focal length.
    This can often be found in the EXIF data under the tag FocalLengthIn35mmFilm
    :returns: An estimated distance to the object in mm
   """

    sensor_object_height = get_35mm_sensor_object_height(
        pixel_object_height, image_height)
    return (real_object_height * focal_length) / sensor_object_height
