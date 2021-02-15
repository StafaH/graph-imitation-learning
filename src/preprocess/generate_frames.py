#!/usr/bin/env python

"""generate_frames.py: Generates individual frames from video data.

    Arguments:
        Required:
        -f, --file = name and location of file you want to parse
        -o, --output = output folder for images

        Optional:
        -r, --fps = number of frames to extract per second of video (default = 1)
        -n, --name = prefix used to name the file (default = None)
        -c, --crop = crop the video using the box-coordinates (left (x), top (y), right (x), bottom (y))

    Usage: generate_frames.py [-h] -f FILE -o OUTPUT [-r FPS] [-n NAME] [-c CROP]
    Usage: generate_frames.py [-h] --file FILE --output OUTPUT [--fps FPS] [--name NAME] [--crop CROP]
    Example: python src/preprocess/generate_frames.py --file data/raw/incision_1.mp4 --output data/processed/incision_2/ --fps 0.25 --name incision --crop "400, 0, 1120, 720"
"""

import cv2
import argparse
from os.path import isdir, isfile


def coords(s):
    try:
        left, top, right, bottom = map(int, s.split(','))
        return left, top, right, bottom
    except:
        raise argparse.ArgumentTypeError("Coordinates must be in format \"left (x), top (y), right (x), bottom (y)\" e.g. \"400, 0, 1120, 720\"")


def main():
    parser = argparse.ArgumentParser(description='Seperate video file into individual frames')
    parser.add_argument('-f', '--file', type=str, required=True, help='name and location of file you want to parse')
    parser.add_argument('-o', '--output', type=str, required=True, help='output folder for images')
    parser.add_argument('-r', '--fps', type=float, default=1, help='number of frames to extract per second of video (default = 1)')
    parser.add_argument('-n', '--name', type=str, default='', help='prefix used to name the file (default = None)')
    parser.add_argument('-c', '--crop', type=coords, dest="crop", default='-1, -1, -1, -1', help="bounding box to crop from in the format of 'left, top, right, bottom', where the top-left of the image is the origin (0,0)")
    args = parser.parse_args()

    video_path = args.file
    output_folder = args.output
    frame_rate = args.fps
    file_name = args.name
    left, top, right, bottom = args.crop

    # Check if input file exists
    if not isfile(video_path):
        print('Input file does not exist')
        return

    if not isdir(output_folder):
        print('Output folder does not exist')
        return

    # Use frame rate to calculate time increment for extracting frames
    time_increment = 1 / frame_rate

    time = 0
    frame_counter = 1

    video_capture = cv2.VideoCapture(video_path)

    # Read the first frame of the video
    video_capture.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
    success, image = video_capture.read()
    if success:
        if top != -1:
            # OpenCV image cropping is based on slice img[y:y+h, x:x+w]
            image = image[top:bottom, left:right]
        cv2.imwrite(output_folder + file_name + str(frame_counter) + ".jpg", image) 

    while success:
        frame_counter += 1
        time = time + time_increment
        time = round(time, 2)

        # We set our pointer to the frame at the corresponding time point in milliseconds
        video_capture.set(cv2.CAP_PROP_POS_MSEC, time * 1000)

        success, image = video_capture.read()
        if success:
            if top != -1:
                image = image[top:bottom, left:right]
            cv2.imwrite(output_folder + file_name + str(frame_counter) + ".jpg", image)

    video_capture.release()


if __name__ == '__main__':
    main()