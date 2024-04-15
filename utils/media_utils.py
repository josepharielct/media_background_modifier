import mimetypes
import cv2
from PIL import Image
import skimage.io
import numpy as np
def is_video(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is not None:
        return mime_type.startswith('video/')
    return False

def get_media_info(media_path):
    media_info = media_path.split("/")[1]
    return media_info.split(".")[0],media_info.split(".")[1]

def read_media(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is not None:
        if mime_type.startswith('video/'):
            capture = cv2.VideoCapture(file_path)
            frames =[]
            while True:
                returns, frame = capture.read()
                if not returns:
                    break
                frames.append(frame)
            capture.release()
            return frames
        elif mime_type.startswith('image/'):
            # Read image
            image = skimage.io.imread(file_path)
            return image

def save_media(output_frame, output_path):
    if isinstance(output_frame, list):
        # Handle video frames
        save_video(output_frame, output_path)
    elif isinstance(output_frame, np.ndarray):
        print('reached image save_media')
        # Handle image
        save_image(output_frame, output_path)
    else:
        print('Unsupported output format')

def save_media(output_frame, output_path):
    if isinstance(output_frame, list):
        # Handle video frames
        output_frame(output_frame, output_path)
    elif isinstance(output_frame, np.ndarray):
        # Handle image
        save_image(output_frame, output_path)
    else:
        print('Unsupported output format')

def save_video(output_frame,output_path):
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        output = cv2.VideoWriter(f"{output_path}.mp4", fourcc, 24, (output_frame[0].shape[1], output_frame[0].shape[0]))
        for frame in output_frame:
            output.write(frame)
        output.release()
        print(f'Video saved to {output_path}.mp4')

def save_image(image, output_path):
    output_path = f"{output_path}.png"
    skimage.io.imsave(output_path, image)
    print(f'Image saved to {output_path}')

