from utils import is_video,read_media,save_media,get_media_info
from visualize_cv2 import model, display_instances, class_names
import cv2

def main():
    #Read input
    input_media_path = "input/1.jpg"
    input_media_name, input_media_extension = get_media_info(input_media_path)
    print(input_media_extension)
    media = read_media(input_media_path)

    output_frames = []
    #Isolate Media
    ## get three modes: greyscale, green, blue
   # mode = input("Choose isolation mode: greyscale, green, blue")
   # if mode == 'green':
    print(is_video(input_media_path))
    if is_video(input_media_path):
        for frame in media:
            results = model.detect([frame], verbose=1)
            # Visualize results
            r = results[0]
            masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
            cv2.imshow("masked_image",masked_image)
            output_frames.append(masked_image)
            save_media(output_frames, f"output/{input_media_name}",input_media_extension)
    else:
        results = model.detect([media], verbose=1)
        r = results[0]
        masked_image = display_instances(media, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
        print('reached_here')
        save_media(masked_image, f"output/{input_media_name}")
        print('saved_successfully')
main()