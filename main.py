from utils import is_video,read_media,save_media,get_media_info
from visualize_cv2 import model, display_instances, class_names
import cv2

def main():
    #Read input
    input_media_path = "input/messi.jpg" #input
    mode = 'blue' #input ;  select from ['blue','green','gray', 'external']
    vbg = read_media('virtual_background/vbg_vertical.jpg') #input
    output_frames = []

    input_media_name, input_media_extension = get_media_info(input_media_path)
    media = read_media(input_media_path)
    frame_counter = 0    
    if is_video(input_media_path):
        print(f"Number of frame: {len(media)}")
        for frame in media:
            print(f"Frame: {frame_counter}")
            results = model.detect([frame], verbose=0)
            # Visualize results
            r = results[0]
            masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'],mode,vbg)
            cv2.putText(masked_image, f"Frame: {frame_counter}", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2) 
            output_frames.append(masked_image)
            frame_counter += 1
        save_media(output_frames, f"output/{input_media_name}")
    else:
        results = model.detect([media], verbose=1)
        r = results[0]
        masked_image = display_instances(media, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'],mode,vbg)
        save_media(masked_image, f"output/{input_media_name}")



main()