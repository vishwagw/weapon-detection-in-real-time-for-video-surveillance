# this is the main program script:
# libs:
import cv2

# import detection files:
from video_detector import detectig_from_videos
from img_detector import detecting_anomaly_img
from plotting import plotting_results

# the main function to combine the system:
def main_program():

    # 1. Test detecting objects in a single image:
    # TODO: define the input image path:
    # this image path is the image path for detecting images file.
    img_path = './Test_imgs/test1.jpg'
    # for output image:
    result_img_path =  detecting_anomaly_img(img_path)
    print("Object detection result saved at:", result_img_path)

    # 2. Test detecting objects in a video:
    # TODO: include the path:
    vid_path = './Test_vid/sample_vids/'
    result_video_path = detectig_from_videos(vid_path)
    print("Object detection result video saved at:", result_video_path)

    # 3. Test detecting objects in an image and displaying it:
    path_org = 'test_image.jpg'
    plotting_results(path_org)

if __name__ == "__main__":
    main_program()



