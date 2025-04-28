# python script for anomaly detecting from saved videos
# libs:
import cv2
from ultralytics import YOLO

def detectig_from_videos(vid_path):

    y_model = YOLO('./YOLO/best.pt')
    video_capture = cv2.VideoCapture(vid_path)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # TODO: incldue output video path
    result_vid_pth = './Test_vid/Results/sample_vid1.mp4' # either mp4 or avi
    out = cv2.VideoWriter(result_vid_pth, fourcc, 20.0, (width, height))

    # detection process loop:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = y_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                 if conf[pos] >= 0.5:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)
    video_capture.release()
    out.release()

    return result_vid_pth
