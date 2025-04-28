# for visualizing the frames after dtection
# libs:
import cv2
from ultralytics import YOLO

# function:
def plotting_results(path_org):

    image_orig = cv2.imread(path_org)

    y_model = YOLO('./YOLO/best.pt')

    results = y_model(image_orig)

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
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
    # plot:
    cv2.imshow('TEST1', image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

