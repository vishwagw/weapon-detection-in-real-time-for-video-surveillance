# libs
import cv2
from ultralytics import YOLO

# building the function for image detection:
def detecting_anomaly_img(img_path):
    img_org = cv2.imread(img_path)

    y_model = YOLO('./YOLO/best.pt')
    
    results = y_model(img_org)
    
    # detection loop:
    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            xmin, ymin, xmax, ymax = detection
            label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
            color = (0, int(cls[pos]), 255)
            cv2.rectangle(img_org, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(img_org, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # TODO: include result path -result img name
    result_path = './Test_imgs/results/tr1.jpg'
    cv2.imwrite(result_path, img_org)
    return result_path

detecting_anomaly_img()


