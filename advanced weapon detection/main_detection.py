import cv2
import torch


# Load your trained model
model = torch.load('weapon_detection_model.pth')
model.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_tensor = preprocess(frame)
    
    # Run the model
    with torch.no_grad():
        outputs = model([input_tensor])
    
    # Process outputs
    for box, score in zip(outputs[0]['boxes'], outputs[0]['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    cv2.imshow('Weapon Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
