import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Adjust the model's head for your dataset
num_classes = 2  # 1 class (weapon) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
