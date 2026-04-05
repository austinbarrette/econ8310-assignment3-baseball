"""
Econ 8310 - Assignment 3: Baseball Detection
Austin Barrette

This script demonstrates a complete pipeline for detecting baseballs
in training videos using a fine-tuned Faster R-CNN model in PyTorch.

When run this script will:
    1. Load baseball videos and XML annotations into a custom dataset
    2. Train a Faster R-CNN neural network to detect baseballs
    3. Save the trained model weights to baseball_model.pt
    4. Load the saved model back in and run inference on a new video
       to confirm the model works without retraining

Satisfies:
    - Custom data loader that can import new videos
    - Working trained neural network using only pytorch
    - Saving weights & importing script for evaluation
"""

#IMPORTS
import os
import xml.etree.ElementTree as ET
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

# Use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")


#Parse for XML ANNOTATIONS
"""Parses each CVAT annotation file to extract bounding box coordinates for every baseball visible in each frame.
Returns a dictionary mapping frame numbers to a list of bounding boxes which should equal one bounding box per baseball in that frame."""

def parse_xml_annotations(xml_path):

    #ET is Python's built-in XML reader
    #root is the top level of the XML tree
    tree = ET.parse(xml_path)
    root = tree.getroot()

    #Get original video dimensions (height and width) from the XML data (we normalize coordinates later)
    #Float() converts from a string to an actual number so we can do math later
    height = float(root.find('.//original_size/height').text)
    width  = float(root.find('.//original_size/width').text)

    #creates empty dictionary for each frame
    frame_boxes = {}

    #XML structure organizes annotations as tracks containing boxes
    #One track per baseball that was annotated
    #Each box in a track is the baseball's coordinates in each frame
    for track in root.findall('track'):
        for box in track.findall('box'):

            #Skips the frames where the ball is marked outside=1 (not visible)
            #We only want frames where the ball is visible That's what we're teaching/training the model to detect
            if box.attrib.get('outside', '0') == '1':
                continue

            #Stores pixel coordinates for each frame (box)
            frame_num = int(box.attrib['frame'])
            xtl = float(box.attrib['xtl'])  # x top-left
            ytl = float(box.attrib['ytl'])  # y top-left
            xbr = float(box.attrib['xbr'])  # x bottom-right
            ybr = float(box.attrib['ybr'])  # y bottom-right

            #Normalize to [0, 1]
            #This makes the coordinates work regardless of image size
            xtl_n = max(0.0, min(1.0, xtl / width))
            ytl_n = max(0.0, min(1.0, ytl / height))
            xbr_n = max(0.0, min(1.0, xbr / width))
            ybr_n = max(0.0, min(1.0, ybr / height))

            #Adds boxes to the lists for each frame
            #If frame not seen yet, create a new empty list first
            if frame_num not in frame_boxes:
                frame_boxes[frame_num] = []
            frame_boxes[frame_num].append((xtl_n, ytl_n, xbr_n, ybr_n))

    return frame_boxes


#CUSTOM DATA LOADER
""" This satisfies the requirement for a custom data loader that can import a new video or set of videos.
Data loader follows the PyTorch object detection dataset format required by Faster R-CNN.
__getitem__ returns the target dictionary. """

class BaseballVideoDataset(Dataset):

    def __init__(self, video_xml_pairs, img_size=224):
        self.img_size = img_size
        self.samples  = []  #stores all (frame_array, list_of_boxes) pairs

        for video_path, xml_path in video_xml_pairs:
            print(f"  Loading: {os.path.basename(video_path)}")

            #Parses XML to get coordinates
            frame_boxes = parse_xml_annotations(xml_path)

            #Setup failure Warnings
            if not frame_boxes:
                print(f"    Warning: No annotations found!")
                continue

            #Did the video actually open successfully
                #Check File Path
                #Is the file corrupt?
                #Check Video Format
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"    Warning: Could not open video!")
                continue

            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  #reached end of video

                #Only store frames that have at least one bounding box
                if frame_num in frame_boxes:
                    #OpenCV loads BGR by default.  We need to convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #Resize to standard dimensions for the model
                    frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
                    self.samples.append((frame_resized, frame_boxes[frame_num]))

                #frame counter
                frame_num += 1

            cap.release()
            print(f"    Frames: {frame_num} | Annotated: {len(frame_boxes)}")

        print(f"\n  Total samples loaded: {len(self.samples)}")

    def __len__(self):
        """Required by PyTorch: returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Required by PyTorch: returns one (image, target) pair.
        Target dictionary format is required by Faster R-CNN.
        """
        frame, boxes = self.samples[idx]

        #Convert numpy array to float PyTorch tensor (C, H, W)
        #Divide by 255 to normalize pixel values to [0, 1]
        image = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0

        #Convert normalized [0,1] coordinates back to pixel values
        # Faster R-CNN expects actual pixel coordinates not normalized ones
        boxes_pixels = []
        for (xtl, ytl, xbr, ybr) in boxes:
            boxes_pixels.append([
                xtl * self.img_size,
                ytl * self.img_size,
                xbr * self.img_size,
                ybr * self.img_size
            ])

        #Build target dictionary:
        target = {}

        #All bounding boxes in this frame as a float tensor of shape [N, 4]
        target["boxes"] = torch.tensor(boxes_pixels, dtype=torch.float32)

        #All boxes are class 1 (baseball).  0 is reserved for background
        target["labels"] = torch.ones(len(boxes_pixels), dtype=torch.int64)

        #Unique frame identifier which must be a tensor for Faster R-CNN
        target["image_id"] = torch.tensor(idx)

        #Bounding box areas used in COCO evaluation metrics
        target["area"] = torch.tensor(
            [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes_pixels],
            dtype=torch.float32
        )

        #iscrowd = 0 means each baseball is a distinct object
        target["iscrowd"] = torch.zeros(len(boxes_pixels), dtype=torch.int64)

        #returns image, target pair
        return image, target


#Load the video and annotation data
"""Load all 4 annotated videos into our custom dataset and split into training and validation sets.

A custom collate function is required because each frame can have a different number of baseballs.
Batches cannot be stacked into a single tensor like normal."""

VIDEO_DIR = "videos/"
XML_DIR = "annotations/"

video_xml_pairs = [
    (os.path.join(VIDEO_DIR, "IMG_8923_souleymane.mov"), os.path.join(XML_DIR, "IMG_8923_souleymane.xml")),
    (os.path.join(VIDEO_DIR, "IMG_8924_souleymane.mov"), os.path.join(XML_DIR, "IMG_8924_souleymane.xml")),
    (os.path.join(VIDEO_DIR, "IMG_8946_souleymane.mov"), os.path.join(XML_DIR, "IMG_8946_souleymane.xml")),
    (os.path.join(VIDEO_DIR, "IMG_8947_souleymane.mov"), os.path.join(XML_DIR, "IMG_8947_souleymane.xml")),
]

print("Loading dataset...\n")
full_dataset = BaseballVideoDataset(video_xml_pairs=video_xml_pairs, img_size=224)

#Split 80% for training, 20% for validation
training_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - training_size

training_dataset, validation_dataset = torch.utils.data.random_split(
    full_dataset, [training_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Training samples:   {len(training_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")

"""Faster R-CNN requires custom collate function since
each frame has a variable number of boudning boxes (baseballs)"""
def collate_fn(batch):
    return tuple(zip(*batch))

BATCH_SIZE = 2
training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#Print Confirmation Data is Loaded with Batch Size
print(f"DataLoaders ready with batch size {BATCH_SIZE}\n")


#Create Neural Network
""" We use Faster R-CNN pretrained on COCO: a large dataset of 91 object classes.

Rather than training from scratch (which would require far more data and compute), we use transfer learning:
    keep all the pretrained feature extraction layers
    and only replace the final classification head with one 
    that predicts our 2 classes (background + baseball)."""

def get_baseball_model(num_classes):
    """
    Loads pretrained Faster R-CNN and replaces the prediction
    head for our baseball detection task.

    num_classes = 2:
        class 0 = background
        class 1 = baseball
    """
    #Load Faster R-CNN pretrained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    #Get number of input features from the existing classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    #Replace the pretrained head with a new one for our 2 classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

NUM_CLASSES = 2
model = get_baseball_model(NUM_CLASSES)
#moves the model to the correct hardware (GPU / CPU)
model.to(device)

print(f"Model loaded. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

#Train the Neural Network
""" I used SGD optimizer as recommended for Faster R-CNN.

The learning rate scheduler reduces the learning rate (90%) every 3 epochs 
to help the model fine-tune more carefully as training progresses."""

LEARNING_RATE = 0.005
EPOCHS = 5

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005
)

#Reduce learning rate by 90% every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

#Free up memory before training
import gc
gc.collect()

print(f"\nTraining for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (images, targets) in enumerate(training_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #Faster R-CNN returns a dictionary of losses during training
        #including: classifier, box regression, objectness, and RPN losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if batch_idx % 10 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}] "
                  f"Batch [{batch_idx+1}/{len(training_loader)}] "
                  f"Loss: {losses.item():.4f}")

    lr_scheduler.step()
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Complete — Avg Loss: {total_loss/len(training_loader):.4f}\n")
    print("-" * 50)

#Confirmation of completed nueral network training step
print("Training complete!")

#STORES MODEL WEIGHTS
""" Save trained weights so the results can  be evaluated without having to retrain the model.
This stores the model weights, optimizer state, and number of epochs trained."""

MODEL_PATH = "baseball_model.pt"

torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_classes': NUM_CLASSES,
    'img_size': 224,
}, MODEL_PATH)

print(f"\nModel weights saved to: {MODEL_PATH}")

#Final Testing
#Load Model and run Inference
"""This final section demonstrates that the model can be loaded from saved weights 
and used to make predictions on a new video without any retraining."""

print("\nLoading saved model for inference...")

#Create blank model with the same architecture
loaded_model = get_baseball_model(NUM_CLASSES)

#Load the saved weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.to(device)

#Set to evaluation mode
loaded_model.eval()

print(f"Model loaded from {MODEL_PATH}")
print(f"Trained for {checkpoint['epoch']} epochs\n")

#Run inference on the first frame of a "new" video
    #this just confirms the custom loader can import a new video
"""NOTE: We are running inference on a video used during training to verify the model's predictions against known annotations.
Additional videos are available and will be used for held-out testing in the final project to properly evaluate generalization performance."""

SAMPLE_VIDEO = os.path.join(VIDEO_DIR, "IMG_8923_souleymane.mov")

print(f"Running inference on: {os.path.basename(SAMPLE_VIDEO)}")

cap = cv2.VideoCapture(SAMPLE_VIDEO)
ret, frame = cap.read()
cap.release()

#Prepare the frame exactly as we did during training
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (224, 224))
input_tensor = torch.tensor(frame_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0

#Run prediction. No gradient is needed for inference
with torch.no_grad():
    predictions = loaded_model([input_tensor.to(device)])

pred = predictions[0]
print(f"\nInference Results:")
print(f"  Baseballs detected: {len(pred['boxes'])}")
print(f"  Confidence scores:  {pred['scores'].tolist()}")
print(f"  Bounding boxes:     {pred['boxes'].tolist()}")
print(f"\nAll checks satisfied:")
print(f"  Custom loader imports videos         ✅")
print(f"  Neural network trained on baseball data ✅")
print(f"  Model weights saved to baseball_model.pt ✅")
print(f"  Model loaded and evaluated without retraining ✅")