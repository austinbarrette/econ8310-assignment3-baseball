# Assignment 3
## Econ 8310 - Business Forecasting

For homework assignment 3, you will work with our baseball pitch data (available in Canvas).

- You must create a custom data loader as described in the first week of neural network lectures to load the baseball videos [2 points]
- You must create a working and trained neural network (any network focused on the baseball pitch videos will do) using only pytorch [2 points]
- You must store your weights and create an import script so that I can evaluate your model without training it [2 points]

Submit your forked repository URL on Canvas! :) I'll be manually grading this assignment.

Some checks you can make on your own:
- Can your custom loader import a new video or set of videos?
- Does your script train a neural network on the assigned data?
- Did your script save your model?
- Do you have separate code to import your model for use after training?
- 
## Solution
**Austin Barrette**

I fine-tuned a Faster R-CNN model pretrained on COCO to detect baseballs 
in baseball training videos. Rather than training from scratch, I used 
transfer learning — keeping all pretrained feature extraction layers and 
replacing only the final classification head for the 2 classes 
(background + baseball).

### Approach
- Parsed CVAT XML annotations to extract bounding box coordinates for 
  every baseball visible in each frame
- Built a custom PyTorch Dataset class that loads video frames with 
  OpenCV and pairs them with their annotations
- Fine-tuned Faster R-CNN across 5 epochs using SGD optimizer
- Saved trained weights to baseball_model.pt for evaluation without retraining

### Repository Structure
- `assignment_script.py` — loads data, trains model, saves weights
- `model_import.py` — loads saved weights and runs inference without retraining
- `requirements.txt` — package dependencies
- `videos/` — baseball training videos
- `annotations/` — XML annotation files

### Setup
1. Install dependencies:
    pip install -r requirements.txt

2. Place video files (.mov) in the videos/ folder
3. Place XML annotation files in the annotations/ folder

### Running
Train the model:

    python assignment_script.py

Evaluate without retraining:

    python model_import.py

### Grader Checks
- Custom loader imports new videos ✅
- Neural network trained on baseball data ✅
- Model weights saved to baseball_model.pt ✅
- Separate import script evaluates without retraining ✅