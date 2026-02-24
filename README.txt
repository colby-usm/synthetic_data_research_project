Annotation Pipeline README

This pipeline describes how the user can recreate the dataset useed in our test split.

Environment Setup
1. Install Python 3.10 or higher.
2. Install OpenCV:
   pip install opencv-python

Download the Dataset
Download the military assets dataset from Kaggle:
https://www.kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format

Selecting Images for Annotation
1. Open image_selector.py.
2. Update the following in the script:
   - dataset_root: Path to the dataset you downloaded.
   - desired_classes: List of classes you want to include in your subset.
3. Run the script:
   - Use 'y' to select and add an image to your subset.
   - Use 'q' to quit the program.

Generating Referring Expressions
1. Open ref_writer.py and ensure all paths are correctly set to your dataset and output locations.
2. Run the script.
3. For each highlighted object, enter a referring expression via the CLI prompt.
