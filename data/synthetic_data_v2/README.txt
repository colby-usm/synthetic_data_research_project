SETUP:

1. install python >= 3.12
2. install opencv python package with your package manager install: opencv-python


CRITERIA FOR ANNOTATIONS:
1. Grammar:
a. write a sample phrase between 3-10 words for each phrase
b. write all phrases in lowercase.  Use no punctuations

2. Describing the Object:
a. describe ONLY the object with the GREEN bouding box with respect to context around it.
b. The class names are military_truck and military_tank. Describe each respetive class as "military truck" or "tank".  Note that "military" is not required for tank, but it is for truck.

3. Phrases:
a. Color
b. Size
c. Orientation of the object
d. charcteristics of each object


Examples:
the military truck behind the blue motorbike
the large tank in the street
the military truck in the rear of the traffic
the small military truck in the center of the image




HOW TO USE:
1. navigate to synthetic_data_v2/
2. run python ref_expr_writer.py --annotator <annotator_name>
2.a the first image should appear
3. type the annotation for the object with the green bounding box in the terminal and press 'Enter'
3.a the referring expression annotations will be refexps.json
4. Press 'q' to quit early


HOW TO RESUME YOUR WORK
1. run python3 ref_expr_writer.py --annotator <annotator_name> --image <image_id>

BUGS in the script: Sometimes a black image may appear - just press enter in the terminal and this will skip.
