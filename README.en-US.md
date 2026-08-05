

Open Source Code for the 16th National College Student Smart Car Competition - Baidu Track
=============================================
Our team annotated nearly 4,000 images for the landmark and side tasks. The annotations were performed using LabelImg in VOC format.

The annotated landmark dataset has been shared on Aistudio: https://aistudio.baidu.com/aistudio/projectdetail/2170723 .

The annotated side dataset has been shared on Aistudio: https://aistudio.baidu.com/aistudio/projectdetail/2170794 .

The line-following dataset has been shared on Aistudio: https://aistudio.baidu.com/aistudio/datasetdetail/108767.

The main program for the car's operation is located in `src`.
------
Please run `control_collect_line_data.py` or `nocontrol_collect_line_data.py` within the `src` directory.

A necessary prerequisite for successfully completing all tasks is that the car can complete Task 1 — autonomous navigation, which serves as the foundation for executing all other tasks. Therefore, our team devoted the majority of our time and effort to Task 1. Our approach involves using deep learning to train a line-following model. Since model performance heavily depends on the dataset, the initial step of collecting lane data is crucial.

Traditionally, lane data is collected by using a joystick to manually drive the car. During operation, the onboard camera captures images. We used a normalization method to calculate the car's deviation angle from the center line, saved all angle values into a JSON file, and fed both the images and angle values into our neural network for training to generate a model. However, after the model was generated and deployed on the car, the autonomous cruising performance was suboptimal. Upon analysis, we found that the system was highly sensitive to lighting conditions. Especially under strong light sources, severe glare on the track caused large white areas to appear, making it impossible for the camera to discern track details. Consequently, the system could not calculate the deviation angle from the center line, leading to failure in the cruising task. Additionally, image blur captured by the camera during high-speed cruising could also lead to failure in the cruising task. Furthermore, during the manual data collection process, we observed that manipulating the joystick caused significant fluctuations in the returned steering angles. This led to severe oscillation after the model was trained, especially when navigating roundabouts or sharp turns.

To address these issues, our team optimized the aforementioned approach. For Problem 1, we installed a polarizing filter on the camera to reduce light reflection on the track. We also incorporated images collected by the car under various lighting intensities and speeds into the lane-following dataset, thereby enhancing the car's robustness during autonomous navigation. For Problem 2, our solution was to stop using manual remote control for data collection. Instead, we utilized color extraction to identify the track's center line, allowing the smart car to cruise along this line and automatically collect the dataset. The advantage of this method is that during straight-line driving, the angle values corresponding to consecutive frames do not fluctuate significantly. Additionally, turns are executed smoothly, and the car body experiences minimal jitter after navigating corners.

![HSV Color Extraction](https://user-images.githubusercontent.com/90136090/133626826-2b1f99d4-d0ce-465f-8fb0-2ef6cce038d1.jpg)

### HSV Color Extraction

Regarding Problem 2, to minimize the impact of light during image capture, we extracted colors in the HSV color space and used a PD control (without the I component) to determine the turning angle. The formula is as follows:

differ=(x×p+d×(x-last_x))/120

Where `differ` is the turning angle, `x` is the pixel difference between the center of the current image and the midpoint between the two yellow lines, and `last_x` is the same measurement for the previous frame. Through extensive experimentation, we determined that `p=1.05`, and `d=0.098`.

As a result, based on solid line-following performance, we ranked 7th in the Eastern Division and ultimately won the First Prize at the National Finals. During subsequent tuning, the car could complete the entire track in under 30 seconds at a speed of 70 (with a maximum speed of 100). After further task optimization, all tasks could be completed in under 1.5 minutes.
