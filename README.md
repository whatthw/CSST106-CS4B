# CSST106-CS4B


https://github.com/user-attachments/assets/7edf1f21-5b42-4c83-9657-3517f9910ae9


# Intoduction to Computer Vision 
Computer Vision is a field of artificial intelligence that deals with teaching computers to understand and interpret visual information, such as images and videos. It's essentially giving machines the ability to "see" and make sense of the world around them.

Computer vision is a field of artificial intelligence (AI) that enables computers to interpret and understand visual information from the world, similar to how humans use their vision. It involves the development of algorithms and models that allow computers to process, analyze, and make decisions based on visual data, such as images and videos


# Role of Image Processing in Artificial intelligence

The term "image processing" describes techniques for enhancing, analyzing, and extracting information from digital images. Image segmentation, feature extraction, geometric modifications, image enhancement, image compression, and picture reconstruction are the primary tasks in image processing. With the help of these methods, we may enhance the quality of photos, eliminate noise and artifacts, extract useful data and metadata, and identify patterns and objects.

Scientific image analysis, medical imaging systems, industrial inspection systems, surveillance and security systems, photo editing software, and image-to-text converters are among the devices that use image processing techniques. Machine learning and deep learning models supplement traditional image-processing approaches with the advent of artificial intelligence.

## Key Tasks in Image Processing

### Image Segmentation

Objective- Divide an image into regions or segments that represent different objects or parts of the scene.
Techniques- Thresholding, clustering (e.g., K-means), region growing, and edge detection.
Applications- Identifying tumors in medical images, segmenting objects in autonomous driving.

### Feature Extraction

Objective- Extract meaningful attributes or characteristics from an image that can be used for further analysis.
Techniques- Edge detection (e.g., Canny edge detector), corner detection (e.g., Harris corner detector), texture analysis.
Applications- Face recognition, object tracking, and scene classification.

### Geometric Modifications

Objective- Transform images in terms of scaling, rotation, or warping to correct distortions or align them for further processing.
Techniques- Affine transformations, perspective corrections, and image registration.
Applications- Aligning satellite images, correcting lens distortions.

### Image Enhancement
Objective- Improve the visual quality or contrast of an image to make it more suitable for analysis.
Techniques- Histogram equalization, noise reduction (e.g., Gaussian filter), sharpening (e.g., unsharp mask).
Applications- Enhancing medical scans, improving visibility in low-light conditions.

### Image Compression

Objective- Reduce the size of an image file while preserving as much information as possible.
Techniques- Lossy compression (e.g., JPEG), lossless compression (e.g., PNG).
Applications- Reducing storage requirements for images, speeding up image transmission.

### Image Reconstruction

Objective- Restore or reconstruct images from incomplete, noisy, or distorted data.
Techniques- Inpainting, super-resolution, and interpolation.
Applications- Repairing damaged images, enhancing resolution of old photographs.
Applications of Image Processing

### Scientific Image Analysis

Objective- Analyze images for scientific research and experimentation.
Examples- Analyzing astronomical images to study celestial bodies, examining microscopic images in biology.

### Medical Imaging Systems

Objective- Aid in diagnosing and treating medical conditions through imaging techniques.
Examples- MRI, CT scans, and X-ray images to detect and monitor diseases.

### Industrial Inspection Systems

Objective- Inspect and ensure the quality of products in manufacturing processes.
Examples- Automated inspection of electronic components, detecting defects in manufacturing.

### Surveillance and Security Systems

Objective- Enhance security by monitoring and analyzing video feeds.
Examples- Facial recognition for access control, and detecting unusual behavior in surveillance footage.

### Photo Editing Software

Objective- Provide tools for modifying and enhancing images.
Examples- Adjust brightness, contrast, color balance, and applying filters in applications like Photoshop.

### Image-to-Text Conversion

Objective- Extract and interpret text from images.
Examples- Optical Character Recognition (OCR) for digitizing printed documents, and reading text from scanned forms.

### Integration with Machine Learning and Deep Learning

#### Machine Learning: Machine learning algorithms are enhancing Traditional image processing methods that can automatically learn from data and improve performance over time. For example, machine learning models can classify images or detect objects with high accuracy.

#### Deep Learning: Deep learning, a subset of machine learning, utilizes neural networks with many layers (e.g., Convolutional Neural Networks, or CNNs) to perform tasks such as image recognition and segmentation. Deep learning models can automatically extract features and learn complex patterns from large datasets, making them powerful tools in image processing.

# Types of Image Processing Techniques
**Filtering** - The pixels in an image are directly subjected to the filtering technique. Generally speaking, a mask is added in size to have a certain center pixel. The mask is positioned on the image so that its center crosses every pixel in the image.

![Screenshot 2024-09-05 233711](https://github.com/user-attachments/assets/a6b9d7be-083b-4fe4-828c-0bd70a529d81)

**Segmentation** - In order to assist with object detection and related tasks, image segmentation is a computer vision approach that divides a digital image into distinct groupings of pixels, or image segments.

![Screenshot 2024-09-05 233817](https://github.com/user-attachments/assets/07a04720-0036-4d01-aaa1-97ca6f3affee)

**Edge Detection** - An essential method for locating and recognizing the borders or edges of objects in an image is edge detection. It is employed to extract the outlines of objects that are present in an image as well as to recognize and detect discontinuities in the image intensity.

![Screenshot 2024-09-05 233916](https://github.com/user-attachments/assets/2ea78072-ae40-48f6-99c7-e7d70353a264)


# Case Study Selection: AI Application: Autonomous Vehicles

Autonomous vehicles, or self-driving cars, use a variety of AI technologies to navigate and operate safely without human intervention. Here's a brief overview of the key AI applications in autonomous vehicles:

Autonomous vehicles leverage several key techniques and technologies to achieve safe and efficient self-driving. Here are some of the most important ones:

**1. Computer Vision**
Object Detection and Recognition- Uses algorithms like YOLO (You Only Look Once) and SSD (Single Shot MultiBox Detector) to identify and classify objects in the environment.
Semantic Segmentation- Employs models like DeepLab and U-Net to classify each pixel in an image, helping to understand the scene's context.

**2. Sensor Fusion**
Data Integration- Combines data from multiple sensors (cameras, LiDAR, radar) to create a comprehensive and accurate view of the surroundings.
Kalman Filters- Used for estimating the state of dynamic systems by combining noisy sensor measurements with predictions.

**3. Localization**
Global Positioning System (GPS)- Provides real-time location data.
Simultaneous Localization and Mapping (SLAM)- Builds and updates maps of an environment while tracking the vehicle’s position within it.

**4. Path Planning and Decision-Making**
A and Dijkstra’s Algorithms Commonly used for finding the shortest path in known environments.
Rapidly-exploring Random Trees (RRT)- Used for path planning in complex, dynamic environments.
Model Predictive Control (MPC)- Optimizes vehicle control actions by predicting future states based on current conditions.

**5. Control Systems**
PID Controllers- Proportional-Integral-Derivative controllers manage the vehicle’s speed, steering, and braking for smooth driving.
Adaptive Cruise Control- Adjusts the vehicle’s speed based on the distance from the car in front.

**6. Machine Learning and Deep Learning**
Convolutional Neural Networks (CNNs)- Used for image recognition and feature extraction from camera feeds.
Reinforcement Learning- Helps in training autonomous vehicles to make decisions by simulating different scenarios and learning from them.

**7. Radar and LiDAR**
Radar- Measures the speed and distance of objects using radio waves. Effective in various weather conditions.
LiDAR- Uses laser pulses to create high-resolution 3D maps of the environment, allowing precise distance measurements.

**8. High-Definition Maps**
Detailed Mapping- Provides detailed road information such as lane markings, road signs, and intersections. These maps are often updated with real-time data.

**9. Simulation and Testing**
Virtual Environments- Use simulations to test and validate autonomous driving algorithms in controlled and diverse scenarios.
Hardware-in-the-Loop (HIL) Testing- Combines real hardware with simulation to test systems under realistic conditions.

**10. Ethical and Safety Frameworks**
Decision-Making Protocols- Develop frameworks to ensure the vehicle makes safe and ethical decisions in complex situations.
Fail-Safe Mechanisms- Implement backup systems and safety protocols to handle potential failures or emergencies.

**Implementation Creation**

Edge detection is crucial in autonomous vehicles for tasks such as lane detection, obstacle recognition, and road sign identification. It helps in understanding the structure and boundaries within the vehicle’s environment, which is essential for navigation and safety.
Edge Detection in Autonomous Vehicles

cv2 (OpenCV): Provides powerful functions for image processing and computer vision, including edge detection.
matplotlib.pyplot (imported as plt): Used for visualizing processed images and results, such as plotting the detected edges.
google.colab.patches (cv2_imshow): An optional library for displaying images in Google Colab notebooks (not essential for core functionality but useful for visualization in Colab).

###**Edge Detection Using OpenCV's Canny Algorithm**

Edge detection is crucial for tasks like lane detection and obstacle recognition in autonomous vehicles. The following Python code demonstrates how to apply Canny edge detection to an image using OpenCV, with added noise reduction through Gaussian blur.


```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
```
## import Library 

###Load the image
```
img = cv2.imread("city-road.jpg")  # Replace with your image path
```


### Convert the image to grayscale
```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
### This simplifies the image by removing color information, making it easier to process

### Apply Gaussian blur to reduce noise
```
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```
### The blur helps to smooth out the image and reduce noise, which improves edge detection

### Detect edges using Canny edge detection
```
edges = cv2.Canny(blurred, 100, 200)
```
### The Canny algorithm detects edges by looking for areas of rapid intensity change
### The thresholds (100, 200) are used to identify strong and weak edges

```
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Detected Image'), plt.xticks([]), plt.yticks([])
plt.show()
```
### Display the original image and the edge-detected image using matplotlib
![Screenshot 2024-09-06 001723](https://github.com/user-attachments/assets/dedd61f6-db68-460d-b15b-a8785cd67744)

** Process Description **

Process Description:
Import Libraries

cv2: OpenCV library for image processing.
numpy: For numerical operations (not used in this snippet but typically imported alongside OpenCV).
pyplot from matplotlib: For creating visualizations.
cv2_imshow from google.colab.patches: For displaying images in Google Colab (optional).
Load the Image:

cv2.imread("city-road.jpg"): Reads the image file into memory. Make sure to replace "city-road.jpg" with the actual path to your image.
Convert to Grayscale:

cv2.cvtColor(img, cv2.COLOR_BGR2GRAY): Converts the image from BGR color space to grayscale. Grayscale simplifies the image, making edge detection more effective.
Apply Gaussian Blur:

cv2.GaussianBlur(gray, (5, 5), 0): Applies Gaussian blur to the grayscale image to reduce noise. The kernel size (5, 5) defines the size of the blur filter, and 0 means the standard deviation is calculated from the kernel size.
Detect Edges:

cv2.Canny(blurred, 100, 200): Uses the Canny edge detection algorithm to identify edges in the image. The thresholds 100 and 200 determine which gradients are considered edges.
Display Images:

Uses matplotlib to plot the original image and the edge-detected image side by side.
cv2.cvtColor(img, cv2.COLOR_BGR2RGB): Converts the original image from BGR to RGB for correct color display in matplotlib.
plt.imshow(edges, cmap='gray'): Displays the edge-detected image in grayscale.

# Diagram / Flowchart
![Screenshot 2024-09-05 215021](https://github.com/user-attachments/assets/40f62bd0-6ac5-4598-ba54-6f76d0bd5815)

# Conclusion
Artificial intelligence (AI) requires image processing, particularly in areas like computer vision, robotics, and medical imaging. In order to extract relevant information, digital images must be manipulated and analyzed. AI systems can better comprehend and interact with the visual environment by creating effective image processing algorithms, which will promote various kinds of industries and applications.

In this activity, I have learned different things about computer vision, especially in image processing. The image processing techniques help the image to enhance its quality, extract its important features, and analyze it.

# Emerging Form of Image Processing 
** Edge Computing **
Edge Computing: Processing images directly on edge devices (e.g., smartphones, drones) to reduce latency and bandwidth usage, enabling real-time applications like augmented reality (AR) and autonomous vehicles

### Benefits of Edge Computing 
Edge computing offers several benefits when processing images directly on edge devices like smartphones and drones, especially for applications such as augmented reality (AR) and autonomous vehicles. Here are the key advantages:

**1. Reduced Latency:**
Real-Time Processing: By processing data locally on the edge device, you minimize the time required for data to travel to and from a central server. This results in faster response times, which is crucial for real-time applications like AR and autonomous vehicles where split-second decisions are needed.

**2. Lower Bandwidth Usage:**
Data Transmission: Local processing reduces the amount of data that needs to be transmitted over networks, as only relevant or processed information is sent to central servers or other devices. This is particularly beneficial for bandwidth-constrained environments and reduces network congestion.

**3. Enhanced Privacy and Security:**
Data Security: Processing sensitive information locally helps to keep data private and secure, as it avoids transmitting potentially sensitive data over networks. This is important for applications involving personal information or security-sensitive operations.

**4. Improved Reliability and Resilience:**
Offline Functionality: Edge devices can continue to operate and make decisions even if they lose connection to central servers or networks, ensuring that critical functions are maintained without relying on continuous cloud connectivity.

**5. Cost Efficiency:**
Reduced Cloud Costs: Local processing can decrease the need for extensive cloud computing resources, leading to cost savings in data storage and processing. It also minimizes data transfer costs associated with cloud services.

**6. Scalability:**
Distributed Processing: By distributing processing tasks across many edge devices, you can handle larger volumes of data and scale applications more effectively without overloading central servers.

**7. Enhanced User Experience:**
Responsive Applications: Applications that rely on edge computing can offer smoother and more responsive experiences, as processing delays and network latency are minimized. This is crucial for immersive AR experiences and real-time decision-making in autonomous vehicles.

**8. Optimized Resource Utilization:**
Efficient Processing: Edge devices are often optimized for specific tasks, such as image processing or sensor data analysis, leading to more efficient use of computational resources compared to general-purpose cloud servers.

**9. Localized Data Analytics:**
Context-Aware Processing: Edge devices can perform localized data analytics, making decisions based on immediate context and environment, which is beneficial for adaptive and intelligent systems like autonomous driving and AR applications.

**10. Scalable IoT Solutions:**
Integration with IoT: Edge computing is well-suited for Internet of Things (IoT) applications, where numerous devices collect and process data locally, leading to more efficient and scalable IoT ecosystems.
