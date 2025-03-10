 Motion Detection and Tracking with OpenCV

 Project Description

This project implements "motion detection and tracking" using "Python" and "OpenCV". It detects moving objects in a video stream and tracks them in real-time.

 a.) Features

- Real-time Motion Detection
- Object Tracking
- YOLOv3 Model Integration
- Video Processing using OpenCV
- Frame Differencing & Contour Detection

Installation

1️⃣ Clone the Repository


git clone https://github.com/Naga2609/Motion-Detection-Tracking.git
cd Motion-Detection-Tracking


2️⃣ Install Dependencies


pip install -r requirements.txt


3️⃣ Download YOLOv3 Weights

Due to GitHub's 100MB file limit, **yolov3.weights** is stored separately. Download it from **Google Drive** and place it in the `models/` directory:

🔗 [Download yolov3.weights](https://drive.google.com/drive/u/0/folders/1Kwp_kZPeEmRyBy8-tSQ5Y69jQWj5QHUM)


mkdir models
mv yolov3.weights models/


📄 Usage

Run the following command to start motion detection and tracking:


python motion_detection.py


For real-time tracking using a webcam:


python motion_detection.py --webcam


📂 Project Structure


Motion-Detection-Tracking/
│── models/               # YOLOv3 model files
│   ├── yolov3.weights    # YOLOv3 weights file
│   ├── yolov3.cfg        # YOLOv3 configuration
│── videos/               # Sample input videos
│── output/               # Processed output videos
│── motion_detection.py   # Main motion detection script
│── requirements.txt      # Required Python dependencies
│── README.md             # Project documentation


✨ Contributing

Feel free to submit pull requests or open issues to improve this project.

📜 License

This project is open-source under the MIT License.

generate a ReadMe, file 

generate a README.md file.


