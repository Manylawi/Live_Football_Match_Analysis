# ⚽ **Football Analysis AI System**

### *Player Tracking, Ball Possession, Speed Estimation, and Tactical Insights Using YOLO + CV + ML*

![Demo](output_videos/output_video.mp4)

---

## 🚀 **Overview**

This project performs **full football match analysis** using computer vision and machine learning.
It detects and tracks players, referees, and the ball using **YOLO**, assigns players to teams using **K-Means clustering**, estimates ball possession, measures player movement, and calculates real-world speed and distance using perspective transformation and camera movement compensation.

This system is implemented with **Streamlit**, making it fully interactive and easy to use in a browser.

---

Sure! Here is a **Team section with exactly 6 names only** (no roles, no contributions):

---

## 👥 **Team**

* Ahmed Mohamed Ahmed Fouad El-Manylawi
* Ali Ahmed
* Hazem Ashraf
* Ahmed Said
* Ahmed Reda
* Ahmed Ali

---

## 🧠 **Key Features**

### ✔ **Player, Referee, and Ball Detection**

Powered by a custom-trained **YOLO model** (`models/best.pt`).

### ✔ **Multi-object Tracking**

Track players consistently across frames with unique IDs.

### ✔ **Team Classification (K-Means)**

Automatically identifies team colors from jersey pixels.

### ✔ **Ball Possession Estimation**

Detects which team controls the ball frame-by-frame.

### ✔ **Camera Motion Compensation (Optical Flow)**

Ensures movement/speed estimates remain accurate even with camera panning.

### ✔ **Perspective Transformation (Bird’s-Eye View)**

Maps pixel coordinates → actual field meters.

### ✔ **Player Speed & Distance Calculation**

Shows each player's speed (km/h) and total distance run.

### ✔ **Video Rendering & Visualization**

Outputs a fully annotated video with overlays for teams, camera motion, ball ownership, and player stats.

---

# 📦 **Project Structure**

```
📂 Football-Analysis-AI
│── main.py
│── utils.py
│── trackers.py
│── team_assigner.py
│── player_ball_assigner.py
│── view_transformer.py
│── speed_and_distance_estimator.py
│── camera_movement_estimator/
│     └── camera_movement_estimator.py
│── models/
│     └── best.pt
│── stubs/
│── output_videos/
│── README.md
```

---

# 🧩 **Models & Libraries Used**

| Component                      | Purpose                                      |
| ------------------------------ | -------------------------------------------- |
| **YOLO (Ultralytics)**         | Player, referee, and ball detection          |
| **OpenCV**                     | Video I/O, drawing, resizing, optical flow   |
| **NumPy**                      | Math, tracking data, coordinate processing   |
| **Streamlit**                  | Web interface, video upload, results display |
| **Pickle**                     | Caching tracking & camera movement stubs     |
| **Tracker Module**             | YOLO detection + ID tracking + annotation    |
| **TeamAssigner**               | K-means clustering of jersey colors          |
| **PlayerBallAssigner**         | Estimate which player has the ball           |
| **CameraMovementEstimator**    | Optical-flow-based motion compensation       |
| **ViewTransformer**            | Pixel → meter conversion (homography)        |
| **SpeedAndDistance_Estimator** | Player movement & speed calculation          |

---

# 🛠️ **Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/your-username/football-analysis-ai.git
cd football-analysis-ai
```

### **2. Create a Python environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. Add your YOLO model**

Place your custom trained YOLO file here:

```
models/best.pt
```

---

# ▶️ **Usage**

### **Run the Streamlit App**

```bash
streamlit run main.py
```

### **Steps inside the Web App**

1. Upload a match video
2. Click **Process Video**
3. Wait while the AI performs:

   * Detection
   * Tracking
   * Team assignment
   * Camera compensation
   * Speed & distance measurements
4. Watch or download the fully analyzed output video

---

# 📊 **How It Works — Processing Pipeline**

### **1. Read video frames**

Using OpenCV.

### **2. Detect players/referees/ball using YOLO**

Custom model loaded by `Tracker`.

### **3. Multi-object tracking**

Adds consistent IDs and paths.

### **4. Camera motion estimation**

Optical flow → frame-by-frame displacement.

### **5. Perspective transformation**

Maps 2D pixels → real-world meters.

### **6. Team classification**

K-means clustering on jersey colors.

### **7. Ball possession estimation**

Assigns ball to nearest player per frame.

### **8. Speed & distance calculation**

Combines movement + real-world mapping.

### **9. Render output video**

Overlays:

* IDs
* Teams
* Ball control
* Camera arrows
* Speed & distance

---

# 🎥 **Sample Resources**

* **Trained YOLOv5 Model**
  👉 [https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

* **Sample Input Video**
  👉 [https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

---

# 🧪 **Outputs**

The system produces:
✔ Annotated video
✔ Player speed overlays
✔ Team ball possession timeline
✔ Camera movement visualization
✔ Real-world position mapping

---

# 📄 **License**

This project is open-source under the **MIT License**.

---

# 🤝 **Contributing**

Pull requests are welcome!
Feel free to add:

* More analytics
* Heatmaps
* Tactical metrics
* Passing detection

---

# 📬 **Contact**

For questions, improvements, or collaborations:

**Ahmed El-Manylawi**
📧 [ahmed.elmanylawi@gmail.com](mailto:ahmed.elmanylawi@gmail.com)
🔗 LinkedIn: linkedin.com/in/ahmed-el-manylawi-67b6162aa
