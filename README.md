# Sign Language Recognition System

A real-time sign language recognition system that uses computer vision and machine learning to detect and classify American Sign Language (ASL) alphabet letters from webcam input.

## 🚀 Features

- **Real-time Detection**: Live hand gesture recognition using webcam
- **Data Collection**: Automated data collection for training custom gesture models
- **Machine Learning**: Random Forest classifier for accurate gesture classification
- **Hand Tracking**: Advanced hand landmark detection using MediaPipe
- **Easy Training**: Simple scripts to collect and train on new gestures
- **High Accuracy**: Optimized model with configurable confidence thresholds

## 📋 Requirements

- Python 3.8+
- Webcam
- Windows/Linux/MacOS

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Saadhmmmandia/sign_language_project.git
   cd sign_language_project
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### 1. Data Collection
Collect training data for sign language gestures:

```bash
python collect_data.py
```

- Enter the letter (A-Z) when prompted
- Position your hand in front of the camera
- The system will capture hand landmarks automatically
- Press 'q' to stop collecting data for current letter

### 2. Train the Model
Train the machine learning model on collected data:

```bash
python train_model.py
```

- The script will load data from `dataset/hand_sign_data.csv`
- Train a Random Forest classifier
- Save the trained model to `models/sign_model.pkl`
- Display training accuracy and classification report

### 3. Real-time Prediction
Run live gesture recognition:

```bash
python predict.py
```

- Open webcam and show hand gestures
- The system will display predicted letters in real-time
- Confidence threshold set to 50% for reliable predictions
- Press 'q' to exit

## 📁 Project Structure

```
sign_language_project/
├── collect_data.py          # Data collection script
├── train_model.py           # Model training script
├── predict.py              # Real-time prediction script
├── requirements.txt        # Python dependencies
├── dataset/                # Training data directory
│   └── hand_sign_data.csv  # Collected landmark data
├── models/                 # Trained models directory
│   └── sign_model.pkl      # Saved Random Forest model
└── README.md              # Project documentation
```

## 🔧 How It Works

1. **Hand Detection**: Uses MediaPipe Hands to detect 21 hand landmarks
2. **Feature Extraction**: Captures x, y, z coordinates of each landmark
3. **Data Collection**: Stores landmark data with corresponding labels (A-Z)
4. **Model Training**: Random Forest classifier learns patterns in landmark positions
5. **Prediction**: Real-time classification of live hand gestures

## 🎯 Supported Gestures

Currently supports recognition of ASL alphabet letters A-Z. The system can be extended to recognize:
- Numbers (0-9)
- Common phrases
- Custom gestures

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Features**: 63 (21 landmarks × 3 coordinates)
- **Training**: 80% of collected data
- **Testing**: 20% of collected data
- **Accuracy**: Depends on quality and quantity of training data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [OpenCV](https://opencv.org/) for computer vision
- [scikit-learn](https://scikit-learn.org/) for machine learning
- [pandas](https://pandas.pydata.org/) for data manipulation

## 📞 Support

If you encounter any issues or have questions:
1. Check the requirements are properly installed
2. Ensure your webcam is working
3. Verify you have sufficient training data for each gesture
4. Open an issue on GitHub

---

**Happy signing! 🤟**