# 🧠 Real-Time Cognitive Load Estimation via Wearable EEG  
_A Level-3 Interactive Simulation and Classification Prototype_

This project delivers a real-time cognitive load classification system using synthetic EEG data. It features a modular training pipeline, a lightweight 1D-CNN model, and a Streamlit-based dashboard for live simulation, visualization, and latency tracking. Designed for rapid prototyping and neuroadaptive interface research.

---

## 🛠️ Tech Stack

- Python 3.x  
- NumPy & SciPy for signal preprocessing  
- TensorFlow/Keras for 1D-CNN modeling  
- Streamlit for interactive dashboard  
- Pandas & Matplotlib for data handling and visualization  

---

## 🚀 Features

- 🎛️ Sliding-window EEG segmentation with bandpass and notch filtering  
- 🧠 1D-CNN classifier for low vs. high cognitive load detection  
- 📊 Real-time dashboard with live EEG plots and classification feedback  
- ⚙️ Adjustable parameters: window length, step size, refresh rate, noise level  
- 📈 Metrics: confidence tracking, inference latency, prediction probabilities  

---

<pre>
## 📂 Folder Structure
wearable_bci_simulator/ 
├── data/ # EEG CSV files (8-channel, labeled 'low' or 'high') 
├── models/ # Trained CNN model ('project3_cnn_model.keras')
├── train_cnn_eeg_classifier.py # Script to train and save the 1D-CNN model 
├── app.py # Streamlit dashboard for real-time simulation and classification 
├── README.md # Project overview and structure 
├── screenshot.pdf # Visual reference of folder layout and dashboard UI
</pre>

---

## 🧪 How It Works

- **Train Model** → Run `train_cnn_eeg_classifier.py` to train and save the CNN  
- **Launch Dashboard** → Use `app.py` to simulate real-time EEG streaming and classification  
- **Interact** → Adjust parameters, visualize signals, and monitor classification results live  

---

## 📌 Applications

- Wearable BCI prototyping  
- Neuroergonomic interface design  
- Cognitive workload monitoring  
- Educational demos in signal processing and machine learning  

---

## 📄 Documentation

- `Paper.docx`: Academic write-up of methodology and results  
- `screenshot.pdf`: Folder structure and UI overview  

---

## 🔓 License

This project is licensed under the MIT License. Free to use, modify, and extend.

---

## 👤 Author

**Keerthi Kumar K J**  
📧 inteegrus.research@gmail.com
