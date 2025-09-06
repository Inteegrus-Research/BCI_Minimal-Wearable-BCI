# app.py — Wearable BCI Simulation with Enhanced Layout
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from scipy.signal import butter, lfilter, iirnotch

# === Configuration ===
DATA_DIR = 'data'
MODEL_PATH = 'models/project3_cnn_model.keras'
CHANNELS = 8
FS = 256  # Hz

# Precompute filters for efficiency
BP_B, BP_A = butter(4, [1/(0.5*FS), 45/(0.5*FS)], btype='band')
NOTCH_B, NOTCH_A = iirnotch(50/(0.5*FS), 30)

# === Streamlit UI ===
st.set_page_config(page_title="Wearable BCI Simulator", layout="wide")
st.title("🧠 Real-time Cognitive Load Classifier")

# === Parameter Configuration Section ===
with st.sidebar:
    st.header("⚙️ Configuration Parameters")
    window_sec = st.select_slider("Window length (s)", [1, 2, 3], value=2)
    step_sec = st.select_slider("Step interval (s)", [0.1, 0.5, 1.0], value=0.5)
    noise_sigma = st.slider("Sensor Noise (σ)", 0.0, 1.0, 0.2, 0.05)
    st.divider()
    
    # Auto-refresh control
    refresh_rate = st.slider("Refresh rate (Hz)", 1, 20, 2, 1)
    st.caption(f"Effective update: {1/refresh_rate:.1f}s")
    
    st.divider()
    if st.button("🔄 Reset Simulation"):
        st.session_state.clear()

# === Initialize session state ===
def init_state():
    return {
        'run_demo': False,
        'file_idx': 0,
        'time_ptr': 0,
        'pred_history': [],
        'buffer': None,
        'data_full': None,
        'files': sorted([
            os.path.join(DATA_DIR, f) 
            for f in os.listdir(DATA_DIR) 
            if f.lower().endswith('.csv')
        ])
    }

if 'app' not in st.session_state:
    st.session_state.app = init_state()

# === Load model with caching ===
@st.cache_resource(show_spinner="Loading neural network...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# === Signal processing ===
def preprocess(signal):
    """Apply optimized filtering to EEG signal"""
    s = lfilter(BP_B, BP_A, signal)
    s = lfilter(NOTCH_B, NOTCH_A, s)
    return s - np.mean(s)

# === File selection ===
if st.session_state.app['files']:
    selected_file = st.selectbox(
        "Select EEG data file", 
        [os.path.basename(f) for f in st.session_state.app['files']],
        index=0
    )
    file_idx = [os.path.basename(f) for f in st.session_state.app['files']].index(selected_file)
else:
    st.error("No EEG data files found in 'data' directory!")
    st.stop()

# === Simulation Control Buttons ===
control_col1, control_col2, _ = st.columns([1,1,3])
start_btn = control_col1.button("▶️ Start Simulation", use_container_width=True)
stop_btn = control_col2.button("⏹️ Stop Simulation", use_container_width=True)

if start_btn:
    st.session_state.app = {
        'running': True,
        'file_idx': file_idx,
        'time_ptr': 0,
        'pred_history': [],
        'buffer': np.zeros((int(window_sec * FS), CHANNELS)),
        'data_full': pd.read_csv(
            st.session_state.app['files'][file_idx], index_col=0
        ).values.T,
        'files': st.session_state.app['files']
    }
    
if stop_btn:
    st.session_state.app['running'] = False

# === Display Containers ===
status_area = st.empty()

# Create layout with tabs for better organization
tab1, tab2 = st.tabs(["Real-time Visualization", "Classification Results"])

with tab1:
    st.subheader("EEG Signals")
    signal_plot = st.empty()
    
    st.subheader("Confidence History")
    conf_plot = st.empty()

# === Simulation Loop ===
if 'app' not in st.session_state:
    st.session_state.app = init_state()

if st.session_state.app.get('running', False):
    app = st.session_state.app
    win_len = int(window_sec * FS)
    step = int(step_sec * FS)
    
    # Main simulation loop
    while app['running']:
        start_time = time.perf_counter()
        
        # === Data Processing ===
        ptr = app['time_ptr']
        dfull = app['data_full']
        
        # Extract segment with noise
        seg = dfull[:, ptr:ptr+win_len].copy()
        seg += np.random.normal(0, noise_sigma, seg.shape)
        
        # Advance pointer and handle file transitions
        ptr += step
        if ptr + win_len > dfull.shape[1]:
            app['file_idx'] = (app['file_idx'] + 1) % len(app['files'])
            dfull = pd.read_csv(app['files'][app['file_idx']], index_col=0).values.T
            ptr = 0
        
        # Update state
        app['time_ptr'] = ptr
        app['data_full'] = dfull
        
        # Process and update buffer
        for ch in range(CHANNELS):
            # Shift buffer and add new processed data
            app['buffer'] = np.roll(app['buffer'], -step, axis=0)
            app['buffer'][-step:, ch] = preprocess(seg[ch])[-step:]
        
        # === Model Inference ===
        inp = app['buffer'][np.newaxis, ...]  # Add batch dimension
        pred = model.predict(inp, verbose=0)[0]
        cls = np.argmax(pred)
        label = "😌 Low Cognitive Load" if cls == 0 else "😫 High Cognitive Load"
        conf = float(pred[cls])
        app['pred_history'].append(conf)
        latency = (time.perf_counter() - start_time) * 1000
        
        # === Update Display ===
        status_area.success(f"Simulation running | File: {os.path.basename(app['files'][app['file_idx']])}")
        
        # EEG Signal Plot
        with signal_plot:
            st.line_chart(pd.DataFrame(
                app['buffer'], 
                columns=[f"Channel {i}" for i in range(CHANNELS)]
            ))
        
        # Confidence History Plot
        with conf_plot:
            hist_df = pd.DataFrame({
                'Confidence': app['pred_history'][-50:],
                'Threshold': [0.7]*min(50, len(app['pred_history']))
            })
            st.area_chart(hist_df)
        
        # Classification Results (in second tab)
        with tab2:
            st.subheader("Current Classification")
            
            # Enhanced state display with medium font size
            if cls == 0:
                st.markdown(f"<h2 style='text-align: center; color: green;'>LOW COGNITIVE LOAD</h2>", 
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='text-align: center; color: red;'>HIGH COGNITIVE LOAD</h2>", 
                            unsafe_allow_html=True)
            
            # Confidence display with progress bar
            st.markdown(f"<h3 style='text-align: center;'>Confidence: {conf:.2f}</h3>", 
                        unsafe_allow_html=True)
            st.progress(conf, text="Classification Confidence")
            
            # Metrics in columns
            col1, col2 = st.columns(2)
            col1.metric("Data File", os.path.basename(app['files'][app['file_idx']]))
            col2.metric("Inference Latency", f"{latency:.1f} ms")
            
            # Prediction distribution
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': ['Low Load', 'High Load'],
                'Probability': pred
            })
            st.bar_chart(prob_df, x='Class', y='Probability')
        
        # Control loop timing
        elapsed = time.perf_counter() - start_time
        sleep_time = max(0, step_sec - elapsed)
        time.sleep(sleep_time)
        
        # Check if stop requested
        if stop_btn:  # Will be updated on next script run
            app['running'] = False
            st.session_state.app['running'] = False
            st.rerun()
            
else:
    status_area.info("Configure parameters and press 'Start Simulation'")
    
    # Show sample data visualization
    if st.session_state.app['files']:
        sample_data = pd.read_csv(
            st.session_state.app['files'][file_idx], index_col=0
        ).iloc[:FS*5]  # First 5 seconds
        
        with tab1:
            st.subheader("Sample EEG Data (First 5 seconds)")
            st.line_chart(sample_data)
            
        with conf_plot:
            st.subheader("Simulation will display confidence history here")
        
        with tab2:
            st.subheader("Classification Results Will Appear Here")
            st.info("Start the simulation to see real-time classification results")