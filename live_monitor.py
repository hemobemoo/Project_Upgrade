# This is for generating live monitoring test data for ECG classification.
# The generated data will be saved as a .npy file for testing purposes.

import numpy as np

def generate_ecg_beat(fs=100, is_abnormal=False):
    if not is_abnormal:
        t = np.linspace(-0.2, 0.4, int(0.6 * fs))
        p_wave = 0.15 * np.exp(-((t + 0.12) / 0.025)**2)
        q_wave = -0.15 * np.exp(-((t + 0.03) / 0.01)**2)
        r_wave = 1.00 * np.exp(-((t - 0.00) / 0.015)**2)
        s_wave = -0.25 * np.exp(-((t - 0.03) / 0.012)**2)
        t_wave = 0.25 * np.exp(-((t - 0.18) / 0.05)**2)
        beat = p_wave + q_wave + r_wave + s_wave + t_wave
    else:
        t = np.linspace(-0.15, 0.25, int(0.4 * fs))
        r_wave = 1.2 * np.exp(-((t - 0.0) / 0.04)**2)
        s_wave = -0.6 * np.exp(-((t - 0.06) / 0.03)**2)
        t_wave = -0.3 * np.exp(-((t - 0.18) / 0.06)**2)
        beat = r_wave + s_wave + t_wave
    return beat

def generate_30s_12lead_windows(fs=100, duration_sec=30, abnormal_start_sec=21, window_len=1000):
    total_samples = fs * duration_sec  # 3000 samples total
    ecg_single = np.zeros(total_samples)
    
    # 0s to 21s: Normal Sinus Rhythm (~70 BPM)
    curr_sample = 0
    normal_beat = generate_ecg_beat(fs=fs, is_abnormal=False)
    normal_interval = int(0.85 * fs)
    
    while curr_sample < abnormal_start_sec * fs:
        end_idx = min(curr_sample + len(normal_beat), total_samples)
        beat_len = end_idx - curr_sample
        ecg_single[curr_sample:end_idx] += normal_beat[:beat_len]
        curr_sample += normal_interval
        
    # 21s to 30s: Abnormal Rhythm (~150 BPM)
    abnormal_beat = generate_ecg_beat(fs=fs, is_abnormal=True)
    abnormal_interval = int(0.40 * fs)
    
    while curr_sample < total_samples:
        end_idx = min(curr_sample + len(abnormal_beat), total_samples)
        beat_len = end_idx - curr_sample
        ecg_single[curr_sample:end_idx] += abnormal_beat[:beat_len]
        curr_sample += abnormal_interval
        
    # Expand to 12 leads -> shape (3000, 12)
    lead_gains = [1.0, 0.8, 0.6, -0.4, 1.2, 0.9, 0.5, 0.7, 1.1, 1.3, 1.0, 0.8]
    ecg_12lead = np.zeros((total_samples, 12))
    for lead_idx in range(12):
        noise = np.random.normal(0, 0.02, total_samples)
        ecg_12lead[:, lead_idx] = (ecg_single * lead_gains[lead_idx]) + noise

    # Split 30s stream into consecutive windows of shape (1, 1000, 12)
    # Window 0: 0s-10s (Normal)
    # Window 1: 10s-20s (Normal)
    # Window 2: 20s-30s (Abnormal triggers at 21s)
    num_windows = total_samples // window_len
    windows = []
    
    for w in range(num_windows):
        start = w * window_len
        end = start + window_len
        chunk = ecg_12lead[start:end, :]  # Shape: (1000, 12)
        chunk_batched = np.expand_dims(chunk, axis=0)  # Shape: (1, 1000, 12)
        windows.append(chunk_batched)
        
    return np.array(windows)

if __name__ == "__main__":
    # Generate array of shape (3, 1, 1000, 12)
    ecg_windows = generate_30s_12lead_windows()
    np.save("live_test_ecg_30s_batched.npy", ecg_windows)
    
    print(f"Total dataset shape: {ecg_windows.shape}")
    print(f"Single window shape: {ecg_windows[0].shape}")  # Outputs: (1, 1000, 12)