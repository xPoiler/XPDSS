import numpy as np
import sounddevice as sd
import time
from scipy.signal import lfilter


def generate_log_sweep(duration=3.0, sample_rate=48000, f1=20.0, f2=20000.0):
    """
    Generates a logarithmic sine sweep.
    """
    N = int(duration * sample_rate)
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Log Sweep formula
    p_num = np.log(f2 / f1)
    phi = 2 * np.pi * f1 * duration / p_num * (np.exp(t * p_num / duration) - 1.0)
    sweep = np.sin(phi).astype(np.float32)
    
    # Apply cosine fade-in/out to avoid speaker pops
    fade_samples = int(sample_rate * 0.05) # 50ms
    fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples))**2
    fade_out = np.sin(np.linspace(np.pi/2, 0, fade_samples))**2
    
    sweep[:fade_samples] *= fade_in
    sweep[-fade_samples:] *= fade_out
    
    return sweep


def deconvolve(recorded, played, eps=1e-5):
    """
    Recovers the Impulse Response (h) using regularized division in the frequency domain.
    h = ifft( R(f) / S(f) )
    """
    N_total = len(played) * 2 # Zero-pad to avoid cyclic convolution wrap
    
    R = np.fft.rfft(recorded, n=N_total)
    S = np.fft.rfft(played, n=N_total)
    
    # Wiener deconvolution / Regularized Division
    S_conj = np.conj(S)
    denom = S * S_conj + eps # Water-level regularization
    
    H = R * S_conj / denom
    h = np.fft.irfft(H, n=N_total)
    
    return h[:len(played)] # Return back to original duration size


def measure_speaker(output_device_id, input_device_id, sample_rate, speaker_index, max_out_ch, max_in_ch, duration=3.0):
    """
    Plays a log-sweep out of speaker_index and records from input_device_id.
    """
    sweep = generate_log_sweep(duration, sample_rate)
    
    # Create multi-channel output buffer
    out_buf = np.zeros((len(sweep), max_out_ch), dtype=np.float32)
    out_buf[:, speaker_index] = sweep # Play only on one speaker
    
    recorded_data = []
    
    # We use a combined stream for exact sample-sync playing & recording
    def callback(indata, outdata, frames, time_info, status):
        nonlocal recorded_data
        if status:
            print(f"Calibration Callback Status: {status}")
        avail = len(out_buf) - callback.ptr
        if avail > 0:
            count = min(frames, avail)
            outdata[:count] = out_buf[callback.ptr:callback.ptr+count]
            if count < frames:
                outdata[count:] = 0 # Zero-pad tail
        else:
            outdata[:] = 0
        recorded_data.append(indata.copy())
        callback.ptr += frames
        
    callback.ptr = 0
    
    stream = sd.Stream(
        device=(input_device_id, output_device_id),
        samplerate=sample_rate,
        blocksize=1024,
        dtype='float32',
        channels=(max_in_ch, max_out_ch),
        callback=callback
    )
    
    with stream:
        while callback.ptr < len(sweep) - 1024:
            sd.sleep(100)
            
    # Concatenate the recorded buffers
    recorded = np.concatenate(recorded_data, axis=0)
    recorded = recorded[:len(sweep)] # Truncate to match exact played length
    
    return recorded, sweep


def calculate_transaural_params(L_FL, R_FL, L_FR, R_FR, sweep):
    """
    Analyzes Left/Right Ear data from sequential measurements.
    
    Returns:
    - itd_samples: The averaged interaural time difference (Samples).
    - crosstalk_scalar: The averaged attenuation factor k.
    """
    # 1. Deconvolve to recover Impulse Responses (h)
    h_L_FL = deconvolve(L_FL, sweep)
    h_R_FL = deconvolve(R_FL, sweep)
    h_L_FR = deconvolve(L_FR, sweep)
    h_R_FR = deconvolve(R_FR, sweep)
    
    # 2. Find exact peak arrival within plausible direct-sound window
    search_window = int(0.15 * 48000) 
    max_head = 40 # Max ITD samples (~0.8ms)
    
    # FL Speaker: Sound hits Left ear first
    peak_L_FL = np.argmax(np.abs(h_L_FL[:search_window]))
    # Restrict Right ear search to plausible head radius arrival times
    r_start_FL = max(0, peak_L_FL - max_head)
    r_end_FL = min(search_window, peak_L_FL + max_head)
    peak_R_FL = r_start_FL + np.argmax(np.abs(h_R_FL[r_start_FL:r_end_FL]))
    
    # FR Speaker: Sound hits Right ear first 
    peak_R_FR = np.argmax(np.abs(h_R_FR[:search_window]))
    # Restrict Left ear search
    l_start_FR = max(0, peak_R_FR - max_head)
    l_end_FR = min(search_window, peak_R_FR + max_head)
    peak_L_FR = l_start_FR + np.argmax(np.abs(h_L_FR[l_start_FR:l_end_FR]))
    
    # ITD calculations:
    itd_FL = peak_R_FL - peak_L_FL
    itd_FR = peak_L_FR - peak_R_FR
    
    # Average ITD (Algebraic average cancels out off-center sitting bias)
    itd_samples = int((itd_FL + itd_FR) / 2)
    if itd_samples <= 0:
         # Fallback to absolute average if extreme seating skew is present
         itd_samples = int((abs(itd_FL) + abs(itd_FR)) / 2)
         
    itd_samples = max(0, itd_samples) # Hard floor safeguard
    
    # Peak debug logging
    debug_peaks = (peak_L_FL, peak_R_FL, peak_L_FR, peak_R_FR)
    
    # 3. Calculate Level Differences (ILD) via Windowed RMS of direct impulses
    window = int(0.005 * 48000) 
    
    def get_rms_windowed(h, peak):
         win = h[max(0, peak - window//2) : min(len(h), peak + window//2)]
         return np.sqrt(np.mean(win**2) + 1e-12)
         
    rms_L_FL = get_rms_windowed(h_L_FL, peak_L_FL)
    rms_R_FL = get_rms_windowed(h_R_FL, peak_R_FL)
    rms_R_FR = get_rms_windowed(h_R_FR, peak_R_FR)
    rms_L_FR = get_rms_windowed(h_L_FR, peak_L_FR)
    
    k_FL = rms_R_FL / rms_L_FL if rms_L_FL > 0 else 0.0
    k_FR = rms_L_FR / rms_R_FR if rms_R_FR > 0 else 0.0
    
    crosstalk_scalar = (k_FL + k_FR) / 2.0
    
    return itd_samples, crosstalk_scalar, debug_peaks

