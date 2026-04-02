import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import math
from scipy import signal
from scipy.signal import lfilter, butter, sosfilt, sosfilt_zi, firwin2, lfilter_zi, iirnotch
from scipy.ndimage import gaussian_filter1d
import winreg
import ctypes
from ctypes import wintypes
import sys
import threading
import queue
import time
import argparse
import numpy as np
import json
from PIL import Image
import pystray

# Constants for detection
RMS_THRESHOLD = 0.005
STABILITY_WINDOW = 3.0
CHECK_INTERVAL = 0.2
BLOCK_SIZE = 0           # 0 allows WASAPI to choose the most stable hardware buffer
# Jitter buffer removed in favor of hardware-managed buffering

# Win32 Constants
SW_HIDE = 0
SW_SHOW = 5
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
SWP_FRAMECHANGED = 0x0020
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
KERNEL32 = ctypes.WinDLL('kernel32')
USER32 = ctypes.WinDLL('user32')

# Helper for 64-bit compatibility
SetWindowLongPtr = USER32.SetWindowLongPtrW if hasattr(USER32, 'SetWindowLongPtrW') else USER32.SetWindowLongW
GetWindowLongPtr = USER32.GetWindowLongPtrW if hasattr(USER32, 'GetWindowLongPtrW') else USER32.GetWindowLongW

# ITaskbarList COM Interface setup
class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", wintypes.BYTE * 8)
    ]

def string_to_guid(guid_str):
    guid = GUID()
    ctypes.windll.ole32.IIDFromString(guid_str, ctypes.byref(guid))
    return guid

CLSID_TaskbarList = string_to_guid("{56FDF344-FD6D-11d0-958A-006097C9A090}")
IID_ITaskbarList = string_to_guid("{56FDF342-FD6D-11d0-958A-006097C9A090}")

class ITaskbarListVTable(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", ctypes.c_void_p),
        ("AddRef", ctypes.c_void_p),
        ("Release", ctypes.c_void_p),
        ("HrInit", ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p)),
        ("AddTab", ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, wintypes.HWND)),
        ("DeleteTab", ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, wintypes.HWND)),
        ("ActivateTab", ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, wintypes.HWND)),
        ("SetActiveAlt", ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, wintypes.HWND)),
    ]

class ITaskbarList(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(ITaskbarListVTable))]

def get_taskbar_list():
    try:
        ctypes.windll.ole32.CoInitialize(None)
        tbl = ctypes.POINTER(ITaskbarList)()
        hr = ctypes.windll.ole32.CoCreateInstance(
            ctypes.byref(CLSID_TaskbarList), None, 1,
            ctypes.byref(IID_ITaskbarList), ctypes.byref(tbl)
        )
        if hr == 0:
            tbl.contents.lpVtbl.contents.HrInit(tbl)
            return tbl
    except:
        pass
    return None

TASKBAR_LIST = get_taskbar_list()

class AudioBypass:
    def __init__(self, input_name, output_name, latency='high', simulation=False, gain=1.0, start_minimized=False, blocksize=0, exclusive_mode=False, debug=False):
        self.debug = debug

        self.input_name = input_name
        self.output_name = output_name
        self.exclusive_mode = exclusive_mode
        self.latency = latency
        self.simulation = simulation
        self.gain = gain
        self.peak_hit = False
        self.start_minimized = start_minimized
        self.console_hidden = False
        self.blocksize = blocksize
        
        # GUI State
        self.last_peaks = [0.0] * 8
        self._gui_peaks_smooth = np.zeros(8)
        
        # System Tray components
        self.tray_icon = None
        self.tray_thread = None
        
        # Diagnostic counter
        self.callback_count = 0
        
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        try:
            with open(self.config_file, 'r') as f:
                d = json.load(f)
                self.input_name = d.get("input_device", input_name)
                self.output_name = d.get("output_device", output_name)
                self.webcam_name = d.get("webcam_device", "")
                self.webcam_index = d.get("webcam_index", 0)
                self.mic_comp_ratio = d.get("mic_comp_ratio", 4.0)
        except Exception:
            self.input_name = input_name
            self.output_name = output_name
            self.webcam_name = ""
            self.webcam_index = 0
            self.mic_name = ""
            self.mic_output_name = ""
            self.mic_enabled = False
            self.mic_highpass = True
            self.mic_hp_freq = 100.0
            self.mic_noise_gate = True
            self.mic_gate_threshold_db = -40.0
            self.mic_denoise = False
            self.mic_aec = False
            self.mic_compressor = False
            self.mic_comp_threshold_db = -20.0
            self.mic_comp_ratio = 4.0
            
        if not simulation:
            self.input_device_id, self.max_input_ch, self.input_rate = self._find_device(self.input_name, is_input=True)
            self.output_device_id, self.max_output_ch, self.output_rate = self._find_device(self.output_name, is_input=False)
            self.sample_rate = int(self.output_rate) # Use hardware default
        else:
            self.input_device_id = None
            self.max_input_ch = 8
            self.input_rate = 48000
            self.output_device_id = None
            self.max_output_ch = 8
            self.output_rate = 48000
            self.sample_rate = 48000
        
        # Audio buffering system (Decoupled Architecture - Optimized for latency)
        self.ring_buffer_size = int(self.sample_rate * 0.12) # 120ms safety wrap
        self.target_latency_samples = int(self.sample_rate * 0.010) # 10ms target for low latency
        self.buffer = np.zeros((self.ring_buffer_size, self.max_input_ch), dtype=np.float32)
        self.write_ptr = 0
        self.read_ptr = 0
        self.available_samples = 0
        self.buffer_lock = threading.Lock()
        
        # Pre-allocate for analysis
        self.pool_size = 10
        self.buffer_pool = [np.zeros((8192, self.max_input_ch), dtype=np.float32) for _ in range(self.pool_size)]
        self.pool_index = 0
        
        self.analysis_queue = queue.Queue(maxsize=self.pool_size)
        self.callback_count = 0
        self.stop_event = threading.Event()
        self.analysis_thread = None
        self.input_stream = None
        self.output_stream = None
        
        # Upmixing State
        self.upmix_enabled = True # Overall bypass
        self.manual_upmix = False # Forced upmix if True
        self.lfe_muted = False
        self.center_extraction = True # Hard center vs Phantom
        self.surround_split = False # Surround width boost
        self.independent_panning = False # Centered by default
        self.stage_equalizer = False # Psychoacoustic weighting
        self.gui_panning_x = 0.0 # -1 to 1 (Left to Right)
        self.gui_panning_y = 0.0 # -1 to 1 (Rear to Front)
        self.current_channels = 2 # Default until hardware detected
        self.failed_channel_counts = set()
        
        # Multi-Profile Output Management
        self.current_mode = "speakers" # "speakers" or "headphones"
        self.speakers_config = {}
        self.headphones_config = {}
        
        # Advanced Upmix Controls
        self.center_spread = 0.0 # 0.0 (Hard) to 1.0 (Phantom)
        self.side_gain_db = 0.0  # -6.0 to +6.0 dB
        self.width_factor = 1.0  # 0.5 to 2.0
        
        # Dynamic EQ (Safe Bass Extension)
        self.dynamic_eq = False
        self.dyn_eq_level = 0.0 # Track master volume smoothed
        
        # Consistent Stereo Width (Dynamic Expansion)
        self.consistent_width = False  # DISABLED INDEFINITELY
        self.unity_power_mode = False  # Constant Power Trig Panning - DISABLED INDEFINITELY
        self.cinema_steering = False  # Cinema surround steering
        self.gaming_panning = False   # Gaming 5.1 positional panning
        self.spectral_steering = False # Frequency-Domain steering mode - DISABLED INDEFINITELY
        self._prev_SL = np.array([0.0], dtype=np.float64)  # 1-Pole LPF state L
        self._prev_SR = np.array([0.0], dtype=np.float64)  # 1-Pole LPF state R
        self._dyn_narrowness_smooth = 0.5 # Track real-time narrowness
        self._applied_width_smooth = 1.0  # Track gain match smoothing
        self.correlation_smooth = 0.0 # 0.0 (Wide) to 1.0 (Mono)
        
        # Fixed STFT for Spectral Steering (decoupled from driver block size)
        self._stft_size = 2048                       # Fixed FFT size
        self._stft_hop = self._stft_size // 2        # 50% overlap
        self._stft_window = np.sin(np.linspace(0, np.pi, self._stft_size)).astype(np.float32)
        self._stft_in_L = np.zeros(self._stft_size // 2, dtype=np.float32)  # History buffer L
        self._stft_in_R = np.zeros(self._stft_size // 2, dtype=np.float32)  # History buffer R
        self._stft_accum_L = np.zeros(0, dtype=np.float32)  # Input accumulator L
        self._stft_accum_R = np.zeros(0, dtype=np.float32)  # Input accumulator R
        self._spec_carry = None       # Overlap-and-Add resynthesis output matrix
        self._spec_cn_smooth = None   # Per-bin cn smoothing state
        self._spec_width_smooth = None # Width scalar smoothing
        self._stft_out_accum = None   # Output accumulator for OLA results
        
        # P2: Surround Decorrelation (fixed random‐phase all-pass)
        n_fft_bins = self._stft_size // 2 + 1
        self._decorr_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, n_fft_bins)).astype(np.complex64)
        
        # P3: Per-band frequency bias array (precomputed once)
        self._freq_bins = np.linspace(0, 1, n_fft_bins, dtype=np.float32)  # Normalized, remapped at runtime
        
        # --- Optimized: Real-time Pre-allocations ---
        self._tilt_update_cnt = 0
        self._tilt_taps = None
        # Pre-allocate assembling buffer for the ring-buffer assembly
        self._src_blk_pre = np.zeros((8192, self.max_input_ch), dtype=np.float32)
        # Pre-allocate output upmix buffer
        self._upmix_out_buf = np.zeros((8192, 8), dtype=np.float32) # Standard 7.1/8ch max
        
        # P4: Rear-channel delay (Haas effect, 15ms)
        self._rear_delay_ms = 15.0
        rear_buf_len = int(self.sample_rate * 0.1)  # 100ms max
        self._rear_delay_buf_L = np.zeros(rear_buf_len, dtype=np.float32)
        self._rear_delay_buf_R = np.zeros(rear_buf_len, dtype=np.float32)
        self._rear_delay_ptr = 0
        
        # P5: Transient preservation history
        self._prev_mid_amp = None
        
        # P1: Temporally-averaged coherence accumulators
        self._coh_LR_smooth = None   # E[L * conj(R)] per bin
        self._coh_LL_smooth = None   # E[|L|²] per bin
        self._coh_RR_smooth = None   # E[|R|²] per bin
        
        # Widening Mode (Center Spread bleed to surrounds with delay)
        self.widening_mode = False
        self._widening_was_on = False  # Edge detect for buffer zero-fill
        self.widening_delay_ms = 20.0
        self.widening_buffer_size = 48000
        self.widening_buffer = np.zeros(self.widening_buffer_size, dtype=np.float32)
        self.widening_write_ptr = 0
        
        # #5 Dialogue / Vocal Lock to Center
        self.vocal_lock = False
        self._vl_prev_vocal_score = 0.0  # Smoothed vocal detection score
        
        # #9 Spectral Tilt Correction
        self.spectral_tilt = False
        self._tilt_smooth = 0.0  # Smoothed tilt measurement (dB/octave)
        self._tilt_target = -0.5  # Target slope: slightly warm
        self._tilt_filter_zi = {}  # Per-channel shelf filter state
        
        # #11 Subharmonic Synthesizer
        self.sub_synth = False
        self.sub_gain = 0.5
        self._sub_drive_cutoff = 80.0
        self._sub_low_cutoff = 20.0
        self._sub_filter_zi = {}
        self._sub_prev_sign = 0.0
        self._sub_toggle_state = 1.0
        self._sub_armed = False
        self._sub_out_sos = self._sub_hp_sos = self._sub_vocal_sos = None
        self._sub_drive_sos = None
        
        # #13 Harmonic Bass Exciter (PBE)
        self.harmonic_exciter = False
        self.harmonic_gain = 0.7
        self._exciter_filter_zi = {}
        
        # #14 Fletcher-Munson Loudness Compensation
        self.loudness_comp = False
        self._last_loudness_db = -20.0
        self._loud_b_l = self._loud_a_l = self._loud_b_h = self._loud_a_h = None
        self._loudness_filter_zi = {}
        
        # #15 True-Peak Limiter
        self.true_peak_limiter = False
        self._limiter_buffer = None
        self._limiter_ptr = 0
        
        # #16 Dynamic Dialogue Intelligibility
        self.dialogue_enhance = False
        self._diag_gain_smooth = 1.0
        
        # #12 Transaural Widening (Crosstalk Cancellation)
        self.transaural_widening = False
        self.transaural_delay = 7   # Standard average human head fallback (samples at 48kHz)
        self.transaural_gain = 0.35  # Standard cancellation factor fallback
        self.transaural_feedback = 0.08
        
        # ─── Head Tracking 3D Spatial Audio setup ───
        self.head_tracking = False
        self.head_shadow_eq = False
        self.head_tracker = None
        self.speaker_width_cm = 60.0
        self.camera_fov = 70.0
        self.speaker_distance_cm = 70.0
        self.room_width_m = 4.0

        # #17 Loudness Normalization (AGC)
        self.loudness_norm = False
        self._agc_gain_smooth = 1.0

        # #18 Transient Restoration
        self.transient_restore = False
        self.transient_amount = 4.0  # dB boost on transients

        # #19 Declipping
        self.declip = False

        # #20 Frequency-Dependent Stereo Width
        self.freq_dep_width = False

        # #21 Headphone Externalization
        self.hp_externalization = False

        # #22 Surround Decorrelation (All-pass diffuser for mix separation)
        self.surround_decorrelation = False

        # #23 Micro-Dynamics (Upward Compression)
        self.micro_dynamics = False
        self.micro_dyn_amount = 4.0  # dB max uplift on quiet detail

        # #24 Dynamic Clarity (De-Mudding)
        self.clarity_dsp = False
        self.clarity_amount = 4.0  # dB max attenuation
        self._clarity_gain_smooth = 1.0
        self._clarity_mud_sos = None
        self._clarity_peak_sos = None
        self._clarity_zi = {}

        # #25 Virtual Room (Early Reflections)
        self.virtual_room = False
        self.virtual_room_mix = 0.25
        self._room_delay_buf = None
        self._room_delay_ptr = 0
        self._room_lp_sos = None
        self._room_lp_zi = {}

        # #26 Harmonic Air Exciter
        self.air_exciter = False
        self.air_amount = 0.3
        self._air_hp_sos = None
        self._air_hp_zi = {}

        # ─── Program Level Estimator (always-on, no toggle) ──────────────
        # Very slow RMS tracker that all dynamics processors use as a reference
        # so their thresholds adapt to whatever volume the signal arrives at.
        self._program_level_db = -20.0  # design-center reference level
        self._program_level_smooth = None  # None = seed from first measurement
        self._plr_reference_db = -20.0  # the level all thresholds were designed for

        self.gui_window = None

        self.gui_window = None
        
        # Initialize defaults for Profile Safety
        self._apply_profile_dict({})

        # Load persistent settings
        self.load_config()
        self.energy_l_smooth = 1e-6
        self.energy_r_smooth = 1e-6
        self.energy_cross_smooth = 0.0
        self.smoothing_factor = 0.90 # Faster smoothing for responsiveness
        self.upmix_steering_intensity = 0.707
        self.last_mode = None

        # Set process priority to High on Windows
        try:
            import psutil
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            print("Process priority set to HIGH")
        except ImportError:
            # Fallback using os if psutil is not available
            try:
                import ctypes
                ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080) # HIGH_PRIORITY_CLASS
                print("Process priority set to HIGH (via ctypes)")
            except:
                print("Warning: Could not set process priority.")

        # We'll use the device's max channels for the multi-channel mode
        self.multi_ch_target = self.max_output_ch
        if not simulation:
            print(f"Detected hardware: {self.sample_rate}Hz, {self.max_output_ch} channels max")

        self.current_channels = 0
        self.last_active_time = time.time()
        self.is_multichannel_detected = False
        
        if self.start_minimized:
            self.hide_console()
            
        self._setup_tray()

    def _setup_tray(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        try:
            image = Image.open(icon_path)
        except Exception as e:
            print(f"Tray Icon Error: Could not load icon.png ({e}). Using default.")
            image = Image.new('RGB', (64, 64), color=(73, 109, 137))
            
        menu = pystray.Menu(
            pystray.MenuItem("Show Mixer", self.show_gui),
            pystray.MenuItem("Show/Hide Console", self.toggle_console),
            pystray.MenuItem("Exit", self.stop)
        )
        self.tray_icon = pystray.Icon("audio_analyzer", image, "Audio Analyzer", menu)

    def toggle_console(self, icon=None, item=None):
        if self.console_hidden:
            self.show_console()
        else:
            self.hide_console()

    def hide_console(self):
        hwnd = KERNEL32.GetConsoleWindow()
        if hwnd:
            # 1. Standard Win32 Style change (TOOLWINDOW removes from ALT-TAB too)
            style = GetWindowLongPtr(hwnd, GWL_EXSTYLE)
            new_style = (style | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
            SetWindowLongPtr(hwnd, GWL_EXSTYLE, new_style)
            USER32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)
            
            # 2. Explicitly remove from Taskbar via ITaskbarList (Windows 10/11 robust method)
            if TASKBAR_LIST:
                TASKBAR_LIST.contents.lpVtbl.contents.DeleteTab(TASKBAR_LIST, hwnd)
            
            USER32.ShowWindow(hwnd, SW_HIDE)
            self.console_hidden = True

    def show_gui(self, icon=None, item=None):
        if self.gui_window:
            # Reveal existing window
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(self.gui_window, "show", Qt.QueuedConnection)
            QMetaObject.invokeMethod(self.gui_window, "raise_", Qt.QueuedConnection)
            QMetaObject.invokeMethod(self.gui_window, "activateWindow", Qt.QueuedConnection)
        else:
            # Launch new GUI thread
            import gui_components
            def _gui_thread():
                gui_components.launch_gui(self)
            
            t = threading.Thread(target=_gui_thread, daemon=True)
            t.start()

    def show_console(self):
        hwnd = KERNEL32.GetConsoleWindow()
        if hwnd:
            # Re-add to Taskbar if we are showing (optional, but requested for visibility)
            if TASKBAR_LIST:
                 TASKBAR_LIST.contents.lpVtbl.contents.AddTab(TASKBAR_LIST, hwnd)
            
            USER32.ShowWindow(hwnd, SW_SHOW)
            self.console_hidden = False

    def stop(self, icon=None, item=None):
        print("Stopping application...")
        self.stop_event.set()
        if self.tray_icon:
            self.tray_icon.stop()
        
        if self.input_stream:
            try: self.input_stream.stop(); self.input_stream.close()
            except: pass
        if self.output_stream:
            try: self.output_stream.stop(); self.output_stream.close()
            except: pass
        # Mic stream cleanup removed as mic processing is deprecated

    def _find_device(self, name, is_input=True):
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        
        # Find WASAPI host API index
        wasapi_idx = -1
        for i, h in enumerate(host_apis):
            if 'wasapi' in h['name'].lower():
                wasapi_idx = i
                break
        
        if wasapi_idx == -1:
            raise ValueError("Error: Windows WASAPI host API not found.")

        # Search for device within WASAPI host
        for i, d in enumerate(devices):
            if d['hostapi'] == wasapi_idx:
                if name.lower() in d['name'].lower():
                    if (is_input and d['max_input_channels'] > 0) or (not is_input and d['max_output_channels'] > 0):
                        return i, (d['max_input_channels'] if is_input else d['max_output_channels']), d['default_samplerate']
        
        # Fallback: if not found, pick the first available WASAPI device
        print(f"Warning: Device '{name}' not found. Falling back to default WASAPI device.")
        for i, d in enumerate(devices):
            if d['hostapi'] == wasapi_idx:
                if (is_input and d['max_input_channels'] > 0) or (not is_input and d['max_output_channels'] > 0):
                    print(f"Fallback match: '{d['name']}'")
                    return i, (d['max_input_channels'] if is_input else d['max_output_channels']), d['default_samplerate']
        
        raise ValueError(f"Error: No WASAPI {'Input' if is_input else 'Output'} devices found.")

    def _analysis_worker(self):
        while not self.stop_event.is_set():
            try:
                # Get data from queue with timeout
                data = self.analysis_queue.get(timeout=0.1)
                
                # Perform Peak calculation (lighter than RMS)
                # Perform analysis on channels
                # data is (frames, channels)
                frames, chs = data.shape
                if chs <= 2:
                    self.is_multichannel_detected = False
                    continue

                # 1. Check for signal activity in surround channels (2+)
                peak = np.max(np.abs(data), axis=0)
                active = peak > RMS_THRESHOLD
                
                has_surround_signal = False
                if np.any(active[2:]):
                    # 2. Distinguish: Is it Real Surround or just our Upmix?
                    # We EXCLUDE LFE (index 3) from discrete detection
                    L = data[:, 0]
                    R = data[:, 1]
                    main_peak = max(peak[0], peak[1], 1e-6)
                    L_plus_R = L + R
                    L_minus_R = L - R

                    is_discrete = False
                    
                    # Check Center Channel (Index 2)
                    # Must be active AND have significant energy relative to L/R
                    if active[2] and (peak[2] / main_peak > 0.1):
                        # Sanity Check: if fronts are silent or softer, center is discrete (upmix can't create it from silence)
                        if peak[2] > max(peak[0], peak[1]) * 0.8:
                            is_discrete = True
                        else:
                            C = data[:, 2]
                            corr_center = np.abs(np.corrcoef(C, L_plus_R)[0, 1]) if np.any(L_plus_R) else 0
                            # corr_center > 0.99 catches pure mirrors from isolated test files on quiet channels
                            if np.isnan(corr_center) or corr_center < 0.95 or corr_center > 0.999:
                                is_discrete = True
                    
                    # Check Surrounds (Index 4, 5) - Ignore LFE at index 3
                    indices_to_check = [i for i in [4, 5] if i < chs]
                    for idx in indices_to_check:
                        if not is_discrete and active[idx] and (peak[idx] / main_peak > 0.1):
                            # Sanity Check: if fronts are silent or softer, surround is discrete (upmix can't create it from silence)
                            if peak[idx] > max(peak[0], peak[1]) * 0.8:
                                is_discrete = True
                                break
                            else:
                                S = data[:, idx]
                                corr_surround = np.abs(np.corrcoef(S, L_minus_R)[0, 1]) if np.any(L_minus_R) else 0
                                # corr_surround > 0.99 catches pure mirrors from isolated test files on quiet channels
                                if np.isnan(corr_surround) or corr_surround < 0.95 or corr_surround > 0.999:
                                    is_discrete = True
                                    break
                    
                    if is_discrete:
                        has_surround_signal = True

                if has_surround_signal:
                    self.last_active_time = time.time()
                    self.is_multichannel_detected = True
                
                # Update last_peaks for GUI metering (with smoothing for premium feel)
                # Leaky integrator: smooth_peak = 0.7 * smooth_peak + 0.3 * current_peak
                peak_arr = np.array(peak)
                if len(self._gui_peaks_smooth) != len(peak_arr):
                    self._gui_peaks_smooth = peak_arr
                else:
                    self._gui_peaks_smooth = self._gui_peaks_smooth * 0.7 + peak_arr * 0.3

                # Thread-safe publish to GUI
                self.last_peaks = self._gui_peaks_smooth.tolist()  # list copy is atomic
                
                # Master Volume Tracker (Fast attack, slow release for Dynamic EQ)
                # Tracks the highest peak between L and R
                master_peak = max(peak[0], peak[1])
                if master_peak > getattr(self, 'dyn_eq_level', 0.0):
                    self.dyn_eq_level = master_peak
                else:
                    self.dyn_eq_level = getattr(self, 'dyn_eq_level', 0.0) * 0.99 + master_peak * 0.01
                
                # Pearson Correlation Tracker for Consistent Width
                # 0.95 coeff. creates a ~1.0 second Response window
                L_data = data[:, 0]
                R_data = data[:, 1]
                if np.any(L_data) and np.any(R_data):
                    try:
                        corr = np.corrcoef(L_data, R_data)[0, 1]
                        if not np.isnan(corr):
                            self.correlation_smooth = self.correlation_smooth * 0.95 + abs(corr) * 0.05
                    except: pass

                if time.time() - self.last_active_time > STABILITY_WINDOW:
                    self.is_multichannel_detected = False
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {e}")

    def get_active_channels(self, data):
        # This is no longer used in the callback; moved to worker
        pass

    def open_input_stream(self):
        if self.input_stream: return
        try:
            # Re-allocate buffer dynamically if channel count changed
            with self.buffer_lock:
                if self.buffer.shape[1] != self.max_input_ch:
                    print(f"[Stream] Resizing input buffer to {self.max_input_ch} channels")
                    self.buffer = np.zeros((self.ring_buffer_size, self.max_input_ch), dtype=np.float32)
                    self.buffer_pool = [np.zeros((8192, self.max_input_ch), dtype=np.float32) for _ in range(self.pool_size)]
                    self.write_ptr = 0
                    self.read_ptr = 0
                    self.available_samples = 0
                    self.pool_index = 0

            self.input_stream = sd.InputStream(
                device=self.input_device_id,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype='float32',
                channels=self.max_input_ch,
                callback=self.input_callback,
                latency=self.latency
            )
            self.input_stream.start()
            print(f"Input Stream started: {self.max_input_ch} channels")
        except Exception as e:
            print(f"Failed to open input stream: {e}")

    def open_output_stream(self, channels, silent=False):
        if channels in self.failed_channel_counts:
            return 

        try:
            # 1. Close current stream
            if self.output_stream:
                try:
                    self.output_stream.stop()
                    self.output_stream.close()
                    time.sleep(0.05)
                except: pass

            # 2. Try to open new one
            wasapi_settings = sd.WasapiSettings(exclusive=self.exclusive_mode) if self.exclusive_mode else None
            
            self.output_stream = sd.OutputStream(
                device=self.output_device_id,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype='float32',
                channels=channels,
                callback=self.output_callback,
                latency='low',
                extra_settings=wasapi_settings
            )
            self.output_stream.start()
            self.current_channels = channels 
            if not silent:
                mode_str = "(Exclusive)" if self.exclusive_mode else "(Shared)"
                print(f"Output Stream active: {channels} channels {mode_str}")

        except Exception:
            self.failed_channel_counts.add(channels)
            # Silently fallback to Stereo if we were trying Surround
            if channels != 2:
                self.open_output_stream(2, silent=True)
            elif not silent:
                print(f"CRITICAL ERROR: Could not open any output stream on device {self.output_device_id}")

    def _init_multiband_filters(self):
        sr = getattr(self, 'sample_rate', 48000)
        if sr < 4000: sr = 48000 # Absolute safety for normalization
        
        # Tilt shelf: simple 1st-order shelf via biquad
        self._tilt_lo_sos = butter(1, 1000.0 / (sr / 2.0), btype='low', output='sos')
        self._tilt_hi_sos = butter(1, 1000.0 / (sr / 2.0), btype='high', output='sos')
        
        # #11 Subharmonic Synthesizer Filters
        # Drive bandpass: [low_cutoff .. high_cutoff] Hz
        low  = max(15.0, getattr(self, '_sub_low_cutoff', 20.0))
        high = max(low + 5.0, getattr(self, '_sub_drive_cutoff', 80.0))
        self._sub_drive_sos = butter(2, [low / (sr / 2), high / (sr / 2)], btype='bandpass', output='sos')
        
        # #4 Room Boundary LP Filter (400Hz cut)
        self._boundary_lp_sos = butter(1, 400.0 / (sr / 2), btype='low', output='sos')
        
        # #5 Air Absorption LP Filter (2000Hz cut)
        self._absorp_lp_sos = butter(1, 2000.0 / (sr / 2), btype='low', output='sos')

        # Output Smooth Lowpass: 45Hz
        self._sub_out_sos = butter(2, 45.0 / (sr / 2), btype='low', output='sos')
        # Safety Highpass: 20Hz
        self._sub_hp_sos = butter(2, 20.0 / (sr / 2), btype='high', output='sos')
        # Vocal Range Detection: 100-300Hz
        self._sub_vocal_sos = butter(2, [100.0 / (sr / 2), 300.0 / (sr / 2)], btype='bandpass', output='sos')
        
        # #12 Transaural Crosstalk Lowpass (Dynamic Cutoff bound to ITD)
        cutoff = 2200.0
        delay_samps = getattr(self, 'transaural_delay', 10)
        if delay_samps > 1:
             delay_sec = delay_samps / sr
             if delay_sec > 0:
                 cutoff = min(3000.0, 1.0 / (2.0 * delay_sec))  # Rule: f_c = 1 / (2 * ITD)
             
        self._trans_lp_sos = butter(2, cutoff / (sr / 2), btype='low', output='sos')
        
        # #13 Head-Shadow Lowpass for static blending (Rolls off highs above 1.8kHz)
        self._shadow_lp_sos = butter(1, 1800.0 / (sr / 2), btype='low', output='sos')

        # #15 Elevation compensation bandpass (6-10kHz pinna notch region)
        hi_elev = min(0.95, 10000.0 / (sr / 2))
        lo_elev = 6000.0 / (sr / 2)
        if lo_elev < hi_elev:
            self._elev_bp_sos = butter(2, [lo_elev, hi_elev], btype='band', output='sos')

        # #16 Distance-dependent bass shelf (gentle LF compensation for front speakers)
        self._dist_bass_sos = butter(1, 200.0 / (sr / 2), btype='low', output='sos')

        # #24 Dynamic Clarity (De-Mudding) filters
        # Mud band: 200-500Hz for detection
        self._clarity_mud_sos = butter(2, [200.0 / (sr / 2), 500.0 / (sr / 2)], btype='bandpass', output='sos')
        # Attenuation peak: 350Hz, Q=1.4
        w0 = 350.0 / (sr / 2)
        bw = w0 / 1.4
        self._clarity_peak_sos = butter(2, [max(0.01, w0 - bw/2), min(0.99, w0 + bw/2)], btype='bandstop', output='sos')

        # #25 Virtual Room Damping (High-frequency absorption at 4kHz)
        self._room_lp_sos = butter(1, 4000.0 / (sr / 2.0), btype='low', output='sos')

        # #26 Harmonic Air Exciter HPF (10.5kHz)
        self._air_hp_sos = butter(2, 10500.0 / (sr / 2.0), btype='high', output='sos')

    def _apply_head_tracking_panning(self, out_block, frames, out_ch, is_passthrough=False):
        """Dynamic Sweet-Spot Loudness Panning applied globally (Fronts & Surrounds)."""
        is_ht = getattr(self, 'head_tracking', False)
        is_hp = getattr(self, 'headphone_tracking', False)
        if not (is_ht or is_hp) or not hasattr(self, 'head_tracker') or not self.head_tracker:
            return
            
        pos_x = self.head_tracker.pos_x # Range -0.5 to 0.5
        pos_z = self.head_tracker.pos_z # ~1.0
        
        S_WIDTH = getattr(self, 'speaker_width_cm', 60.0)
        S_DIST = getattr(self, 'speaker_distance_cm', 70.0)
        
        user_z = S_DIST * pos_z
        camera_fov = getattr(self, 'camera_fov', 70.0)
        fov_mult = 2.0 * np.tan(np.radians(camera_fov / 2.0))
        camera_view_width_cm = user_z * fov_mult
        user_x = pos_x * camera_view_width_cm
        
        # Default compensation values to avoid NameError in logging if bypassed
        comp_L = 1.0; comp_R = 1.0; comp_C = 1.0
        comp_SL = 1.0; comp_SR = 1.0; comp_BL = 1.0; comp_BR = 1.0

        # --- Front speaker distances ---
        x_L = user_x - (-S_WIDTH/2)
        dist_L = np.sqrt(x_L**2 + user_z**2)

        x_R = user_x - (+S_WIDTH/2)
        dist_R = np.sqrt(x_R**2 + user_z**2)

        dist_C = np.sqrt(user_x**2 + user_z**2)

        current_base_dist = np.sqrt((S_WIDTH/2)**2 + user_z**2)
        current_base_dist_C = user_z if user_z > 0.1 else 1.0

        # Aggressive Amplitude Compensation (Negative logic: Attenuate only)
        # Never boost > 1.0 to prevent digital clipping/gain addition.
        comp_L = min(1.0, (dist_L / current_base_dist) ** 2.5)
        comp_R = min(1.0, (dist_R / current_base_dist) ** 2.5)
        comp_C = min(1.0, (dist_C / current_base_dist_C) ** 2.5)

        # ITD (Precedence Effect) Delay Compensation
        # In normal/cinema mode: FRONTS only — surrounds carry diffuse ambience
        # where delay comp creates comb filtering and amplifies tracker noise.
        # In gaming mode: ALL speakers — surrounds carry discrete positional audio
        # (point sources) that benefit from proper time alignment.
        is_gaming = getattr(self, 'gaming_panning', False)
        cm_per_sample = 34300.0 / self.sample_rate

        # Surround speaker geometry (ITU-R BS.775: SL/SR at ±110°, BL/BR at ±150°)
        R_spk = S_DIST
        SL_x = -R_spk * 0.9397;  SL_z = S_DIST + R_spk * 0.3420
        SR_x =  R_spk * 0.9397;  SR_z = SL_z
        BL_x = -R_spk * 0.5;     BL_z = S_DIST + R_spk * 0.8660
        BR_x =  R_spk * 0.5;     BR_z = BL_z

        all_dists = [dist_L, dist_R, dist_C]

        if is_gaming and not is_hp and out_ch > 4:
            dist_SL = np.sqrt((user_x - SL_x)**2 + (user_z - SL_z)**2)
            dist_SR = np.sqrt((user_x - SR_x)**2 + (user_z - SR_z)**2)
            all_dists += [dist_SL, dist_SR]
        if is_gaming and not is_hp and out_ch > 6:
            dist_BL = np.sqrt((user_x - BL_x)**2 + (user_z - BL_z)**2)
            dist_BR = np.sqrt((user_x - BR_x)**2 + (user_z - BR_z)**2)
            all_dists += [dist_BL, dist_BR]

        max_dist = max(all_dists)

        target_delay_L  = max(0.0, (max_dist - dist_L)  / cm_per_sample)
        target_delay_R  = max(0.0, (max_dist - dist_R)  / cm_per_sample)
        target_delay_C  = max(0.0, (max_dist - dist_C)  / cm_per_sample)

        # Smoothing (fast for amplitude, heavy for delays to prevent Doppler)
        _comp_keys  = ['_ht_cs_L','_ht_cs_R','_ht_cs_C']
        _comp_vals  = [comp_L, comp_R, comp_C]
        _delay_keys = ['_ht_ds_L','_ht_ds_R','_ht_ds_C']
        _delay_vals = [target_delay_L, target_delay_R, target_delay_C]

        # Gaming mode: surrounds carry discrete point sources → full compensation
        if is_gaming and not is_hp:
            if out_ch > 4:
                # Surround base distance = R_spk (distance from sweet-spot to surround speaker)
                comp_SL = min(1.0, (dist_SL / R_spk) ** 2.5)
                comp_SR = min(1.0, (dist_SR / R_spk) ** 2.5)
                _comp_keys += ['_ht_cs_SL','_ht_cs_SR']
                _comp_vals += [comp_SL, comp_SR]
                _delay_keys += ['_ht_ds_SL','_ht_ds_SR']
                _delay_vals += [max(0.0, (max_dist - dist_SL) / cm_per_sample),
                                max(0.0, (max_dist - dist_SR) / cm_per_sample)]
            if out_ch > 6:
                comp_BL = min(1.0, (dist_BL / R_spk) ** 2.5)
                comp_BR = min(1.0, (dist_BR / R_spk) ** 2.5)
                _comp_keys += ['_ht_cs_BL','_ht_cs_BR']
                _comp_vals += [comp_BL, comp_BR]
                _delay_keys += ['_ht_ds_BL','_ht_ds_BR']
                _delay_vals += [max(0.0, (max_dist - dist_BL) / cm_per_sample),
                                max(0.0, (max_dist - dist_BR) / cm_per_sample)]

        if not hasattr(self, '_ht_cs_L'):
            for k, v in zip(_comp_keys, _comp_vals):   setattr(self, k, v)
            for k, v in zip(_delay_keys, _delay_vals):  setattr(self, k, v)
        else:
            for k, v in zip(_comp_keys, _comp_vals):
                setattr(self, k, getattr(self, k) * 0.30 + v * 0.70)
            for k, v in zip(_delay_keys, _delay_vals):
                setattr(self, k, getattr(self, k) * 0.95 + v * 0.05)
            # When switching OUT of gaming mode, decay surround comp/delays to neutral
            if not is_gaming:
                for k in ['_ht_ds_SL','_ht_ds_SR','_ht_ds_BL','_ht_ds_BR']:
                    if hasattr(self, k):
                        setattr(self, k, getattr(self, k) * 0.95)  # → 0
                for k in ['_ht_cs_SL','_ht_cs_SR','_ht_cs_BL','_ht_cs_BR']:
                    if hasattr(self, k):
                        setattr(self, k, getattr(self, k) * 0.95 + 1.0 * 0.05)  # → 1.0

        # Apply Amplitude Panning — fronts always, surrounds only in gaming mode
        ramp_comp_L = self._get_ramp('_last_ht_comp_L', self._ht_cs_L, frames)
        ramp_comp_R = self._get_ramp('_last_ht_comp_R', self._ht_cs_R, frames)
        ramp_comp_C = self._get_ramp('_last_ht_comp_C', self._ht_cs_C, frames)

        out_block[:frames, 0] *= ramp_comp_L
        out_block[:frames, 1] *= ramp_comp_R
        if out_ch > 2:
            out_block[:frames, 2] *= ramp_comp_C

        if is_gaming and not is_hp:
            if out_ch > 4:
                ramp_SL = self._get_ramp('_last_ht_comp_SL', getattr(self, '_ht_cs_SL', 1.0), frames)
                ramp_SR = self._get_ramp('_last_ht_comp_SR', getattr(self, '_ht_cs_SR', 1.0), frames)
                out_block[:frames, 4] *= ramp_SL
                out_block[:frames, 5] *= ramp_SR
            if out_ch > 6:
                ramp_BL = self._get_ramp('_last_ht_comp_BL', getattr(self, '_ht_cs_BL', 1.0), frames)
                ramp_BR = self._get_ramp('_last_ht_comp_BR', getattr(self, '_ht_cs_BR', 1.0), frames)
                out_block[:frames, 6] *= ramp_BL
                out_block[:frames, 7] *= ramp_BR
                
        # --- Head-Shadow EQ (Angle-Dependent HF Roll-Off — FRONTS only) ---
        # Surrounds are excluded: shadow EQ on diffuse ambient content creates
        # tonal asymmetry and makes the room envelope shift with tracker noise.
        yaw = getattr(self.head_tracker, 'rot_yaw', 0.0)

        if getattr(self, 'head_shadow_eq', False):
            if not hasattr(self, '_shadow_lp_zi'):
                self._shadow_lp_zi = {}

            angle_L = np.abs(np.arctan2(user_x - (-S_WIDTH/2), user_z))
            angle_R = np.abs(np.arctan2(user_x - (+S_WIDTH/2), user_z))

            shadow_map = {
                0: np.abs(angle_L - yaw),          # FL
                1: np.abs(angle_R + yaw),           # FR
            }
            if out_ch > 2:
                angle_C = np.arctan2(user_x, max(0.1, user_z))
                shadow_map[2] = np.abs(angle_C - yaw)  # Center

            for ch, eff_angle in shadow_map.items():
                blend_raw = float(np.clip((eff_angle - 0.26) / 0.52, 0.0, 0.45))
                blend = self._get_ramp(f'_last_shadow_{ch}', blend_raw, frames)

                if ch not in self._shadow_lp_zi:
                    self._shadow_lp_zi[ch] = sosfilt_zi(self._shadow_lp_sos) * 0

                lp_sig, self._shadow_lp_zi[ch] = sosfilt(self._shadow_lp_sos, out_block[:frames, ch], zi=self._shadow_lp_zi[ch])
                out_block[:frames, ch] = out_block[:frames, ch] * (1 - blend) + lp_sig * blend
            
        # (Delay Alignment logic resides below inside 'not is_hp' scope)
        
        # --- Advanced Features ---
        # #1 Dynamic Soundfield Rotation offset
        # If turning left (yaw > 0), offset is negative (pan right to cancel)
        if hasattr(self, 'head_tracker') and self.head_tracker:
             self._dynamic_pan_offset = -self.head_tracker.rot_yaw / (np.pi / 2.0)
        else:
             self._dynamic_pan_offset = 0.0

        # #3 Dynamic Stereo Width Based on Subtended Angle
        S_WIDTH = getattr(self, 'speaker_width_cm', 60.0)
        u_z = max(10.0, float(user_z)) # Avoid divide by zero
        theta = 2 * np.arctan((S_WIDTH / 2.0) / u_z)
        theta_base = 2 * np.arctan((S_WIDTH / 2.0) / 70.0)
        self._dynamic_width_ratio = theta_base / theta

        if not is_hp:
            # #4 Room Boundary Compensation (Wall proximity)
            room_width_cm = getattr(self, 'room_width_m', 4.0) * 100.0
            dist_L_wall = max(0.1, (room_width_cm / 2.0) + user_x)
            dist_R_wall = max(0.1, (room_width_cm / 2.0) - user_x)
            
            weight_L = max(0.0, 1.0 - (dist_L_wall / 100.0))
            weight_R = max(0.0, 1.0 - (dist_R_wall / 100.0))
            if not hasattr(self, '_bound_weight_L'):
                self._bound_weight_L = weight_L
                self._bound_weight_R = weight_R
            else:
                self._bound_weight_L = self._bound_weight_L * 0.95 + weight_L * 0.05
                self._bound_weight_R = self._bound_weight_R * 0.95 + weight_R * 0.05

            if hasattr(self, '_boundary_lp_sos'):
                if not hasattr(self, '_bound_lp_zi_L'):
                    self._bound_lp_zi_L = sosfilt_zi(self._boundary_lp_sos) * 0.0
                    self._bound_lp_zi_R = sosfilt_zi(self._boundary_lp_sos) * 0.0
                    
                lp_L, self._bound_lp_zi_L = sosfilt(self._boundary_lp_sos, out_block[:frames, 0], zi=self._bound_lp_zi_L)
                out_block[:frames, 0] = out_block[:frames, 0] - 0.4 * self._bound_weight_L * lp_L
                
                lp_R, self._bound_lp_zi_R = sosfilt(self._boundary_lp_sos, out_block[:frames, 1], zi=self._bound_lp_zi_R)
                out_block[:frames, 1] = out_block[:frames, 1] - 0.4 * self._bound_weight_R * lp_R

            # #5 Air Absorption Compensation
            boost_factor = max(0.0, min(0.3, (u_z - 70.0) * 0.005))
            if boost_factor > 0 and hasattr(self, '_absorp_lp_sos'):
                # Reset stale filter state if the user just crossed the depth threshold
                if not hasattr(self, '_abs_lp_zi_L') or getattr(self, '_last_abs_boost', 0.0) <= 0.0:
                    self._abs_lp_zi_L = sosfilt_zi(self._absorp_lp_sos) * 0.0
                    self._abs_lp_zi_R = sosfilt_zi(self._absorp_lp_sos) * 0.0
                self._last_abs_boost = boost_factor
                
                lp_L_hf, self._abs_lp_zi_L = sosfilt(self._absorp_lp_sos, out_block[:frames, 0], zi=self._abs_lp_zi_L)
                hp_L = out_block[:frames, 0] - lp_L_hf
                out_block[:frames, 0] += hp_L * boost_factor
                
                lp_R_hf, self._abs_lp_zi_R = sosfilt(self._absorp_lp_sos, out_block[:frames, 1], zi=self._abs_lp_zi_R)
                hp_R = out_block[:frames, 1] - lp_R_hf
                out_block[:frames, 1] += hp_R * boost_factor
            else:
                self._last_abs_boost = 0.0

            # --- #16 Distance-Dependent Bass Compensation (front speakers) ---
            dist_ratio = user_z / max(1.0, S_DIST)
            # Positive = farther (boost bass), negative = closer (cut bass), ±1.5 dB max
            bass_factor = float(np.clip((dist_ratio - 1.0) * 0.3, -0.15, 0.15))

            if not hasattr(self, '_ht_bass_smooth'):
                self._ht_bass_smooth = bass_factor
            else:
                self._ht_bass_smooth = self._ht_bass_smooth * 0.95 + bass_factor * 0.05

            if abs(self._ht_bass_smooth) > 0.005 and hasattr(self, '_dist_bass_sos'):
                if not hasattr(self, '_dist_bass_zi'):
                    self._dist_bass_zi = {}
                bass_ramp = self._get_ramp('_last_bass_factor', self._ht_bass_smooth, frames)
                for ch in [0, 1, 2]:  # Front speakers only
                    if ch >= out_ch: break
                    if ch not in self._dist_bass_zi:
                        self._dist_bass_zi[ch] = sosfilt_zi(self._dist_bass_sos) * 0
                    lp_sig, self._dist_bass_zi[ch] = sosfilt(self._dist_bass_sos, out_block[:frames, ch], zi=self._dist_bass_zi[ch])
                    out_block[:frames, ch] += lp_sig * bass_ramp

            # --- Per-Speaker Circular Buffer Delay Alignment ---
            # Fronts always; surrounds only in gaming mode (discrete point sources)
            _delay_ch_map = {
                0: '_ht_ds_L', 1: '_ht_ds_R', 2: '_ht_ds_C',
            }
            if is_gaming:
                if out_ch > 4:
                    _delay_ch_map[4] = '_ht_ds_SL'
                    _delay_ch_map[5] = '_ht_ds_SR'
                if out_ch > 6:
                    _delay_ch_map[6] = '_ht_ds_BL'
                    _delay_ch_map[7] = '_ht_ds_BR'
            any_delay = any(getattr(self, v, 0.0) > 0.01 for ch, v in _delay_ch_map.items() if ch < out_ch)

            if any_delay:
                if not hasattr(self, '_ht_delay_buf') or self._ht_delay_buf.shape[0] < frames + 2048 or self._ht_delay_buf.shape[1] != out_ch:
                    self._ht_delay_buf = np.zeros((8192, out_ch), dtype=np.float32)
                    self._ht_delay_ptr = 0

                buf_len = self._ht_delay_buf.shape[0]
                w_indices = (self._ht_delay_ptr + np.arange(frames)) % buf_len
                self._ht_delay_buf[w_indices, :] = out_block[:frames, :]

                for ch, delay_attr in _delay_ch_map.items():
                    if ch >= out_ch:
                        continue
                    d_val = getattr(self, delay_attr, 0.0)
                    if d_val < 0.01:
                        continue
                    d_arr = self._get_ramp(f'_last_ht_d_{ch}', d_val, frames)
                    d_int = d_arr.astype(np.int32)
                    frac = d_arr - d_int
                    r_0 = (w_indices - d_int) % buf_len
                    r_1 = (r_0 - 1) % buf_len
                    out_block[:frames, ch] = self._ht_delay_buf[r_0, ch] * (1 - frac) + self._ht_delay_buf[r_1, ch] * frac

                self._ht_delay_ptr = (self._ht_delay_ptr + frames) % buf_len

        import time
        current_time = time.time()
        if getattr(self, 'debug', False) and (not hasattr(self, '_last_ht_log_time') or (current_time - getattr(self, '_last_ht_log_time', 0)) >= 1.0):
             self._last_ht_log_time = current_time
             yaw_deg = np.degrees(yaw)
             print(f"[{current_time:.2f}][HT-PAN] X:{user_x:+.1f}cm Z:{user_z:.1f}cm | Yaw: {yaw_deg:+.1f}° | Comp L:{comp_L:.2f}x R:{comp_R:.2f}x")

    # ─── #19 Declipping ───────────────────────────────────────────────────
    def _apply_declip(self, block):
        """Detect hard-clipped peaks and restore via linear interpolation."""
        threshold = 0.99
        for ch in range(block.shape[1]):
            sig = block[:, ch]
            clip_mask = np.abs(sig) >= threshold
            if not np.any(clip_mask):
                continue
            indices = np.arange(len(sig))
            good = ~clip_mask
            if np.sum(good) < 4:
                continue
            sig[clip_mask] = np.interp(indices[clip_mask], indices[good], sig[good])
            peak = np.max(np.abs(sig))
            if peak > 0.98:
                sig *= 0.98 / peak

    # ─── #17 Loudness Normalization (AGC) ─────────────────────────────────
    def _apply_loudness_norm(self, block):
        """Slow-acting automatic gain control targeting -23 LUFS."""
        sr = self.sample_rate
        frames = block.shape[0]

        # Measure RMS on L/R (or all channels if mono/surround)
        ch_count = min(block.shape[1], 2)
        rms = np.sqrt(np.mean(block[:, :ch_count] ** 2) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)

        # Gate: don't adjust during silence or very quiet passages
        if db < -55:
            return

        target_db = -23.0
        desired_gain_db = np.clip(target_db - db, -12.0, 12.0)
        desired_gain = 10.0 ** (desired_gain_db / 20.0)

        # Asymmetric smoothing: fast turn-down (500ms), slow turn-up (3s)
        if desired_gain < self._agc_gain_smooth:
            alpha = 1.0 - np.exp(-frames / (sr * 0.5))
        else:
            alpha = 1.0 - np.exp(-frames / (sr * 3.0))

        self._agc_gain_smooth = self._agc_gain_smooth * (1.0 - alpha) + desired_gain * alpha

        ramp = self._get_ramp('_last_agc_gain', self._agc_gain_smooth, frames)
        block *= ramp[:, np.newaxis]

    # ─── #18 Transient Restoration ────────────────────────────────────────
    @staticmethod
    def _asym_env_decimated(abs_block, att_coeff, rel_coeff, zi, dec=8):
        """Asymmetric envelope follower at 1/dec sample rate.

        Decimates the input (taking per-chunk peak), runs the serial recurrence
        on ~1/8 as many samples, then linearly interpolates back to full rate.
        At 48 kHz dec=8 → 6 kHz envelope rate, still well above transient needs.
        Cuts Python loop iterations by 8x.
        """
        n_samp, n_ch = abs_block.shape
        # Pad to multiple of dec
        pad = (dec - n_samp % dec) % dec
        if pad:
            padded = np.concatenate([abs_block, np.zeros((pad, n_ch), dtype=abs_block.dtype)])
        else:
            padded = abs_block
        n_padded = padded.shape[0]
        n_dec = n_padded // dec

        # Decimated input: peak within each chunk (preserves transient amplitude)
        dec_in = padded.reshape(n_dec, dec, n_ch).max(axis=1)

        # Adjust coefficients for decimated rate: coeff^dec
        att_d = att_coeff ** dec
        rel_d = rel_coeff ** dec

        # Serial loop at decimated rate
        dec_env = np.empty_like(dec_in)
        prev = zi.copy()
        for i in range(n_dec):
            s = dec_in[i:i+1]
            cf = np.where(s > prev, att_d, rel_d)
            prev = cf * prev + (1.0 - cf) * s
            dec_env[i] = prev[0]

        # Linear interpolation back to full rate
        # Build sample indices for the decimated envelope (center of each chunk)
        x_dec = np.arange(n_dec) * dec + dec // 2
        x_full = np.arange(n_samp)
        env = np.empty((n_samp, n_ch), dtype=abs_block.dtype)
        for ch in range(n_ch):
            env[:, ch] = np.interp(x_full, x_dec, dec_env[:, ch])

        return env, prev

    def _apply_transient_restore(self, block, ref_block=None):
        """Transient enhancement via matched-release dual envelope detection.

        Envelope followers run at 1/8 sample rate to avoid per-sample Python
        overhead that caused buffer underruns under CPU contention.

        When ref_block is provided (pre-micro-dynamics snapshot), envelope
        detection runs on the original signal so upward compression doesn't
        reduce the perceived transient-to-sustain ratio.  The computed gain
        is still applied to the current (post-micro-dyn) block.
        """
        sr = self.sample_rate
        n_ch = block.shape[1]
        n_samp = block.shape[0]

        # Use reference block for envelope detection if available
        detect = ref_block if ref_block is not None else block
        abs_block = np.abs(detect).astype(np.float64)

        if not hasattr(self, '_tr_fast_zi') or self._tr_fast_zi.shape[1] != n_ch:
            self._tr_fast_zi = np.zeros((1, n_ch), dtype=np.float64)
            self._tr_slow_zi = np.zeros((1, n_ch), dtype=np.float64)
            self._tr_gain_zi = np.ones((1, n_ch), dtype=np.float64)

        # Two envelopes with MATCHED release but different attack times.
        fast_att = np.exp(-1.0 / (sr * 0.0005))  # 0.5ms
        slow_att = np.exp(-1.0 / (sr * 0.020))   # 20ms
        release  = np.exp(-1.0 / (sr * 0.100))   # 100ms — shared

        fast_env, self._tr_fast_zi = self._asym_env_decimated(
            abs_block, fast_att, release, self._tr_fast_zi)
        slow_env, self._tr_slow_zi = self._asym_env_decimated(
            abs_block, slow_att, release, self._tr_slow_zi)

        # RMS gate: no boost below program_level - 20 dB (adapts to input volume)
        plr_offset = getattr(self, '_program_level_db', -20.0) - getattr(self, '_plr_reference_db', -20.0)
        rms_floor = 10.0 ** ((-40.0 + plr_offset) / 20.0)
        gate = np.clip((slow_env - rms_floor * 0.5) / (rms_floor * 0.5), 0.0, 1.0)

        # Ratio — clamped denominator to rms_floor so it can never explode
        ratio = fast_env / np.maximum(slow_env, rms_floor)

        # Map ratio to gain: 1.0 = sustained (no boost), above 1.2 = onset.
        # 20% dead zone filters out minor envelope wobble from normal dynamics
        max_boost = 10.0 ** (self.transient_amount / 20.0)
        raw_gain = 1.0 + gate * (max_boost - 1.0) * np.clip(
            (ratio - 1.2) / (2.5 - 1.2), 0.0, 1.0)

        # Asymmetric gain ballistics: instant attack, 60ms decay
        gain_rel = np.exp(-1.0 / (sr * 0.060))
        gain_env, self._tr_gain_zi = self._asym_env_decimated(
            raw_gain, 0.0, gain_rel, self._tr_gain_zi)

        block *= gain_env.astype(np.float32)

    # ─── Program Level Estimator ────────────────────────────────────────
    def _update_program_level(self, block):
        """Ultra-slow RMS measurement (~30s time constant) that tracks the
        average operating level of the incoming signal.  All dynamics
        processors read ``self._program_level_db`` and offset their
        thresholds by ``self._program_level_db - self._plr_reference_db``
        so they behave identically whether Windows volume is at 10% or 100%,
        or whether the source is a quiet podcast vs. a loud master.

        This method **never** touches the audio — it only measures.
        """
        sr = self.sample_rate
        frames = block.shape[0]
        ch_count = min(block.shape[1], 2)  # measure L/R only

        rms = np.sqrt(np.mean(block[:, :ch_count] ** 2) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)

        # Gate: ignore silence / near-silence so pauses between tracks
        # don't drag the estimate down
        if db < -60.0:
            return

        # Seed from first real measurement so we don't spend 30s converging
        if self._program_level_smooth is None:
            self._program_level_smooth = db
            self._program_level_db = db
            return

        # Asymmetric smoothing in dB domain:
        #   - If signal is LOUDER than estimate: 8 second rise (don't jump on transients)
        #   - If signal is QUIETER than estimate: 30 second fall (very slow decay)
        # This gives a stable "what loudness range is this content in?" answer.
        if db > self._program_level_smooth:
            tau = 8.0
        else:
            tau = 30.0

        alpha = 1.0 - np.exp(-frames / (sr * tau))
        self._program_level_smooth += alpha * (db - self._program_level_smooth)
        self._program_level_db = self._program_level_smooth

    # ─── #23 Micro-Dynamics (Upward Compression) ─────────────────────────
    def _apply_micro_dynamics(self, block):
        """Upward compression: lift quiet detail without touching peaks.

        Block-level RMS comparison with a long-term reference:
          - Measure this block's RMS per channel
          - Compare to a 5-second exponential moving average (the "normal level")
          - If block RMS < reference → quiet passage → boost proportionally
          - Smooth gain ramp across block to avoid discontinuities

        Fully volume-independent: the reference is the signal's own average.

        Combined with transient restore (which boosts attacks), this fills in
        the sustain/decay portions that modern loudness-war masters crush.
        """
        sr = self.sample_rate
        n_samp, n_ch = block.shape
        if n_samp < 64:
            return

        max_boost_db = getattr(self, 'micro_dyn_amount', 4.0)

        # ── Lazy-init state ──
        if not hasattr(self, '_ud_ref_db') or len(self._ud_ref_db) != n_ch:
            self._ud_ref_db = np.full(n_ch, -100.0, dtype=np.float64)
            self._ud_prev_gain_db = np.zeros(n_ch, dtype=np.float64)
            self._ud_initialized = False

        # ── Per-channel block RMS in dB ──
        rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=0) + 1e-12)
        block_db = 20.0 * np.log10(rms + 1e-12)

        # ── Update long-term reference (5s EMA) ──
        # Only update from non-silent blocks (gate at -60 dBFS)
        block_seconds = n_samp / sr
        alpha = 1.0 - np.exp(-block_seconds / 5.0)

        if not self._ud_initialized:
            # Seed from first block
            self._ud_ref_db = block_db.copy()
            self._ud_initialized = True
        else:
            # Only update channels above silence threshold
            active = block_db > -60.0
            self._ud_ref_db[active] += alpha * (block_db[active] - self._ud_ref_db[active])

        # ── Deficit: how much quieter than reference (per channel) ──
        deficit_db = np.clip(self._ud_ref_db - block_db, 0.0, 12.0)

        # Aggressive mapping: full boost reached at just 4 dB of deficit
        # Modern mastered music only has 2-4 dB of block-to-block variation
        gain_db = np.clip(deficit_db * (max_boost_db / 4.0), 0.0, max_boost_db)

        # Gate: no boost during silence
        gate = np.clip((block_db + 60.0) / 10.0, 0.0, 1.0)
        gain_db *= gate

        # ── Smooth gain target over time (prevent pumping) ──
        # Asymmetric: 200ms rise (don't suddenly boost), 50ms fall (quickly back off)
        block_seconds = n_samp / sr
        for ch in range(n_ch):
            tau = 0.200 if gain_db[ch] > self._ud_prev_gain_db[ch] else 0.050
            alpha = 1.0 - np.exp(-block_seconds / tau)
            gain_db[ch] = self._ud_prev_gain_db[ch] + alpha * (gain_db[ch] - self._ud_prev_gain_db[ch])

        # ── Smooth ramp across block for click-free transitions ──
        ramp = np.linspace(self._ud_prev_gain_db, gain_db, n_samp, dtype=np.float64)
        gain_linear = 10.0 ** (ramp / 20.0)
        self._ud_prev_gain_db = gain_db.copy()

        block *= gain_linear.astype(np.float32)

    # ─── #22 Surround Decorrelation (All-Pass Diffuser) ────────────────────
    def _apply_surround_decorrelation(self, sig_L, sig_R):
        """Apply cascaded all-pass diffuser to side-channel content for surround routing.

        Makes surrounds sound genuinely ambient rather than 'delayed copy of fronts',
        dramatically improving perceived vocal/center separation.
        All-pass filters are magnitude-transparent: no coloring, no level change,
        only phase dispersion. Completely safe on all content types.
        """
        if not hasattr(self, '_decorr_ap_states'):
            sr = self.sample_rate
            # 6 cascaded all-pass stages with prime-number delays (in samples)
            # Chosen to avoid reinforcing any single frequency
            # Delays span 0.3ms-2.5ms for broadband diffusion
            self._decorr_delays = [int(sr * d) for d in
                                   [0.00029, 0.00071, 0.00113, 0.00167, 0.00199, 0.00251]]
            # All-pass coefficient per stage: controls diffusion density
            # Higher = more diffuse but more latent; 0.5-0.7 is the sweet spot
            self._decorr_coeffs = [0.60, 0.55, 0.62, 0.58, 0.65, 0.52]
            # Different delay/coeff orderings per channel maximize decorrelation
            self._decorr_delays_R = list(reversed(self._decorr_delays))
            self._decorr_coeffs_R = [0.52, 0.65, 0.58, 0.62, 0.55, 0.60]
            # Per-channel state: each stage has a delay buffer
            self._decorr_ap_states = {}
            for ch_tag, delays in [('L', self._decorr_delays), ('R', self._decorr_delays_R)]:
                self._decorr_ap_states[ch_tag] = []
                for d in delays:
                    self._decorr_ap_states[ch_tag].append({
                        'buf': np.zeros(d, dtype=np.float32),
                        'ptr': 0
                    })

        def _process_channel(sig, ch_tag):
            out = sig.copy()
            delays = self._decorr_delays if ch_tag == 'L' else self._decorr_delays_R
            coeffs = self._decorr_coeffs if ch_tag == 'L' else self._decorr_coeffs_R
            for i, (delay, coeff) in enumerate(zip(delays, coeffs)):
                state = self._decorr_ap_states[ch_tag][i]
                buf = state['buf']
                ptr = state['ptr']
                blen = len(buf)
                
                num_samples = len(out)
                result = np.empty(num_samples, dtype=np.float32)
                
                it = 0
                while it < num_samples:
                    chunk = min(blen, num_samples - it)
                    
                    # 1. Fetch delayed samples (ptr iterates the delay buffer of length 'delay')
                    end_idx = min(ptr + chunk, blen)
                    first_part = end_idx - ptr
                    
                    delayed = np.empty(chunk, dtype=np.float32)
                    delayed[:first_part] = buf[ptr:end_idx]
                    if first_part < chunk:
                        delayed[first_part:] = buf[:chunk - first_part]
                    
                    # 2. Apply all-pass: result = -g*in + delayed, buf_new = in + g*result
                    in_chunk = out[it : it + chunk]
                    res_chunk = -coeff * in_chunk + delayed
                    buf_update = in_chunk + coeff * res_chunk
                    
                    # 3. Store back into circular buffer
                    buf[ptr:end_idx] = buf_update[:first_part]
                    if first_part < chunk:
                        buf[:chunk - first_part] = buf_update[first_part:]
                    
                    result[it : it + chunk] = res_chunk
                    ptr = (ptr + chunk) % blen
                    it += chunk
                
                state['ptr'] = ptr
                out = result
            return out

        # Use different delay orderings for L vs R to maximize decorrelation
        out_L = _process_channel(sig_L, 'L')
        out_R = _process_channel(sig_R, 'R')
        return out_L, out_R

    # ─── #20 Frequency-Dependent Stereo Width ─────────────────────────────
    def _apply_freq_dep_width(self, block):
        """Mono bass (<200 Hz), untouched mids, widened highs (>8 kHz)."""
        if block.shape[1] < 2:
            return
        sr = self.sample_rate

        if not hasattr(self, '_fdw_lo_sos'):
            self._fdw_lo_sos = butter(2, 200.0 / (sr / 2), btype='low', output='sos')
            self._fdw_hi_sos = butter(2, 8000.0 / (sr / 2), btype='high', output='sos')
            self._fdw_zi = {}

        L = block[:, 0].copy()
        R = block[:, 1].copy()

        for tag, sos, sig in [('lo_L', self._fdw_lo_sos, L), ('lo_R', self._fdw_lo_sos, R),
                               ('hi_L', self._fdw_hi_sos, L), ('hi_R', self._fdw_hi_sos, R)]:
            if tag not in self._fdw_zi:
                self._fdw_zi[tag] = sosfilt_zi(sos) * 0.0
            filtered, self._fdw_zi[tag] = sosfilt(sos, sig, zi=self._fdw_zi[tag])
            if tag == 'lo_L': lo_L = filtered
            elif tag == 'lo_R': lo_R = filtered
            elif tag == 'hi_L': hi_L = filtered
            elif tag == 'hi_R': hi_R = filtered

        mid_L = L - lo_L - hi_L
        mid_R = R - lo_R - hi_R

        # Bass: collapse to mono
        lo_mono = (lo_L + lo_R) * 0.5

        # Highs: enhance stereo width by 40 %
        hi_mid = (hi_L + hi_R) * 0.5
        hi_side = (hi_L - hi_R) * 0.5 * 1.4
        hi_L_w = hi_mid + hi_side
        hi_R_w = hi_mid - hi_side

        block[:, 0] = lo_mono + mid_L + hi_L_w
        block[:, 1] = lo_mono + mid_R + hi_R_w

    # ─── #21 Headphone Externalization ────────────────────────────────────
    def _apply_hp_externalization(self, block):
        """Crossfeed + early reflections + pinna notch for out-of-head imaging."""
        if block.shape[1] < 2:
            return
        sr = self.sample_rate
        frames = block.shape[0]

        if not hasattr(self, '_ext_xfeed_sos'):
            # Head-shadow crossfeed LP at 700 Hz
            self._ext_xfeed_sos = butter(2, 700.0 / (sr / 2), btype='low', output='sos')
            self._ext_xfeed_zi_L = sosfilt_zi(self._ext_xfeed_sos) * 0.0
            self._ext_xfeed_zi_R = sosfilt_zi(self._ext_xfeed_sos) * 0.0

            # Pinna notch at 8.5 kHz (Q ≈ 5)
            w0 = 8500.0 / (sr / 2)
            if w0 < 1.0:
                b_n, a_n = iirnotch(w0, 5.0)
            else:
                b_n, a_n = np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
            self._ext_pinna_b = b_n
            self._ext_pinna_a = a_n
            self._ext_pinna_zi_L = np.zeros(max(len(b_n), len(a_n)) - 1, dtype=np.float64)
            self._ext_pinna_zi_R = np.zeros(max(len(b_n), len(a_n)) - 1, dtype=np.float64)

            # Early-reflection delay buffer (50 ms max)
            buf_sz = int(sr * 0.05)
            self._ext_delay_buf = np.zeros((buf_sz, 2), dtype=np.float32)
            self._ext_delay_ptr = 0
            self._ext_reflections = [
                (int(sr * 0.006), 0.15),
                (int(sr * 0.013), 0.10),
                (int(sr * 0.021), 0.06),
            ]

        L = block[:, 0].copy()
        R = block[:, 1].copy()

        # 1. Crossfeed (opposite channel through head-shadow LP, mixed at -8 dB)
        xf_L, self._ext_xfeed_zi_L = sosfilt(self._ext_xfeed_sos, R, zi=self._ext_xfeed_zi_L)
        xf_R, self._ext_xfeed_zi_R = sosfilt(self._ext_xfeed_sos, L, zi=self._ext_xfeed_zi_R)
        block[:, 0] = L - xf_L * 0.40
        block[:, 1] = R - xf_R * 0.40

        # 2. Early reflections (cross-channel, precedence-effect room cue)
        buf_len = self._ext_delay_buf.shape[0]
        w_idx = (self._ext_delay_ptr + np.arange(frames)) % buf_len
        self._ext_delay_buf[w_idx, 0] = block[:, 0]
        self._ext_delay_buf[w_idx, 1] = block[:, 1]
        for d_smp, g in self._ext_reflections:
            r_idx = (w_idx - d_smp) % buf_len
            block[:, 0] += self._ext_delay_buf[r_idx, 1] * g
            block[:, 1] += self._ext_delay_buf[r_idx, 0] * g
        self._ext_delay_ptr = (self._ext_delay_ptr + frames) % buf_len

        # 3. Pinna coloration (subtle 8.5 kHz notch)
        block[:, 0], self._ext_pinna_zi_L = lfilter(
            self._ext_pinna_b, self._ext_pinna_a, block[:, 0], zi=self._ext_pinna_zi_L)
        block[:, 1], self._ext_pinna_zi_R = lfilter(
            self._ext_pinna_b, self._ext_pinna_a, block[:, 1], zi=self._ext_pinna_zi_R)

    def _apply_clarity_dsp(self, block):
        """Dynamic De-Mudding: Attenuates 200-500Hz when energy in that band is excessive."""
        if not getattr(self, 'clarity_dsp', False):
            return
            
        sr = self.sample_rate
        frames = block.shape[0]
        n_ch = block.shape[1]
        
        # 1. Detect mud activity (L/R only for efficiency)
        detect_ch = min(n_ch, 2)
        
        if not hasattr(self, '_clarity_mud_zi'):
            self._clarity_mud_zi = {}
            
        # Detect energy in 200-500Hz band
        mud_sig = np.zeros((frames, detect_ch), dtype=np.float32)
        for ch in range(detect_ch):
            if ch not in self._clarity_mud_zi:
                self._clarity_mud_zi[ch] = sosfilt_zi(self._clarity_mud_sos) * 0
            mud_sig[:, ch], self._clarity_mud_zi[ch] = sosfilt(self._clarity_mud_sos, block[:, ch], zi=self._clarity_mud_zi[ch])
            
        mud_rms = np.sqrt(np.mean(mud_sig**2) + 1e-12)
        bb_rms = np.sqrt(np.mean(block[:, :detect_ch]**2) + 1e-12)
        
        # Ratio of mud to broadband
        ratio = mud_rms / bb_rms
        
        # Threshold: if mud is more than 40% of broadband energy (log scale roughly)
        threshold = 0.40
        sensitivity = 2.0 # Response aggressiveness
        
        target_attenuation = 0.0
        if ratio > threshold:
            # Scale of 0.0 to 0.5 (where 0.5 is -6dB at 350Hz)
            target_attenuation = np.clip((ratio - threshold) * sensitivity, 0.0, 0.5)
            
        # Smooth the gain changes (tau = 80ms)
        alpha = 1.0 - np.exp(-frames / (sr * 0.08))
        self._clarity_gain_smooth = self._clarity_gain_smooth * (1.0 - alpha) + target_attenuation * alpha
        
        if self._clarity_gain_smooth > 0.001:
            for ch in range(n_ch):
                if ch not in self._clarity_zi:
                    self._clarity_zi[ch] = sosfilt_zi(self._clarity_peak_sos) * 0
                
                notched_sig, self._clarity_zi[ch] = sosfilt(self._clarity_peak_sos, block[:, ch], zi=self._clarity_zi[ch])
                
                # Dynamic blend: out = sig * (1 - blend) + notched * blend
                block[:, ch] = block[:, ch] * (1.0 - self._clarity_gain_smooth) + notched_sig * self._clarity_gain_smooth

    def _apply_virtual_room(self, block):
        """Virtual Room: Simulates early reflections (3-20ms) for enhanced spatiality."""
        if not getattr(self, 'virtual_room', False):
            return
            
        sr = self.sample_rate
        frames = block.shape[0]
        n_ch = block.shape[1]
        
        # 1. Initialize buffer (50ms max)
        buf_sz = int(sr * 0.05)
        if self._room_delay_buf is None or self._room_delay_buf.shape[0] < buf_sz or self._room_delay_buf.shape[1] != n_ch:
            self._room_delay_buf = np.zeros((buf_sz, n_ch), dtype=np.float32)
            self._room_delay_ptr = 0
            
        # 2. Define Taps (Time in ms, Gain, L/R Swap)
        taps = [
            (4.3, 0.45, False),  # Strong first reflection
            (7.1, 0.35, True),   # Cross-channel reflection
            (11.4, 0.25, False), # Secondary
            (17.8, 0.15, True),  # Late early-reflection
        ]
        
        # Convert ms to samples
        tap_samples = [(int(ms * sr / 1000.0), g, swap) for ms, g, swap in taps]
        
        # 3. Process Block
        buf_len = self._room_delay_buf.shape[0]
        w_idx = (self._room_delay_ptr + np.arange(frames)) % buf_len
        
        # Write current block to delay buffer
        self._room_delay_buf[w_idx, :] = block
        
        # Accumulate reflections
        reflections = np.zeros_like(block)
        for d_smp, g, swap in tap_samples:
            r_idx = (w_idx - d_smp) % buf_len
            if swap and n_ch >= 2:
                reflections[:, 0] += self._room_delay_buf[r_idx, 1] * g
                reflections[:, 1] += self._room_delay_buf[r_idx, 0] * g
                if n_ch > 2:
                    reflections[:, 2:] += self._room_delay_buf[r_idx, 2:] * (g * 0.5)
            else:
                reflections += self._room_delay_buf[r_idx, :] * g
                
        # 4. Apply Damping (LPF) to reflections
        for ch in range(n_ch):
            if ch not in self._room_lp_zi:
                self._room_lp_zi[ch] = sosfilt_zi(self._room_lp_sos) * 0
            reflections[:, ch], self._room_lp_zi[ch] = sosfilt(self._room_lp_sos, reflections[:, ch], zi=self._room_lp_zi[ch])
            
        # 5. Mix back
        mix = getattr(self, 'virtual_room_mix', 0.25)
        # Normalize sum of reflections (Sum of gains ~ 1.2)
        reflections *= 0.5 
        
        block += reflections * mix
        
        # Update pointer
        self._room_delay_ptr = (self._room_delay_ptr + frames) % buf_len

    def _apply_air_exciter(self, block):
        """Harmonic Air Exciter: Restores/Enhances 10kHz+ detail using non-linear saturation."""
        if not getattr(self, 'air_exciter', False):
            return
            
        n_ch = block.shape[1]
        # Use a copy for the side-chain to avoid modifying the original block before summing
        excited = np.zeros_like(block)
        
        # 1. High-pass filter
        for ch in range(n_ch):
            if ch not in self._air_hp_zi:
                self._air_hp_zi[ch] = sosfilt_zi(self._air_hp_sos) * 0
            excited[:, ch], self._air_hp_zi[ch] = sosfilt(self._air_hp_sos, block[:, ch], zi=self._air_hp_zi[ch])
            
        # 2. Architect's "Verified" Soft-Knee Saturation (Vectorized)
        threshold = 0.8
        abs_x = np.abs(excited)
        mask = abs_x < threshold
        
        # Region 1: Normal
        excited[mask] = excited[mask] * (1.0 + 0.5 * (excited[mask]**2))
        
        # Region 2: Knee (Architect's else branch implementation)
        else_mask = ~mask
        if np.any(else_mask):
             # To prevent explosion, we'll cap the x^2 term at 1.0 for the saturation path
             # and ensure the transition is relatively smooth.
             sign = np.sign(excited[else_mask])
             # Minimal gain increase to avoid clipping artifacts
             excited[else_mask] = sign * threshold * (1.05) 
             
        # 3. Mix
        amount = getattr(self, 'air_amount', 0.3)
        block += excited * amount

    def _apply_post_effects(self, out_block, out_ch, skip_subsynth=False, src_block=None, is_passthrough=False):
        """Apply post-upmix effects pipeline to the output block."""
        frames = out_block.shape[0]

        # ── Program Level Estimator (always runs, measures only) ──
        self._update_program_level(out_block)

        # ── Input Restoration (runs first, fixes damaged source) ──
        # #19 Declipping
        if getattr(self, 'declip', False):
            self._apply_declip(out_block)

        # #17 Loudness Normalization (AGC)
        if getattr(self, 'loudness_norm', False):
            self._apply_loudness_norm(out_block)

        # Calculate overall level for loudness compensation
        rms = np.sqrt(np.mean(out_block**2) + 1e-12)

        # #14 Fletcher-Munson Loudness Compensation (Upgrade for Dynamic EQ)
        if getattr(self, 'dynamic_eq', False):
            self._apply_loudness_comp(out_block, rms)

        # #24 Dynamic Clarity (De-Mudding)
        if getattr(self, 'clarity_dsp', False):
            self._apply_clarity_dsp(out_block)

        # #25 Virtual Room (Early Reflections)
        if getattr(self, 'virtual_room', False):
            self._apply_virtual_room(out_block)

        # #26 Harmonic Air Exciter
        if getattr(self, 'air_exciter', False):
            self._apply_air_exciter(out_block)

        # #16 Dynamic Dialogue Intelligibility (standalone, skipped when vocal_lock
        #      is active since vocal_lock already applies its own adaptive boost)
        if getattr(self, 'dialogue_focus', False) and not getattr(self, 'vocal_lock', False) and out_ch > 2:
            self._apply_dialogue_focus(out_block)

        # #13 Harmonic Bass Exciter (PBE) - Standalone toggle
        if getattr(self, 'harmonic_exciter', False):
            self._apply_harmonic_exciter(out_block)

        # #20 Frequency-Dependent Stereo Width
        if getattr(self, 'freq_dep_width', False):
            self._apply_freq_dep_width(out_block)

        # #2 Ambient Reverb Extraction


        # #5 Vocal Lock to Center
        if self.vocal_lock and out_ch > 2:
            self._apply_vocal_lock(out_block, frames)

        # #11 Subharmonic Synthesizer (Skip in passthrough)
        if self.sub_synth and not skip_subsynth:
            self._apply_subharmonic_synth(out_block, frames, out_ch, src_block=src_block)



        # #9 Spectral Tilt Correction
        if self.spectral_tilt:
            self._apply_spectral_tilt(out_block, frames, out_ch)

        # #23 Micro-Dynamics / Upward Compression (lifts quiet detail)
        # Snapshot before micro-dyn so transient restore can measure original dynamics
        pre_micro_dyn = out_block.copy() if (getattr(self, 'micro_dynamics', False)
                                              and getattr(self, 'transient_restore', False)) else None
        if getattr(self, 'micro_dynamics', False):
            self._apply_micro_dynamics(out_block)

        # #18 Transient Restoration (late — enhances final dynamics)
        # When micro-dyn is active, measure envelopes from pre-micro-dyn signal
        # so transient detection isn't fooled by the upward compression
        if getattr(self, 'transient_restore', False):
            self._apply_transient_restore(out_block, ref_block=pre_micro_dyn)

        # #12 Transaural Widening
        if getattr(self, 'transaural_widening', False):
            self._apply_transaural_widener(out_block, frames)

        # #21 Headphone Externalization (after transaural, for HP mode)
        if getattr(self, 'hp_externalization', False):
            self._apply_hp_externalization(out_block)

        # #1 Independent Head Tracking Volume Panning (MUST RUN LAST for Absolute Room Alignment)
        self._apply_head_tracking_panning(out_block, frames, out_ch, is_passthrough=is_passthrough)

    def _apply_transaural_widener(self, out_block, frames):
        """Pure-Python Crosstalk Cancellation with Dynamic Head Tracking circular buffer."""
        N = frames
        
        # ─── Dynamic Head Tracking Coefficients ───
        delay_L = float(getattr(self, 'transaural_delay', 7))
        delay_R = delay_L
        gain_ff_L = float(getattr(self, 'transaural_gain', 0.35))
        gain_ff_R = gain_ff_L
        gain_fb = float(getattr(self, 'transaural_feedback', 0.08))
        
        # Base compensation
        comp_L = 1.0
        comp_R = 1.0
        
        if getattr(self, 'head_tracking', False) and hasattr(self, 'head_tracker') and self.head_tracker:
            pos_x = self.head_tracker.pos_x # Range -0.5 to 0.5
            pos_z = self.head_tracker.pos_z # ~1.0
            
            S_WIDTH = getattr(self, 'speaker_width_cm', 60.0)
            S_DIST = getattr(self, 'speaker_distance_cm', 70.0)
            # Speed of sound standard
            cm_per_sample = 34300.0 / self.sample_rate
            
            # Map head coords to offsets
            user_z = S_DIST * pos_z # Relative depth scaling
            camera_fov = getattr(self, 'camera_fov', 70.0)
            fov_mult = 2.0 * np.tan(np.radians(camera_fov / 2.0))
            camera_view_width_cm = user_z * fov_mult
            user_x = pos_x * camera_view_width_cm
            
            yaw = self.head_tracker.rot_yaw
            offset_x = 8.5 * np.cos(yaw) # Correct Ear spacer for rotation
            
            # Distance from ears to speakers (approx 17cm head spacer)
            x_L_L = user_x - offset_x - (-S_WIDTH/2)
            dist_L_L = np.sqrt(x_L_L**2 + user_z**2) # speaker L -> ear L
            
            x_L_R = user_x + offset_x - (-S_WIDTH/2)
            dist_L_R = np.sqrt(x_L_R**2 + user_z**2) # speaker L -> ear R
            
            x_R_R = user_x + offset_x - (+S_WIDTH/2)
            dist_R_R = np.sqrt(x_R_R**2 + user_z**2) # speaker R -> ear R
            
            x_R_L = user_x - offset_x - (+S_WIDTH/2)
            dist_R_L = np.sqrt(x_R_L**2 + user_z**2) # speaker R -> ear L
            
            # Crossover ITD path delays
            delay_L = max(1, (dist_R_L - dist_R_R) / cm_per_sample)
            delay_R = max(1, (dist_L_R - dist_L_L) / cm_per_sample)
            
            gain_ff_L = gain_ff_L * (S_DIST / dist_R_L)
            gain_ff_R = gain_ff_R * (S_DIST / dist_L_R)

            # --- Asymmetric XTC (Nose-Shadowing Compensation) ---
            angle_L = np.abs(np.arctan2(user_x - (-S_WIDTH/2), user_z))
            angle_R = np.abs(np.arctan2(user_x - (+S_WIDTH/2), user_z))
            eff_angle_L = np.abs(angle_L + yaw)
            eff_angle_R = np.abs(angle_R - yaw)
            
            # Map to blend factor (Starts at 15 deg ~= 0.26 rad, max at 45 deg ~= 0.78 rad)
            blend_L = np.clip((eff_angle_L - 0.26) / 0.52, 0.0, 0.60) # Max 60% reduction
            blend_R = np.clip((eff_angle_R - 0.26) / 0.52, 0.0, 0.60)
            
            # Attenuate cancellation for paths shadowed by head/nose
            gain_ff_R *= (1.0 - blend_L) 
            gain_ff_L *= (1.0 - blend_R)


        # Smooth State Variables to avoid zi/pointer pops
        if not hasattr(self, '_trans_delay_smooth_L'):
             self._trans_delay_smooth_L = delay_L
             self._trans_delay_smooth_R = delay_R
        else:
             self._trans_delay_smooth_L = self._trans_delay_smooth_L * 0.60 + delay_L * 0.40
             self._trans_delay_smooth_R = self._trans_delay_smooth_R * 0.60 + delay_R * 0.40

        d_L_smooth = self._trans_delay_smooth_L
        d_R_smooth = self._trans_delay_smooth_R
        
        d_L = int(d_L_smooth)
        d_R = int(d_R_smooth)
        
        frac_L = d_L_smooth - d_L
        frac_R = d_R_smooth - d_R

        if d_L <= 0 or d_R <= 0:
            return
            
        if not hasattr(self, '_trans_delay_L') or self._trans_delay_L.shape[0] < frames + 2048:
            self._trans_delay_L = np.zeros(8192, dtype=np.float32)
            self._trans_delay_R = np.zeros(8192, dtype=np.float32)
            self._trans_out_L = np.zeros(8192, dtype=np.float32)
            self._trans_out_R = np.zeros(8192, dtype=np.float32)
            self._trans_delay_ptr = 0
            
        buf_len = self._trans_delay_L.shape[0]
        L = out_block[:N, 0].copy()
        R = out_block[:N, 1].copy()
        if not hasattr(self, '_trans_lp_zi_L'):
             self._trans_lp_zi_L = sosfilt_zi(self._trans_lp_sos)
             self._trans_lp_zi_R = sosfilt_zi(self._trans_lp_sos)
             
        L_lp, self._trans_lp_zi_L = sosfilt(self._trans_lp_sos, L, zi=self._trans_lp_zi_L)
        R_lp, self._trans_lp_zi_R = sosfilt(self._trans_lp_sos, R, zi=self._trans_lp_zi_R)
        
        for i in range(N):
            w_idx = (self._trans_delay_ptr + i) % buf_len
            r_idx_L = (self._trans_delay_ptr + i - d_L) % buf_len
            r_idx_R = (self._trans_delay_ptr + i - d_R) % buf_len
            
            r_L_1 = (r_idx_L - 1) % buf_len
            r_R_1 = (r_idx_R - 1) % buf_len
            
            self._trans_delay_L[w_idx] = L_lp[i]
            self._trans_delay_R[w_idx] = R_lp[i]
            
            # Interpolated In (Delayed cross-paths)
            val_in_L = self._trans_delay_L[r_idx_R] * (1 - frac_R) + self._trans_delay_L[r_R_1] * frac_R
            val_in_R = self._trans_delay_R[r_idx_L] * (1 - frac_L) + self._trans_delay_R[r_L_1] * frac_L
            
            # Interpolated Out (Feedback paths)
            val_out_L = self._trans_out_L[r_idx_R] * (1 - frac_R) + self._trans_out_L[r_R_1] * frac_R
            val_out_R = self._trans_out_R[r_idx_L] * (1 - frac_L) + self._trans_out_R[r_L_1] * frac_L
            
            calc_L = L[i] - (gain_ff_L * val_in_R) - (gain_fb * val_out_R)
            calc_R = R[i] - (gain_ff_R * val_in_L) - (gain_fb * val_out_L)
            
            self._trans_out_L[w_idx] = calc_L
            self._trans_out_R[w_idx] = calc_R
            
            out_block[i, 0] = calc_L
            out_block[i, 1] = calc_R

        self._trans_delay_ptr = (self._trans_delay_ptr + N) % buf_len



    def _apply_vocal_lock(self, out_block, frames):
        """Detect vocals in center channel and boost center extraction."""
        N = min(frames, 2048)
        center = out_block[:N, 2]
        
        if np.max(np.abs(center)) < 1e-6:
            return
            
        fft_c = np.fft.rfft(center)
        mag = np.abs(fft_c)
        
        log_mag = np.log(mag + 1e-12)
        geom_mean = np.exp(np.mean(log_mag))
        arith_mean = np.mean(mag)
        flatness = geom_mean / (arith_mean + 1e-12)
        
        n_mag = len(mag)
        if not hasattr(self, '_vl_voice_mask') or len(self._vl_voice_mask) != n_mag:
            freq_bins = np.linspace(0, self.sample_rate / 2, n_mag)
            self._vl_voice_mask = ((freq_bins >= 100) & (freq_bins <= 4000)).astype(np.float32)
        voice_mask = self._vl_voice_mask
        voice_mag = mag * voice_mask
        voice_energy = np.sum(voice_mag) / (np.sum(mag) + 1e-10)
        
        vocal_score = (1.0 - flatness) * voice_energy
        alpha = 0.15
        self._vl_prev_vocal_score = self._vl_prev_vocal_score * (1 - alpha) + vocal_score * alpha
        
        boost = 1.0 + self._vl_prev_vocal_score * 0.414  # Up to +3dB max
        ramp = self._get_ramp('_last_vl_boost', boost, N)
        out_block[:N, 2] *= ramp



    def _apply_spectral_tilt(self, out_block, frames, out_ch):
        """Adaptive Equalization with Overlap-and-Add Carry Buffers."""
        N = frames
        if N < 256: return
        
        # 1. Compute Mono Contour from All Channels
        if not hasattr(self, '_tilt_mono_pre') or self._tilt_mono_pre.shape != (N,):
            self._tilt_mono_pre = np.zeros(N, dtype=np.float32)
            self._tilt_padded_pre = np.zeros(2 * N, dtype=np.float32)
            
        mono = self._tilt_mono_pre
        mono.fill(0)
        
        active = 0
        for ch in [0, 1, 2, 4, 5]: # FL, FR, C, SL, SR (Skip LFE 3)
            if ch < out_ch:
                mono += out_block[:N, ch]
                active += 1
        if active > 0:
            mono /= active
        
        padded_mono = self._tilt_padded_pre
        padded_mono.fill(0)
        padded_mono[:N] = mono
        
        fft_mono = np.fft.rfft(padded_mono)
        mag = np.abs(fft_mono)
        
        if np.max(mag) < 1e-6:
            return
            
        smooth_mag = gaussian_filter1d(mag, sigma=25.0)
        
        # 2. Gain Calculation (Human Preference / Harman-Style Target)
        freqs = np.linspace(0, self.sample_rate / 2, len(smooth_mag))
        target = np.ones_like(freqs)
        
        # Bass boost below 120Hz (+3dB)
        bass_mask = freqs <= 120
        target[bass_mask] = 1.414
        
        # Mid slope from 120Hz to 1000Hz
        mid_mask = (freqs > 120) & (freqs <= 1000)
        target[mid_mask] = np.interp(freqs[mid_mask], [120, 1000], [1.414, 1.0])
        
        # Treble roll-off above 1000Hz (-1.5dB per octave)
        treble_mask = freqs > 1000
        octaves = np.log2(freqs[treble_mask] / 1000.0)
        target[treble_mask] = 10 ** (-1.5 * octaves / 20.0)
        
        # Regularized Inversion (Soft Waterfilling) to avoid boosting noise floor
        reg = 0.02 * np.max(smooth_mag)  # 2% of max magnitude peak
        gain_raw = target / (smooth_mag + reg)

        
        # Normalize to prevent scale shifts
        gain_raw /= (np.mean(gain_raw) + 1e-6)
        
        if not hasattr(self, '_tilt_filter_zi') or not isinstance(self._tilt_filter_zi, dict):
             self._tilt_filter_zi = {}
             
        if '_tilt_gain_smooth' not in self._tilt_filter_zi:
            self._tilt_filter_zi['_tilt_gain_smooth'] = gain_raw.copy()
        else:
            self._tilt_filter_zi['_tilt_gain_smooth'] = self._tilt_filter_zi['_tilt_gain_smooth'] * 0.995 + gain_raw * 0.005
            
        # Prevent treble harshness by tapered maximum boost clamp
        max_gain = np.ones_like(freqs) * 2.0  # Max +6dB absolute boost
        treble_boost_mask = freqs > 1500
        # Taper allowed boost from 1500Hz to 2500Hz down to 1.0 (0dB boost)
        max_gain[treble_boost_mask] = np.interp(freqs[treble_boost_mask], [1500, 2500], [2.0, 1.0])
        max_gain[freqs > 2500] = 1.0 # Strictly prevent ANY boost above 2.5kHz to eliminate harshness
        
        gain = np.clip(self._tilt_filter_zi['_tilt_gain_smooth'], 0.3, max_gain)


        
        # ─── Optimized: Design FIR taps only every 32 blocks to save CPU ───
        if not hasattr(self, '_tilt_update_cnt'):
            self._tilt_update_cnt = 0
            self._tilt_taps = None

        if self._tilt_taps is None or self._tilt_update_cnt % 32 == 0:
            num_taps = 255
            norm_freqs = np.linspace(0, 1.0, len(gain))
            self._tilt_taps = firwin2(num_taps, norm_freqs, gain)
        
        self._tilt_update_cnt += 1
        taps = self._tilt_taps
        
        # 4. Filter all channels with stateful continuous convolution
        for ch in range(min(out_ch, 6)):
            sig = out_block[:N, ch].copy()
            
            key_zi = f'tilt_zi_{ch}'
            if key_zi not in self._tilt_filter_zi:
                self._tilt_filter_zi[key_zi] = lfilter_zi(taps, [1.0]) * 0
                
            filtered, self._tilt_filter_zi[key_zi] = lfilter(taps, [1.0], out_block[:N, ch], zi=self._tilt_filter_zi[key_zi])
            out_block[:N, ch] = filtered.astype(np.float32)
            
            # Continuous Energy Match Level Lock with Linear Ramping
            energy_in = np.sum(sig**2) + 1e-12
            energy_out = np.sum(out_block[:N, ch]**2) + 1e-12
            scale_fact = np.clip(np.sqrt(energy_in / energy_out), 0.3, 3.0)
            
            key_scale = f'tilt_prev_scale_{ch}'
            if key_scale not in self._tilt_filter_zi:
                self._tilt_filter_zi[key_scale] = scale_fact
                
            prev_scale = self._tilt_filter_zi[key_scale]
            ramp = np.linspace(prev_scale, scale_fact, N, dtype=np.float32)
            out_block[:N, ch] *= ramp
            self._tilt_filter_zi[key_scale] = scale_fact

    def _apply_subharmonic_synth(self, out_block, frames, out_ch, src_block=None):
        """Generate clean subharmonic (f/2) via autocorrelation pitch tracking + sine oscillator."""
        N = frames
        if N < 1: return

        # 1. Source Signal (Mono sum of input or output fronts)
        if src_block is not None:
            src = (src_block[:N, 0] + src_block[:N, 1]) * 0.5
        else:
            src = (out_block[:N, 0] + out_block[:N, 1]) * 0.5

        # 2. Filter states init
        if 'drive' not in self._sub_filter_zi or self._sub_vocal_sos is None:
            if self._sub_vocal_sos is None:
                self._init_multiband_filters()
            if self._sub_vocal_sos is not None:
                self._sub_filter_zi['drive'] = sosfilt_zi(self._sub_drive_sos) * 0
                self._sub_filter_zi['vocal'] = sosfilt_zi(self._sub_vocal_sos) * 0
                self._sub_filter_zi['out_lp'] = sosfilt_zi(self._sub_out_sos) * 0
                self._sub_filter_zi['hp20'] = sosfilt_zi(self._sub_hp_sos) * 0
                self._sub_filter_zi['v_avg'] = 1e-12
                self._sub_filter_zi['d_avg'] = 1e-12
            else:
                return

        # 3. Drive bandpass + vocal detection
        drive_sig, self._sub_filter_zi['drive'] = sosfilt(self._sub_drive_sos, src, zi=self._sub_filter_zi['drive'])
        vocal_sig, self._sub_filter_zi['vocal'] = sosfilt(self._sub_vocal_sos, src, zi=self._sub_filter_zi['vocal'])

        # 4. Voice Avoidance (stateful smoothing)
        v_pow = np.mean(vocal_sig**2)
        d_pow = np.mean(drive_sig**2)
        self._sub_filter_zi['v_avg'] = self._sub_filter_zi['v_avg'] * 0.98 + v_pow * 0.02
        self._sub_filter_zi['d_avg'] = self._sub_filter_zi['d_avg'] * 0.98 + d_pow * 0.02

        vocal_energy = np.sqrt(self._sub_filter_zi['v_avg'])
        drive_energy = np.sqrt(self._sub_filter_zi['d_avg'])

        ratio = vocal_energy / (drive_energy + 1e-6)
        cutoff = getattr(self, '_sub_drive_cutoff', 80.0)
        avoid_thresh = np.interp(cutoff, [40.0, 80.0, 200.0], [1.1, 0.9, 0.5])
        avoid_slope  = np.interp(cutoff, [40.0, 80.0, 200.0], [1.0, 1.5, 2.5])
        avoidance = float(np.clip(1.0 - (ratio - avoid_thresh) * avoid_slope, 0.0, 1.0))

        # 5. Autocorrelation pitch detection on accumulated analysis buffer
        # Bass periods (20-80Hz at 48kHz = 600-2400 samples) exceed typical block
        # sizes (256-512), so we accumulate drive signal into a longer window.
        sr = self.sample_rate
        low  = max(15.0, getattr(self, '_sub_low_cutoff', 20.0))
        high = max(low + 5.0, getattr(self, '_sub_drive_cutoff', 80.0))

        # Analysis buffer: 2.5× longest period for reliable autocorrelation
        analysis_len = max(4800, int(2.5 * sr / low))

        if not hasattr(self, '_sub_period_est'):
            self._sub_period_est = sr / 60.0
            self._sub_phase_acc = 0.0
            self._sub_env_state = 0.0
            self._sub_confidence = 0.0
            self._sub_analysis_buf = np.zeros(analysis_len, dtype=np.float32)

        # Shift buffer left and append new drive samples
        buf = self._sub_analysis_buf
        if buf.shape[0] != analysis_len:
            buf = np.zeros(analysis_len, dtype=np.float32)
        if N < analysis_len:
            buf[:-N] = buf[N:]
            buf[-N:] = drive_sig
        else:
            buf[:] = drive_sig[-analysis_len:]
        self._sub_analysis_buf = buf

        # Autocorrelation on the full analysis window
        A = analysis_len
        min_lag = max(1, int(sr / high))    # Shortest period (highest freq)
        max_lag = min(A - 1, int(sr / low)) # Longest period (lowest freq)

        if max_lag > min_lag:
            pad_len = 1
            while pad_len < 2 * A: pad_len <<= 1
            fft_d = np.fft.rfft(buf, n=pad_len)
            autocorr = np.fft.irfft(fft_d * np.conj(fft_d))[:A]

            if autocorr[0] > 1e-10:
                autocorr_norm = autocorr / autocorr[0]
                search = autocorr_norm[min_lag:max_lag]

                if len(search) > 0:
                    peak_val = float(np.max(search))

                    if peak_val > 0.20:  # Clear periodic content detected
                        peak_idx = min_lag + int(np.argmax(search))

                        # Parabolic interpolation for sub-sample accuracy
                        if 0 < peak_idx < A - 1:
                            a, b, c = autocorr_norm[peak_idx - 1], autocorr_norm[peak_idx], autocorr_norm[peak_idx + 1]
                            denom = 2.0 * b - a - c
                            refined = peak_idx + (0.5 * (a - c) / denom if denom > 1e-6 else 0.0)
                        else:
                            refined = float(peak_idx)

                        self._sub_period_est = self._sub_period_est * 0.80 + refined * 0.20

                    self._sub_confidence = self._sub_confidence * 0.85 + peak_val * 0.15
                else:
                    self._sub_confidence *= 0.85
            else:
                self._sub_confidence *= 0.85

        # 6. Synthesize clean sine at f/2
        # phase_increment = 2π × (f/2) / sr = 2π / (2 × period) = π / period
        phase_inc = np.pi / max(1.0, self._sub_period_est)
        phases = self._sub_phase_acc + np.arange(1, N + 1, dtype=np.float64) * phase_inc
        self._sub_phase_acc = float(phases[-1] % (2.0 * np.pi))
        sub_sig = np.sin(phases).astype(np.float32)

        # 7. Spectral gap detection: only synthesize what's missing
        # Measure energy at the fundamental (f) and its subharmonic (f/2).
        # If f/2 is already present in the mix, reduce synthesis proportionally.
        detected_f = sr / max(1.0, self._sub_period_est)
        sub_f = detected_f / 2.0

        # Use the analysis buffer for spectral measurement (already accumulated)
        spec = np.abs(np.fft.rfft(buf * np.hanning(len(buf))))
        freq_res = sr / len(buf)

        # Energy in a ±20% band around fundamental and subharmonic
        def _band_energy(center_hz):
            lo = max(0, int(center_hz * 0.8 / freq_res))
            hi = min(len(spec) - 1, int(center_hz * 1.2 / freq_res)) + 1
            return float(np.mean(spec[lo:hi]**2)) if hi > lo else 0.0

        e_fund = _band_energy(detected_f)
        e_sub  = _band_energy(sub_f)

        # Need ratio: how much subharmonic is already present vs fundamental
        # 1.0 = sub is as strong as fundamental (no synthesis needed)
        # 0.0 = sub is absent (full synthesis)
        presence = min(1.0, (e_sub / (e_fund + 1e-12)) ** 0.5)
        need = 1.0 - presence  # 0 = already there, 1 = completely missing

        if not hasattr(self, '_sub_need_smooth'):
            self._sub_need_smooth = need
        else:
            self._sub_need_smooth = self._sub_need_smooth * 0.90 + need * 0.10

        # 8. Amplitude envelope from smoothed drive energy (Vectorized)
        drive_env = np.abs(drive_sig)
        alpha_env = 0.97
        b_env, a_env = [1.0 - alpha_env], [1.0, -alpha_env]
        zi = np.array([alpha_env * self._sub_env_state], dtype=np.float32)
        env_out, zi_new = lfilter(b_env, a_env, drive_env, zi=zi)
        self._sub_env_state = float(zi_new[0] / alpha_env) if alpha_env != 0 else 0.0

        # Scale by need: full output when subharmonic is missing, zero when already present
        sub_sig *= env_out * 6.0 * self._sub_need_smooth

        # Gate: fade out when autocorrelation confidence is low (noise / no pitch)
        conf_gate = float(np.clip(self._sub_confidence * 3.0 - 0.5, 0.0, 1.0))
        sub_sig *= conf_gate

        # 8. Output LP smoothing + 20Hz safety HP
        sub_sig, self._sub_filter_zi['out_lp'] = sosfilt(self._sub_out_sos, sub_sig, zi=self._sub_filter_zi['out_lp'])
        sub_sig, self._sub_filter_zi['hp20'] = sosfilt(self._sub_hp_sos, sub_sig, zi=self._sub_filter_zi['hp20'])

        # 9. Apply Gain and Avoidance ducking
        if not hasattr(self, '_sub_last_avoid'): self._sub_last_avoid = 1.0
        avoid_ramp = np.linspace(self._sub_last_avoid, avoidance, N, dtype=np.float32)
        final_sub = sub_sig * self.sub_gain * avoid_ramp
        self._sub_last_avoid = avoidance

        # 10. Route to LFE and Fronts
        if out_ch > 3 and not self.lfe_muted:
            out_block[:N, 3] += final_sub * 1.58  # Main LFE

        # Always add to Mains (headphone compat + 2-ch meters)
        out_block[:N, 0] += final_sub
        out_block[:N, 1] += final_sub



    @property
    def sub_drive_cutoff(self):
        return getattr(self, '_sub_drive_cutoff', 80.0)
        
    @sub_drive_cutoff.setter
    def sub_drive_cutoff(self, value):
        self._sub_drive_cutoff = float(value)
        self._rebuild_sub_drive_filter()

    @property
    def sub_low_cutoff(self):
        return getattr(self, '_sub_low_cutoff', 20.0)

    @sub_low_cutoff.setter
    def sub_low_cutoff(self, value):
        self._sub_low_cutoff = float(value)
        self._rebuild_sub_drive_filter()

    def _rebuild_sub_drive_filter(self):
        """Rebuild the subharmonic drive bandpass when either cutoff changes."""
        sr = getattr(self, 'sample_rate', 48000)
        low  = max(15.0, getattr(self, '_sub_low_cutoff', 20.0))
        high = max(low + 5.0, getattr(self, '_sub_drive_cutoff', 80.0))
        self._sub_drive_sos = butter(2, [low / (sr / 2), high / (sr / 2)], btype='bandpass', output='sos')
        # Reset filter state so the new coefficients don't pop
        if hasattr(self, '_sub_filter_zi') and 'drive' in self._sub_filter_zi:
            self._sub_filter_zi['drive'] = sosfilt_zi(self._sub_drive_sos) * 0

    def _apply_loudness_comp(self, block, level):
        """ISO 226:2003 inspired adaptive loudness compensation."""
        if not self.loudness_comp: return block
        
        # Calculate target gains based on average signal level (0.0 to 1.0)
        # We want more boost at lower levels.
        boost_db = max(0, (1.0 - level * 2.0) * 8.0) # Up to 8dB boost at -inf
        if boost_db < 0.1: return block
        
        # Simple Bass/Treble shelving filters
        sr = self.sample_rate
        # For performance, we only re-calculate coefficients if level changed significantly
        if not hasattr(self, '_last_loudness_db') or abs(self._last_loudness_db - boost_db) > 0.5:
            self._last_loudness_db = boost_db
            # Simplified Bass Shelf (100Hz) and High Shelf (8kHz)
            # This 'smile' curve helps maintain perceived transparency at low volumes
            A = 10**(boost_db/40) 
            w0_l = 2 * np.pi * 100 / sr
            alpha_l = np.sin(w0_l)/2 * np.sqrt((A + 1/A)*(1/0.707 - 1) + 2)
            self._loud_b_l = [ A*((A+1)-(A-1)*np.cos(w0_l)+2*np.sqrt(A)*alpha_l),
                               2*A*((A-1)-(A+1)*np.cos(w0_l)),
                               A*((A+1)-(A-1)*np.cos(w0_l)-2*np.sqrt(A)*alpha_l) ]
            self._loud_a_l = [ (A+1)+(A-1)*np.cos(w0_l)+2*np.sqrt(A)*alpha_l,
                               -2*((A-1)+(A+1)*np.cos(w0_l)),
                               (A+1)+(A-1)*np.cos(w0_l)-2*np.sqrt(A)*alpha_l ]
            self._loud_b_l = np.array(self._loud_b_l) / self._loud_a_l[0]
            self._loud_a_l = np.array(self._loud_a_l) / self._loud_a_l[0]

            # High Shelf (8kHz)
            w0_h = 2 * np.pi * 8000 / sr
            alpha_h = np.sin(w0_h)/2 * np.sqrt((A + 1/A)*(1/0.707 - 1) + 2)
            self._loud_b_h = [ A*((A+1)+(A-1)*np.cos(w0_h)+2*np.sqrt(A)*alpha_h),
                               -2*A*((A-1)+(A+1)*np.cos(w0_h)),
                               A*((A+1)+(A-1)*np.cos(w0_h)-2*np.sqrt(A)*alpha_h) ]
            self._loud_a_h = [ (A+1)-(A-1)*np.cos(w0_h)+2*np.sqrt(A)*alpha_h,
                               2*((A-1)-(A+1)*np.cos(w0_h)),
                               (A+1)-(A-1)*np.cos(w0_h)-2*np.sqrt(A)*alpha_h ]
            self._loud_b_h = np.array(self._loud_b_h) / self._loud_a_h[0]
            self._loud_a_h = np.array(self._loud_a_h) / self._loud_a_h[0]

        for ch in range(block.shape[1]):
            if ch == 3: continue # Don't boost LFE
            # Bass shelf
            zi_l = self._loudness_filter_zi.get(f"l_{ch}", np.zeros(2))
            block[:, ch], zi_l = lfilter(self._loud_b_l, self._loud_a_l, block[:, ch], zi=zi_l)
            self._loudness_filter_zi[f"l_{ch}"] = zi_l
            # High shelf
            zi_h = self._loudness_filter_zi.get(f"h_{ch}", np.zeros(2))
            block[:, ch], zi_h = lfilter(self._loud_b_h, self._loud_a_h, block[:, ch], zi=zi_h)
            self._loudness_filter_zi[f"h_{ch}"] = zi_h
        return block

    def _apply_harmonic_exciter(self, block):
        """Psychoacoustic Bass Exciter — missing-fundamental effect.

        Uses full-wave and half-wave rectification to generate 2nd and 3rd
        harmonics at a *fixed proportion* of the input level (no normalisation
        needed).  |sin(ωt)| always has its 2nd harmonic at 42 % of the
        fundamental — regardless of amplitude — so the effect is consistent
        from whisper-quiet to full-scale bass.

        A tight 80–500 Hz bandpass keeps only the low-order harmonics that
        trigger the auditory system's missing-fundamental percept, with zero
        mid/treble splatter.  Designed for non-distracting system-wide use.
        """
        if not self.harmonic_exciter:
            return block

        # -- 1. Lazy filter init ------------------------------------------------
        if not hasattr(self, '_exciter_lp_sos') or self._exciter_lp_sos is None:
            # LP  120 Hz — isolate bass that needs reinforcement
            self._exciter_lp_sos = butter(4, 120, 'lp',
                                          fs=self.sample_rate, output='sos')
            # HP   80 Hz — strip the original fundamental so only generated
            #              harmonics remain (keeps 2nd harmonic of 50 Hz = 100 Hz)
            self._exciter_hp_sos = butter(4, 80, 'hp',
                                          fs=self.sample_rate, output='sos')
            # LP  500 Hz — hard ceiling on generated harmonics; prevents any
            #              energy above the 4th-5th harmonic from leaking out
            self._exciter_tame_sos = butter(4, 500, 'lp',
                                            fs=self.sample_rate, output='sos')
            self._exciter_lp_zi = np.zeros((self._exciter_lp_sos.shape[0], 2))
            self._exciter_hp_zi = np.zeros((self._exciter_hp_sos.shape[0], 2))
            self._exciter_tame_zi = np.zeros((self._exciter_tame_sos.shape[0], 2))


        # -- 2. Extract bass drive (mono sum) -----------------------------------
        drive_mono = (block[:, 0] + block[:, 1]) * 0.5
        drive, self._exciter_lp_zi = sosfilt(
            self._exciter_lp_sos, drive_mono, zi=self._exciter_lp_zi)

        # -- 3. Rectification-based harmonic generation -------------------------
        #   Full-wave  |x|            → 2nd harmonic at 42 % of fundamental
        #   Half-wave  max(x, 0)      → 2nd + 3rd harmonics (asymmetric)
        #   Both scale linearly with input — no level-dependent behaviour.
        full_rect = np.abs(drive)                    # even harmonics (warm)
        half_rect = np.maximum(drive, 0.0)           # even + odd (body)
        harmonics = full_rect * 0.6 + half_rect * 0.4

        # -- 3b. Strip DC offset BEFORE the HP filter ----------------------------
        #   Rectification produces a large DC component (0.64×A for full-wave)
        #   that is bigger than the 2nd harmonic itself.  If the HP filter has
        #   to reject this DC on its own, its states need many cycles to settle
        #   → audible fade-out artefact.  Removing DC here lets the HP start in
        #   steady-state immediately.
        harmonics -= np.mean(harmonics)

        # -- 4. Bandpass: keep only the generated harmonics ---------------------
        # HP removes the original fundamental that leaked through rectification
        harmonics, self._exciter_hp_zi = sosfilt(
            self._exciter_hp_sos, harmonics, zi=self._exciter_hp_zi)
        # LP removes everything above ~500 Hz (no midrange / treble artefacts)
        harmonics, self._exciter_tame_zi = sosfilt(
            self._exciter_tame_sos, harmonics, zi=self._exciter_tame_zi)

        # -- 5. Mix back -------------------------------------------------------
        gain = getattr(self, 'harmonic_gain', 0.7)
        block[:, 0] += harmonics * gain
        block[:, 1] += harmonics * gain
        return block

    def _apply_true_peak_limiter(self, block):
        """Ultra-transparent look-ahead limiter to replace/complement the soft clipper."""
        # Always active as the final safety stage
        
        # 1.5ms look-ahead (approx 72 samples at 48kHz)
        look_ahead = int(0.0015 * self.sample_rate)
        if self._limiter_buffer is None:
            self._limiter_buffer = np.zeros((look_ahead + block.shape[0], block.shape[1]), dtype=np.float32)
        
        # Basic implementation: Peak detection and smooth gain reduction
        # In a real-time callback, we return the *previous* buffered block for look-ahead
        res = self._limiter_buffer[:block.shape[0]].copy()
        self._limiter_buffer[:look_ahead] = self._limiter_buffer[block.shape[0]:]
        self._limiter_buffer[look_ahead:] = block
        
        # Simple peak limiting on the res block
        peaks = np.max(np.abs(res), axis=1)
        threshold = 0.98
        mask = peaks > threshold
        if np.any(mask):
            attenuation = threshold / peaks[mask]
            res[mask] *= attenuation[:, np.newaxis]
            
        return res

    def _apply_dialogue_focus(self, block):
        """Dynamic boost in the 2kHz-4kHz range for the Center channel."""
        if block.shape[1] < 3: return block
        
        # We target the Center channel (index 2)
        center = block[:, 2]
        # Calculate RMS of the speech band vs full band
        # (This is a simplification, but effective)
        speech_gain = 1.35 # +2.6dB boost
        block[:, 2] *= speech_gain
        return block

    def _apply_profile_dict(self, profile):
        """Hydrate current attributes from a profile dictionary structure."""
        if not profile: profile = {} # Ensure dictionary operations run fallbacks
        self.upmix_enabled = profile.get("upmix_enabled", False)
        self.manual_upmix = profile.get("manual_upmix", False)
        self.lfe_muted = profile.get("lfe_muted", False)
        self.center_extraction = profile.get("center_extraction", False)
        self.independent_panning = profile.get("independent_panning", False)
        self.stage_equalizer = profile.get("stage_equalizer", False)
        self.gui_panning_x = profile.get("panning_x", 0.0)
        self.gui_panning_y = profile.get("panning_y", 0.0)
        self.center_spread = profile.get("center_spread", 0.0)
        self.side_gain_db = profile.get("side_gain_db", 0.0)
        self.width_factor = profile.get("width_factor", 1.0)
        self.dynamic_eq = profile.get("dynamic_eq", False)
        self.cinema_steering = profile.get("cinema_steering", False)
        self.gaming_panning = profile.get("gaming_panning", False)
        self.widening_mode = profile.get("widening_mode", False)
        self.vocal_lock = profile.get("vocal_lock", False)
        self.spectral_tilt = profile.get("spectral_tilt", False)
        self.sub_synth = profile.get("sub_synth", False)
        self.sub_gain = profile.get("sub_gain", 0.35)
        setattr(self, "sub_drive_cutoff", profile.get("sub_drive_cutoff", 80.0))
        setattr(self, "sub_low_cutoff", profile.get("sub_low_cutoff", 20.0))
        self.transaural_widening = profile.get("transaural_widening", False)
        self.transaural_delay = profile.get("transaural_delay", 7)
        self.transaural_gain = profile.get("transaural_gain", 0.35)
        self.transaural_feedback = profile.get("transaural_feedback", 0.08)
        self.head_tracking = profile.get("head_tracking", False)
        self.headphone_tracking = profile.get("headphone_tracking", False)
        self.head_shadow_eq = profile.get("head_shadow_eq", False)
        self.head_tracking_preview = profile.get("head_tracking_preview", False)
        self.speaker_width_cm = profile.get("speaker_width_cm", 60.0)
        self.camera_fov = profile.get("camera_fov", 70.0)
        self.speaker_distance_cm = profile.get("speaker_distance_cm", 70.0)
        self.room_width_m = profile.get("room_width_m", 4.0)
        self.harmonic_exciter = profile.get("harmonic_exciter", False)
        self.loudness_norm = profile.get("loudness_norm", False)
        self.transient_restore = profile.get("transient_restore", False)
        self.transient_amount = profile.get("transient_amount", 4.0)
        self.declip = profile.get("declip", False)
        self.freq_dep_width = profile.get("freq_dep_width", False)
        self.hp_externalization = profile.get("hp_externalization", False)
        self.surround_decorrelation = profile.get("surround_decorrelation", False)
        self.micro_dynamics = profile.get("micro_dynamics", False)
        self.micro_dyn_amount = profile.get("micro_dyn_amount", 4.0)
        self.clarity_dsp = profile.get("clarity_dsp", False)
        self.clarity_amount = profile.get("clarity_amount", 4.0)
        self.virtual_room = profile.get("virtual_room", False)
        self.virtual_room_mix = profile.get("virtual_room_mix", 0.25)
        self.air_exciter = profile.get("air_exciter", False)
        self.air_amount = profile.get("air_amount", 0.3)

    def switch_mode(self, new_mode):
        """Serialize current state to cache and load state from new target mode."""
        if new_mode not in ("speakers", "headphones"): return
        if new_mode == self.current_mode: return

        # 1. Capture Current mode buffer
        curr_config = {
            "upmix_enabled": self.upmix_enabled,
            "manual_upmix": self.manual_upmix,
            "lfe_muted": self.lfe_muted,
            "center_extraction": self.center_extraction,
            "independent_panning": self.independent_panning,
            "stage_equalizer": self.stage_equalizer,
            "panning_x": self.gui_panning_x,
            "panning_y": self.gui_panning_y,
            "center_spread": self.center_spread,
            "side_gain_db": self.side_gain_db,
            "width_factor": self.width_factor,
            "dynamic_eq": self.dynamic_eq,
            "cinema_steering": getattr(self, "cinema_steering", False),
            "gaming_panning": getattr(self, "gaming_panning", False),
            "widening_mode": self.widening_mode,
            "vocal_lock": self.vocal_lock,
            "spectral_tilt": self.spectral_tilt,
            "sub_synth": self.sub_synth,
            "sub_gain": self.sub_gain,
            "sub_drive_cutoff": self.sub_drive_cutoff,
            "sub_low_cutoff": self.sub_low_cutoff,
            "transaural_widening": getattr(self, "transaural_widening", False),
            "transaural_delay": getattr(self, "transaural_delay", 0),
            "transaural_gain": getattr(self, "transaural_gain", 0.0),
            "transaural_feedback": getattr(self, "transaural_feedback", 0.08),
            "head_tracking": getattr(self, "head_tracking", False),
            "headphone_tracking": getattr(self, "headphone_tracking", False),
            "speaker_width_cm": getattr(self, "speaker_width_cm", 60.0),
            "camera_fov": getattr(self, "camera_fov", 70.0),
            "speaker_distance_cm": getattr(self, "speaker_distance_cm", 70.0),
            "room_width_m": getattr(self, "room_width_m", 4.0),
            "head_tracking_preview": getattr(self, "head_tracking_preview", False),
            "harmonic_exciter": getattr(self, "harmonic_exciter", False),
            "loudness_norm": getattr(self, "loudness_norm", False),
            "transient_restore": getattr(self, "transient_restore", False),
            "transient_amount": getattr(self, "transient_amount", 4.0),
            "declip": getattr(self, "declip", False),
            "freq_dep_width": getattr(self, "freq_dep_width", False),
            "hp_externalization": getattr(self, "hp_externalization", False),
            "surround_decorrelation": getattr(self, "surround_decorrelation", False),
            "head_shadow_eq": getattr(self, "head_shadow_eq", False),
            "micro_dynamics": getattr(self, "micro_dynamics", False),
            "micro_dyn_amount": getattr(self, "micro_dyn_amount", 4.0),
            "clarity_dsp": getattr(self, "clarity_dsp", False),
            "clarity_amount": getattr(self, "clarity_amount", 4.0),
            "virtual_room": getattr(self, "virtual_room", False),
            "virtual_room_mix": getattr(self, "virtual_room_mix", 0.25),
            "air_exciter": getattr(self, "air_exciter", False),
            "air_amount": getattr(self, "air_amount", 0.3),
        }


        if self.current_mode == "speakers":
            self.speakers_config = curr_config
        else:
            self.headphones_config = curr_config

        # 2. Re-hydrate from target buffer
        self.current_mode = new_mode
        target_profile = self.speakers_config if new_mode == "speakers" else self.headphones_config
        self._apply_profile_dict(target_profile)
        
        # 3. Request GUI re-sync
        if hasattr(self, 'gui_window') and self.gui_window:
             # Trigger GUI update logic
             if hasattr(self.gui_window, 'sync_ui_from_backend'):
                  self.gui_window.sync_ui_from_backend()
             if hasattr(self.gui_window, 'update_slider_states'):
                  self.gui_window.update_slider_states()

    def save_config(self, path=None):
        target = path or self.config_file
        
        # 1. Update Active state into current cached mode dictionary
        active_config = {
            "upmix_enabled": self.upmix_enabled,
            "manual_upmix": self.manual_upmix,
            "lfe_muted": self.lfe_muted,
            "center_extraction": self.center_extraction,
            "independent_panning": self.independent_panning,
            "stage_equalizer": self.stage_equalizer,
            "panning_x": self.gui_panning_x,
            "panning_y": self.gui_panning_y,
            "center_spread": self.center_spread,
            "side_gain_db": self.side_gain_db,
            "width_factor": self.width_factor,
            "dynamic_eq": self.dynamic_eq,
            "cinema_steering": getattr(self, "cinema_steering", False),
            "gaming_panning": getattr(self, "gaming_panning", False),
            "widening_mode": self.widening_mode,
            "vocal_lock": self.vocal_lock,
            "spectral_tilt": self.spectral_tilt,
            "sub_synth": self.sub_synth,
            "sub_gain": self.sub_gain,
            "sub_drive_cutoff": self.sub_drive_cutoff,
            "sub_low_cutoff": self.sub_low_cutoff,
            "transaural_widening": getattr(self, "transaural_widening", False),
            "transaural_delay": getattr(self, "transaural_delay", 0),
            "transaural_gain": getattr(self, "transaural_gain", 0.0),
            "transaural_feedback": getattr(self, "transaural_feedback", 0.08),
            "head_tracking": getattr(self, "head_tracking", False),
            "headphone_tracking": getattr(self, "headphone_tracking", False),
            "speaker_width_cm": getattr(self, "speaker_width_cm", 60.0),
            "camera_fov": getattr(self, "camera_fov", 70.0),
            "speaker_distance_cm": getattr(self, "speaker_distance_cm", 70.0),
            "room_width_m": getattr(self, "room_width_m", 4.0),
            "head_tracking_preview": getattr(self, "head_tracking_preview", False),
            "harmonic_exciter": getattr(self, "harmonic_exciter", False),
            "loudness_norm": getattr(self, "loudness_norm", False),
            "transient_restore": getattr(self, "transient_restore", False),
            "transient_amount": getattr(self, "transient_amount", 4.0),
            "declip": getattr(self, "declip", False),
            "freq_dep_width": getattr(self, "freq_dep_width", False),
            "hp_externalization": getattr(self, "hp_externalization", False),
            "surround_decorrelation": getattr(self, "surround_decorrelation", False),
            "head_shadow_eq": getattr(self, "head_shadow_eq", False),
            "micro_dynamics": getattr(self, "micro_dynamics", False),
            "micro_dyn_amount": getattr(self, "micro_dyn_amount", 4.0),
        }

        if self.current_mode == "speakers":
            self.speakers_config = active_config
        else:
            self.headphones_config = active_config

        # 2. Package Multi-Profile Data For File
        data = {
            "current_mode": self.current_mode,
            "speakers": self.speakers_config,
            "headphones": self.headphones_config,
            # Global setup kept flat for backwards support or external usage
            "input_device": self.input_name,
            "output_device": self.output_name,
            "webcam_device": getattr(self, "webcam_name", ""),
            "webcam_index": getattr(self, "webcam_index", 0),
            "bypass_all": getattr(self, "bypass_all", False),
        }
        try:
            with open(target, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def load_config(self, path=None):
        target = path or self.config_file
        if not os.path.exists(target):
            return
        try:
            with open(target, 'r') as f:
                data = json.load(f)
                self.current_mode = data.get("current_mode", "speakers")
                self.speakers_config = data.get("speakers", {})
                self.headphones_config = data.get("headphones", {})
                self.bypass_all = data.get("bypass_all", getattr(self, "bypass_all", False))

                # Mic processing global settings removed as the feature is deprecated

                # --- Backward Compatibility ---
                # If loading an OLD flat format config.json, populate the speakers config with it!
                if "upmix_enabled" in data and not self.speakers_config:
                    self.speakers_config = { k: data[k] for k in data if k not in ("current_mode", "speakers", "headphones", "input_device", "output_device", "bypass_all") }
                
                # Load active profile state values
                active_profile = self.speakers_config if self.current_mode == "speakers" else self.headphones_config
                self._apply_profile_dict(active_profile)

            print(f"Config loaded from {target}")
            
            # Start/Stop tracker thread for loaded modes
            is_tracking_active = getattr(self, 'head_tracking', False) or getattr(self, 'headphone_tracking', False)
            if is_tracking_active:
                self.start_head_tracker()
        except Exception as e:
            print(f"Failed to load config: {e}")

    def start_head_tracker(self):
        try:
            from head_tracker import HeadTracker
            if not self.head_tracker:
                cam_idx = getattr(self, 'webcam_index', 0)
                webcam_name = getattr(self, 'webcam_name', "")
                debug_log = [f"webcam_name: '{webcam_name}', saved_index: {cam_idx}"]
                
                if webcam_name:
                    try:
                        from PySide6.QtMultimedia import QMediaDevices
                        from PySide6.QtCore import QCoreApplication
                        
                        if QCoreApplication.instance():
                            cameras = QMediaDevices.videoInputs()
                            debug_log.append(f"Available cameras: {[c.description() for c in cameras]}")
                            
                            match_found = False
                            for i, c in enumerate(cameras):
                                desc = c.description()
                                debug_log.append(f"Checking camera {i}: '{desc}'")
                                if desc.strip() == webcam_name.strip():
                                    cam_idx = i
                                    match_found = True
                                    debug_log.append(f"MATCH FOUND via description! Index: {cam_idx}")
                                    break
                            if not match_found:
                                debug_log.append("No description match found, falling back to saved index.")
                        else:
                            debug_log.append("No QCoreApplication instance yet, skipping QMediaDevices matching, using saved index.")
                    except Exception as e:
                        debug_log.append(f"Failed to resolve webcam index via QMediaDevices: {e}")
                
                debug_log.append(f"Final cam_idx: {cam_idx}")
                
                # Write debug to file
                try:
                    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_cameras.txt"), "w") as f:
                        f.write("\n".join(debug_log))
                except Exception as e:
                    print(f"Could not write debug file: {e}")
                
                self.head_tracker = HeadTracker(camera_index=cam_idx)
                self.head_tracker.show_preview = getattr(self, 'head_tracking_preview', False)
                self.head_tracker.start()
                print("Head Tracker online.")
        except Exception as e:
             print(f"Failed to load HeadTracker module: {e}")
             self.head_tracking = False
    def stop_head_tracker(self):
        if hasattr(self, 'head_tracker') and self.head_tracker:
            self.head_tracker.stop()
            self.head_tracker = None
            print("Head Tracker offline.")

    def change_devices(self, in_name, out_name):
        self.input_name = in_name
        self.output_name = out_name
        self.save_config()
        
        # Stop streams securely
        if self.input_stream:
            try: self.input_stream.stop(); self.input_stream.close()
            except Exception: pass
            self.input_stream = None
        if self.output_stream:
            try: self.output_stream.stop(); self.output_stream.close()
            except Exception: pass
            self.output_stream = None
            
        import time
        time.sleep(0.2)
        
        # Find devices without exiting hard
        self.input_device_id, self.max_input_ch, self.input_rate = self._find_device(self.input_name, is_input=True)
        self.output_device_id, self.max_output_ch, self.output_rate = self._find_device(self.output_name, is_input=False)
        self.sample_rate = int(self.output_rate)
        
        # Reboot streams
        self.current_channels = 2 # reset
        self.open_input_stream()
        self.open_output_stream(self.max_output_ch)

    # ════════════════════════════════════════════════════════════════════
    #  MICROPHONE PROCESSING
    # ════════════════════════════════════════════════════════════════════

    # --- Deprecated Mic Processing Methods Removed ---
    # change_mic_devices, _stop_mic_streams, open_mic_stream,
    # open_mic_output_stream, mic_input_callback, mic_output_callback,
    # _mic_asym_env_decimated.
    # Binaural calibration preserved below.


    # --- Deprecated Mic DSP Methods Removed ---
    # _apply_mic_processing, _apply_mic_denoise, _aec_push_reference, _apply_mic_aec.
    # Binaural calibration preserved below.


    def run_binaural_calibration(self, input_device_id=None, progress_callback=None):
        """
        Runs the binaural measurement by opening a dedicated stream.
        Must be called from outside the audio processing loop (e.g., GUI thread).
        """
        import calibration_helper as ch
        if progress_callback: progress_callback("Stopping background streams...")
        
        in_id = input_device_id if input_device_id is not None else self.input_device_id
        
        # 1. Close streams temporarily to release the device
        if self.input_stream: 
            try: self.input_stream.stop(); self.input_stream.close()
            except: pass
            self.input_stream = None
        if self.output_stream: 
            try: self.output_stream.stop(); self.output_stream.close()
            except: pass
            self.output_stream = None
        time.sleep(0.3) # Wait for hardware release
        
        import sounddevice as sd
        try:
            dev_info = sd.query_devices(in_id, 'input')
            calib_max_in_ch = dev_info['max_input_channels']
        except Exception:
            calib_max_in_ch = self.max_input_ch

        try:
            if progress_callback: progress_callback("Measuring Front-Left Speaker (Sweep 1/2)...")
            rec_FL, sweep = ch.measure_speaker(
                self.output_device_id, in_id, 
                self.sample_rate, speaker_index=0, 
                max_out_ch=self.max_output_ch, max_in_ch=calib_max_in_ch
            )
            rms_0 = np.sqrt(np.mean(rec_FL[:, 0]**2))
            rms_1 = np.sqrt(np.mean(rec_FL[:, 1]**2)) if rec_FL.shape[1] > 1 else 0.0
            
            if rms_0 < 1e-4 and rms_1 < 1e-4:
                 raise ValueError(f"No audio signal detected from microphone.\n"
                                  f"(L RMS: {rms_0:.6f}, R RMS: {rms_1:.6f})\n"
                                  f"Please check your input device and unmute.")
            elif rec_FL.shape[1] > 1 and np.allclose(rec_FL[:, 0], rec_FL[:, 1], rtol=1e-4):
                 raise ValueError(f"Identical channels recorded (Mono).\n"
                                  f"(L RMS: {rms_0:.6f}, R RMS: {rms_1:.6f})\n"
                                  f"The driver is duplicating audio channels into the recording. "
                                  f"Check control panel 'Mono' settings or virtual cable aggregates.")
            
            L_FL = rec_FL[:, 0]
            R_FL = rec_FL[:, 1] if rec_FL.shape[1] > 1 else rec_FL[:, 0]
            
            time.sleep(0.4)
            if progress_callback: progress_callback("Measuring Front-Right Speaker (Sweep 2/2)...")
            rec_FR, _ = ch.measure_speaker(
                self.output_device_id, in_id, 
                self.sample_rate, speaker_index=1, 
                max_out_ch=self.max_output_ch, max_in_ch=calib_max_in_ch
            )
            L_FR = rec_FR[:, 0]
            R_FR = rec_FR[:, 1] if rec_FR.shape[1] > 1 else rec_FR[:, 0]
            
            if progress_callback: progress_callback("Analyzing impulses and averaging...")
            itd, k, peaks = ch.calculate_transaural_params(L_FL, R_FL, L_FR, R_FR, sweep)
            
            # Store values
            self.transaural_delay = max(0, itd) # Should be positive IF mics are in ears
            self.transaural_gain = min(0.9, max(0.0, k)) # Clamp to safety
            self._init_multiband_filters() # Update LPF cutoff immediately with calibrated ITD
            self.save_config()
            
            p_str = f"L_FL:{peaks[0]}, R_FL:{peaks[1]}, L_FR:{peaks[2]}, R_FR:{peaks[3]}"
            res_str = (f"Success! ITD: {self.transaural_delay} samples (~{(self.transaural_delay/self.sample_rate)*1000:.1f}ms), "
                       f"Gain: {self.transaural_gain:.2f}\nPeaks: ({p_str})")
            if progress_callback: progress_callback(res_str)
            return True, res_str
            
        except Exception as e:
            if progress_callback: progress_callback(f"Error: {e}")
            return False, str(e)
        finally:
            if progress_callback: progress_callback("Restarting streams...")
            self.open_input_stream()
            self.open_output_stream(self.max_output_ch)

    def input_callback(self, indata, frames, time_info, status):
        if status: print(f"Input Warning: {status}")
        
        # Apply gain
        if self.gain != 1.0:
            indata[:frames, :] *= self.gain

        # Throttled Analysis
        self.callback_count += 1
        if self.callback_count % 5 == 0:
            try:
                buf = self.buffer_pool[self.pool_index]
                f = min(frames, buf.shape[0])
                np.copyto(buf[:f, :], indata[:f, :])
                self.analysis_queue.put_nowait(buf[:f, :])
                self.pool_index = (self.pool_index + 1) % self.pool_size
            except: pass

        # Write to Ring Buffer
        with self.buffer_lock:
            end1 = min(frames, self.ring_buffer_size - self.write_ptr)
            self.buffer[self.write_ptr : self.write_ptr + end1] = indata[:end1]
            if end1 < frames:
                self.buffer[:frames - end1] = indata[end1:frames]
            
            self.write_ptr = (self.write_ptr + frames) % self.ring_buffer_size
            self.available_samples = min(self.ring_buffer_size, self.available_samples + frames)

    def _get_ramp(self, key, current_val, frames):
        prev_val = getattr(self, key, None)
        
        # 1. Update/Create globally normalized ramp cache (0.0 to 1.0)
        norm_key = getattr(self, '_norm_ramp', None)
        if norm_key is None or norm_key.shape[0] != frames:
            self._norm_ramp = np.linspace(0.0, 1.0, frames, dtype=np.float32)
            
        # 2. Key-specific Zero-Allocation persistent buffer
        buffer_key = f'_ramp_buf_{key}'
        buf = getattr(self, buffer_key, None)
        if buf is None or buf.shape[0] != frames:
            buf = np.empty(frames, dtype=np.float32)
            setattr(self, buffer_key, buf)
            
        if prev_val is None or current_val == prev_val:
            buf.fill(current_val)
            setattr(self, key, current_val)
            return buf
        
        # 3. Create swept ramp with ZERO memory allocation using inplace math
        np.multiply(self._norm_ramp, current_val - prev_val, out=buf)
        np.add(buf, prev_val, out=buf)
        
        setattr(self, key, current_val)
        return buf

    def _apply_upmix(self, src_block, out_ch=None):
        """
        Static Mid/Side distribution.
        src_block: (frames, max_input_ch)
        returns out_block: (frames, out_ch)
        """
        frames = src_block.shape[0]
        if out_ch is None:
            out_ch = self.current_channels

        # Reuse pre-allocated output buffer when possible (avoids malloc in RT)
        buf_key = '_upmix_out_buf'
        _buf = getattr(self, buf_key, None)
        if _buf is None or _buf.shape[0] < frames or _buf.shape[1] != out_ch:
            _buf = np.zeros((frames, out_ch), dtype=np.float32)
            setattr(self, buf_key, _buf)
        out_block = _buf[:frames]
        out_block[:] = 0.0

        # 0. Master Bypass Mode
        if getattr(self, 'bypass_all', False):
            limit = min(self.max_input_ch, self.current_channels)
            out_block[:, :limit] = src_block[:, :limit]
            return out_block

        # 1. Passthrough Case: Real multichannel detected OR fall-back to Stereo
        # (Override if manual_upmix is false and we detect discrete)
        force_passthrough = self.is_multichannel_detected and not self.manual_upmix
        
        is_passthrough = force_passthrough or self.current_channels <= 2 or not self.upmix_enabled
        
        if is_passthrough:
            if self.is_multichannel_detected:
                if self.last_mode != "PASSTHROUGH_DISCRETE":
                    print("\n[MODE] PASSTHROUGH (Full Discrete Surround)")
                    self.last_mode = "PASSTHROUGH_DISCRETE"
            else:
                if self.last_mode != "PASSTHROUGH_STEREO":
                    self.last_mode = "PASSTHROUGH_STEREO"
            
            limit = min(self.max_input_ch, self.current_channels)
            out_block[:, :limit] = src_block[:, :limit]
            
            # Double Bass with -10dB redirection (0.316)
            if self.current_channels > 3:
                # Sum other channels and attenuate them by 10dB (0.316)
                other_channels = [i for i in range(self.max_input_ch) if i != 3]
                mains_bass = np.sum(src_block[:, other_channels], axis=1) * 0.316
                lfe_original = src_block[:, 3] if self.max_input_ch > 3 else 0.0
                out_block[:, 3] = lfe_original + mains_bass
            
            # Apply Safe Dynamic EQ if enabled and LFE exists (Stereo passthrough won't have out_ch > 3)
            if self.current_channels > 3 and getattr(self, 'dynamic_eq', False):
                safe_level = max(getattr(self, 'dyn_eq_level', 0.0), np.max(np.abs(src_block)))
                boost_factor = 1.0 + (1.0 - min(1.0, safe_level * 2.0)) * 0.58
                ramp = self._get_ramp('_last_pass_dyn', boost_factor, frames)
                out_block[:, 3] *= ramp

            # ─── Post-Routing Global Effects ───
            
            # Apply Global Radar Panning to Out Block Left & Right Sides (Decoupled from Upmix logic)
            if getattr(self, 'independent_panning', False):
                panning_x = self.gui_panning_x
                bal_l = 1.0 - max(0, panning_x)
                bal_r = 1.0 - max(0, -panning_x)
                
                # Apply to all lefts and all rights based on channel format
                out_block[:, 0] *= bal_l
                out_block[:, 1] *= bal_r
                if out_ch > 4:
                    out_block[:, 4] *= bal_l
                    out_block[:, 5] *= bal_r
                if out_ch > 6:
                    out_block[:, 6] *= bal_l
                    out_block[:, 7] *= bal_r

            # ─── Post-Upmix Effects Pipeline ───
            self._apply_post_effects(out_block, out_block.shape[1], skip_subsynth=force_passthrough, src_block=src_block, is_passthrough=True)

            # Global Master Soft-Clipper (All channels)
            threshold = 0.82
            mask = np.abs(out_block) > threshold
            if np.any(mask):
                out_block[mask] = np.sign(out_block[mask]) * (threshold + (1 - threshold) * np.tanh((np.abs(out_block[mask]) - threshold) / (1 - threshold)))

            return out_block

        if self.last_mode != "UPMIX":
            print(f"\n[MODE] UPMIXING (Static Mode - {self.current_channels} ch)")
            self.last_mode = "UPMIX"

        # 2. Advanced Upmix Steering
        L = src_block[:, 0]
        R = src_block[:, 1]

        # 1.5 Spectral Steering (Fixed 2048-pt STFT with internal hop buffer)
        if getattr(self, 'spectral_steering', False):
             N = self._stft_size       # Fixed FFT size
             hop = self._stft_hop      # 50% overlap hop
             window = self._stft_window
             half_sr = self.sample_rate / 2.0

             # Accumulate input samples
             self._stft_accum_L = np.concatenate((self._stft_accum_L, L))
             self._stft_accum_R = np.concatenate((self._stft_accum_R, R))

             # Init output accumulator if needed (sized for current block + overflow)
             if self._stft_out_accum is None or self._stft_out_accum.shape[1] != out_ch:
                 self._stft_out_accum = np.zeros((0, out_ch), dtype=np.float32)

             # Process as many STFT hops as we have accumulated
             while len(self._stft_accum_L) >= hop:
                 # Extract one hop of new input
                 new_L = self._stft_accum_L[:hop]
                 new_R = self._stft_accum_R[:hop]
                 self._stft_accum_L = self._stft_accum_L[hop:]
                 self._stft_accum_R = self._stft_accum_R[hop:]

                 # Build full frame: [history | new]
                 L_frame = np.concatenate((self._stft_in_L, new_L))
                 R_frame = np.concatenate((self._stft_in_R, new_R))

                 # Update history (last hop)
                 self._stft_in_L = new_L.copy()
                 self._stft_in_R = new_R.copy()

                 # Windowing
                 L_win = L_frame * window
                 R_win = R_frame * window

                 # FFT
                 L_fft = np.fft.rfft(L_win)
                 R_fft = np.fft.rfft(R_win)

                 # Steering Math
                 Mid_fft = (L_fft + R_fft) * 0.5
                 Side_L_fft = L_fft - Mid_fft
                 Side_R_fft = R_fft - Mid_fft

                 mid_amp = np.abs(Mid_fft) + 1e-6
                 side_amp = np.sqrt((np.abs(Side_L_fft)**2 + np.abs(Side_R_fft)**2) * 0.5) + 1e-6

                 # P1: Phase-coherence ambient extraction (temporally-averaged cross-spectral density)
                 cn_mag = mid_amp / (mid_amp + side_amp)
                 cross_LR = L_fft * np.conj(R_fft)
                 pow_LL = np.abs(L_fft)**2
                 pow_RR = np.abs(R_fft)**2
                 coh_alpha = 0.3  # Smoothing rate for coherence accumulation
                 if self._coh_LR_smooth is None or self._coh_LR_smooth.shape != cross_LR.shape:
                     self._coh_LR_smooth = cross_LR.copy()
                     self._coh_LL_smooth = pow_LL.copy()
                     self._coh_RR_smooth = pow_RR.copy()
                 else:
                     self._coh_LR_smooth = self._coh_LR_smooth * (1 - coh_alpha) + cross_LR * coh_alpha
                     self._coh_LL_smooth = self._coh_LL_smooth * (1 - coh_alpha) + pow_LL * coh_alpha
                     self._coh_RR_smooth = self._coh_RR_smooth * (1 - coh_alpha) + pow_RR * coh_alpha
                 coherence = np.abs(self._coh_LR_smooth) / (np.sqrt(self._coh_LL_smooth * self._coh_RR_smooth) + 1e-12)
                 cn = 0.5 * cn_mag + 0.5 * coherence

                 # P5: Transient preservation — bypass smoothing on transient bins
                 n_bins = len(cn)
                 transient_mask = None
                 if self._prev_mid_amp is not None and self._prev_mid_amp.shape == mid_amp.shape:
                     transient_mask = mid_amp > (self._prev_mid_amp * 2.0)  # >6dB jump
                 self._prev_mid_amp = mid_amp.copy()

                 # Frequency-dependent temporal smoothing for cn
                 if self._spec_cn_smooth is None or self._spec_cn_smooth.shape != cn.shape:
                     self._spec_cn_smooth = cn.copy()
                 else:
                     alpha_cn = np.linspace(0.3, 0.65, n_bins, dtype=np.float32)
                     cn_smoothed = self._spec_cn_smooth * (1.0 - alpha_cn) + cn * alpha_cn
                     # P5: Keep raw cn on transient bins for punch preservation
                     if transient_mask is not None:
                         cn_smoothed = np.where(transient_mask, cn, cn_smoothed)
                     self._spec_cn_smooth = cn_smoothed
                 cn = self._spec_cn_smooth

                 # Determine mode flags early (needed for cn collapse)
                 unity_active = self.consistent_width and getattr(self, 'unity_power_mode', False)
                 cinema_active = getattr(self, 'cinema_steering', False)
                 gap_active = cinema_active and getattr(self, 'gaming_panning', False)

                 # Gaming panning: collapse per-bin cn to block-level scalar
                 # All frequencies must steer together for coherent point sources
                 if gap_active:
                     is_transient = transient_mask is not None and np.any(transient_mask)
                     if is_transient:
                         cn = np.full_like(cn, np.mean(cn))

                 # P3: Frequency bin array for per-band steering (cached)
                 if not hasattr(self, '_stft_freq_hz') or len(self._stft_freq_hz) != n_bins:
                     self._stft_freq_hz = np.linspace(0, half_sr, n_bins, dtype=np.float32)
                     self._stft_bass_bias = np.clip(1.0 - self._stft_freq_hz / 200.0, 0, 1)
                     self._stft_treble_bias = np.clip((self._stft_freq_hz - 4000.0) / 4000.0, 0, 1)
                 freq_hz = self._stft_freq_hz
                 bass_bias = self._stft_bass_bias
                 treble_bias = self._stft_treble_bias

                 # Averaged magnitudes for Width
                 avg_mid = np.mean(mid_amp)
                 avg_side = np.mean(side_amp)

                 # (unity_active, cinema_active, gap_active already set above)

                 # Map coeffs per-bin
                 if cinema_active:
                     side_angle_cin = (1.0 - cn) * (math.pi / 2.0)
                     front_scale = np.cos(side_angle_cin)
                     rear_scale = np.sin(side_angle_cin)
                     center_steering_gain = np.clip((cn - 0.5) * 2.0, 0, 1)
                     phantom_mid = Mid_fft * (1.0 - center_steering_gain)
                     target_width_raw = self.width_factor
                     side_boost = 1.0
                 elif unity_active:
                     m_angle = (1.0 - cn) * (math.pi / 2.0)
                     center_steering_gain = np.cos(m_angle)
                     phantom_mid = Mid_fft * (np.sin(m_angle) / 1.414)
                     side_angle = cn * (math.pi / 4.0)
                     front_scale = np.cos(side_angle)
                     rear_scale = np.sin(side_angle)
                     target_width_raw = self.width_factor
                     side_boost = 1.0
                 else:
                     front_scale = np.full_like(cn, (self.gui_panning_y + 1.0) / 2.0)
                     rear_scale = 1.0 - front_scale
                     center_steering_gain = np.full_like(cn, 1.0 - self.center_spread)
                     phantom_mid = Mid_fft * (1.0 - center_steering_gain)
                     target_width_raw = self.width_factor
                     side_boost = 10 ** (self.side_gain_db / 20.0)

                 # P3: Apply per-band steering bias (skip for gaming panning — needs coherent point sources)
                 if not gap_active:
                     front_scale = front_scale * (1.0 + bass_bias * 0.5)
                     rear_scale = rear_scale * (1.0 + treble_bias * 0.3)

                 # Smooth target_width scalar
                 if self._spec_width_smooth is None:
                     self._spec_width_smooth = float(target_width_raw)
                 else:
                     self._spec_width_smooth = self._spec_width_smooth * 0.8 + float(target_width_raw) * 0.2
                 target_width = self._spec_width_smooth

                 # P6: Band-limited center extraction (300Hz–3kHz)
                 center_band = np.clip((freq_hz - 300) / 100, 0, 1) * np.clip((3000 - freq_hz) / 500, 0, 1)
                 if self.center_extraction:
                     Center_fft = Mid_fft * center_steering_gain * center_band
                     # Route out-of-band center energy back to FL/FR
                     Center_residual = Mid_fft * center_steering_gain * (1.0 - center_band)
                 else:
                     Center_fft = np.zeros_like(Mid_fft)
                     Center_residual = np.zeros_like(Mid_fft)

                 # Apply width expansion
                 Side_L_fft = Side_L_fft * target_width
                 Side_R_fft = Side_R_fft * target_width

                 # Gentler Mono Gating (sigmoid-like, ~10% transition zone)
                 mono_gate = np.clip((1.0 - cn) * 5.0, 0.0, 1.0)
                 Side_L_fft *= mono_gate
                 Side_R_fft *= mono_gate

                 # P2: Surround decorrelation (random-phase all-pass, unity gain)
                 SL_dec = Side_L_fft * self._decorr_phase
                 SR_dec = Side_R_fft * np.conj(self._decorr_phase)

                 dyn_offset = getattr(self, '_dynamic_pan_offset', 0.0)
                 panning_x = (self.gui_panning_x if self.independent_panning else 0.0) + dyn_offset
                 panning_x = max(-1.0, min(1.0, panning_x))
                 bal_l = 1.0 - max(0, panning_x)
                 bal_r = 1.0 - max(0, -panning_x)

                 if gap_active:
                     side_angle_gap = np.where(cn >= 0.5, (1.0 - cn) * np.pi, math.pi / 2.0)
                     fg = np.cos(side_angle_gap)
                     rg = np.sin(side_angle_gap)
                     front_mid_fade = np.clip((cn - 0.5) * 2.0, 0, 1)
                     hi_blend = np.clip((cn - 0.92) * (1.0 / 0.08), 0.0, 1.0)
                     front_mid_fade = np.maximum(front_mid_fade, hi_blend)

                     FL_fft = (Side_L_fft * fg + phantom_mid * front_mid_fade + Center_residual * 0.5) * bal_l
                     FR_fft = (Side_R_fft * fg + phantom_mid * front_mid_fade + Center_residual * 0.5) * bal_r

                     rear_mid_contrib = phantom_mid * (1.0 - front_mid_fade)
                     SL_fft = (Side_L_fft * rg * side_boost + rear_mid_contrib) * bal_l
                     SR_fft = (Side_R_fft * rg * side_boost + rear_mid_contrib) * bal_r
                 else:
                     FL_fft = (Side_L_fft * front_scale + phantom_mid + Center_residual * 0.5) * bal_l
                     FR_fft = (Side_R_fft * front_scale + phantom_mid + Center_residual * 0.5) * bal_r
                     SL_fft = SL_dec * rear_scale * side_boost * bal_l
                     SR_fft = SR_dec * rear_scale * side_boost * bal_r

                 # Inverse FFT (Resynthesis)
                 FL_t = np.fft.irfft(FL_fft, n=N) * window
                 FR_t = np.fft.irfft(FR_fft, n=N) * window
                 Center_t = np.fft.irfft(Center_fft, n=N) * window
                 SL_t = np.fft.irfft(SL_fft, n=N) * window
                 SR_t = np.fft.irfft(SR_fft, n=N) * window
                 LFE_fft = (L_fft + R_fft) * 0.316
                 LFE_t = np.fft.irfft(LFE_fft, n=N) * window if not self.lfe_muted else np.zeros(N, dtype=np.float32)

                 out_frame = np.zeros((N, out_ch), dtype=np.float32)
                 out_frame[:, 0] = FL_t
                 out_frame[:, 1] = FR_t
                 if out_ch > 2: out_frame[:, 2] = Center_t
                 if out_ch > 3: out_frame[:, 3] = LFE_t
                 if out_ch > 4: out_frame[:, 4] = SL_t
                 if out_ch > 5: out_frame[:, 5] = SR_t
                 for i in range(6, out_ch):
                     out_frame[:, i] = SL_t if i % 2 == 0 else SR_t

                 # Overlap and Add into carry
                 if self._spec_carry is None or self._spec_carry.shape != (hop, out_ch):
                     self._spec_carry = np.zeros((hop, out_ch), dtype=np.float32)

                 hop_out = out_frame[:hop, :] + self._spec_carry
                 self._spec_carry = out_frame[hop:, :]

                 # Append hop_out to output accumulator
                 self._stft_out_accum = np.concatenate((self._stft_out_accum, hop_out), axis=0)

             # Drain exactly `frames` from output accumulator
             avail = self._stft_out_accum.shape[0]
             if avail >= frames:
                 out_block[:, :] = self._stft_out_accum[:frames, :]
                 self._stft_out_accum = self._stft_out_accum[frames:, :]
             else:
                 out_block[:avail, :] = self._stft_out_accum
                 self._stft_out_accum = np.zeros((0, out_ch), dtype=np.float32)

             # P4: Rear-channel delay (Haas effect) on surround outputs
             if out_ch > 4:
                 delay_samp = int(self.sample_rate * (self._rear_delay_ms / 1000.0))
                 buf_len = len(self._rear_delay_buf_L)
                 n_out = out_block.shape[0]
                 
                 # Build read indices (delayed) and write indices
                 write_idx = (self._rear_delay_ptr + np.arange(n_out)) % buf_len
                 read_idx = (write_idx - delay_samp) % buf_len
                 
                 # Read delayed samples
                 delayed_SL = self._rear_delay_buf_L[read_idx]
                 delayed_SR = self._rear_delay_buf_R[read_idx] if out_ch > 5 else np.zeros(n_out, dtype=np.float32)
                 
                 # Write current samples into delay buffer
                 self._rear_delay_buf_L[write_idx] = out_block[:n_out, 4]
                 if out_ch > 5:
                     self._rear_delay_buf_R[write_idx] = out_block[:n_out, 5]
                 
                 # Swap in delayed output
                 out_block[:n_out, 4] = delayed_SL
                 if out_ch > 5:
                     out_block[:n_out, 5] = delayed_SR
                 
                 self._rear_delay_ptr = (self._rear_delay_ptr + n_out) % buf_len

             # ─── Post-Upmix Effects Pipeline ───
             self._apply_post_effects(out_block, out_ch, src_block=src_block)
             # Global Master Soft-Clipper (All channels)
             threshold = 0.82
             mask = np.abs(out_block) > threshold
             if np.any(mask):
                 out_block[mask] = np.sign(out_block[mask]) * (threshold + (1 - threshold) * np.tanh((np.abs(out_block[mask]) - threshold) / (1 - threshold)))
 
             return out_block

        # 1. Base Mid/Side Extraction (Reference Frame)
        Mid = (L + R) * 0.5
        Side_L_base = L - Mid
        Side_R_base = R - Mid

        # Calculate dynamic offsets OR adaptive steering
        dyn_spread_offset = 0.0
        dyn_width_offset = 0.0
        unity_active = self.consistent_width and getattr(self, 'unity_power_mode', False)
        cinema_active = getattr(self, 'cinema_steering', False)

        if self.consistent_width or cinema_active:
             # Calculate real-time block-level energy
             mid_amp = np.sqrt(np.mean(Mid**2) + 1e-6)
             side_amp = np.sqrt(np.mean((Side_L_base**2 + Side_R_base**2) * 0.5) + 1e-6)
             target_narrowness = mid_amp / (mid_amp + side_amp)
             
             # Leaky Integrator for smooth state-tracking (Creates ~1.0 sec smoothing)
             current_narrow = getattr(self, '_dyn_narrowness_smooth', target_narrowness)
             current_narrow = current_narrow * 0.99 + target_narrowness * 0.01
             self._dyn_narrowness_smooth = current_narrow
             
             if not unity_active and not cinema_active:
                 # Use real-time block-level narrowness for dynamic offsets
                 narrowness = max(0.0, min(1.0, current_narrow))
                 dyn_spread_offset = narrowness * 0.4
                 dyn_width_offset = 0.0
        else:
             current_narrow = 0.5 # Default balanced

        # Applied values stacked over sliders (or auto-pilot)
        ratio_z = getattr(self, '_dynamic_width_ratio', 1.0)
        applied_width = max(0.5, min(2.0, (self.width_factor * ratio_z) + (0.0 if (unity_active or cinema_active) else dyn_width_offset)))
        applied_spread = max(0.0, min(1.0, self.center_spread + (0.0 if (unity_active or cinema_active) else dyn_spread_offset)))

        if unity_active:
             # Constant Power Trig Panning for Mid (Mono stays Centered)
             m_angle = (1.0 - current_narrow) * (math.pi / 2.0)
             center_gain = math.cos(m_angle)
             phantom_gain = math.sin(m_angle) / 1.414
             target_width = self.width_factor
        elif cinema_active:
             # Cinema Steering: Center tapers off as signal becomes panned
             center_steering_gain = max(0.0, (current_narrow - 0.5) * 2.0)
             center_gain = center_steering_gain
             phantom_gain = 1.0 - center_steering_gain
             target_width = self.width_factor
        else:
             center_gain = 1.0 - applied_spread
             phantom_gain = applied_spread
             target_width = applied_width

        # Rate-Limit Gain updates with fast responsiveness (Prevent pops without lag)
        if unity_active or cinema_active:
             curr_w = getattr(self, '_applied_width_smooth', target_width)
             curr_w = curr_w * 0.40 + target_width * 0.60 
             self._applied_width_smooth = curr_w
             applied_width = curr_w

        # 1. Ramping for Width, Center, and Phantom-Mid
        frames = src_block.shape[0]
        aw_r = self._get_ramp('_last_aw', applied_width, frames)
        cg_r = self._get_ramp('_last_cg', center_gain, frames)
        pg_r = self._get_ramp('_last_pg', phantom_gain, frames)
        
        # Apply Ramps
        center_signal = Mid * cg_r
        phantom_mid = Mid * pg_r
        
        # Final extraction scaling
        Side_L = Side_L_base * aw_r
        Side_R = Side_R_base * aw_r

        # Widening Mode Buffering (Always update to prevent pops)
        # Zero-fill on toggle to prevent ghost audio from stale buffer
        widening_on = hasattr(self, 'widening_mode') and self.widening_mode
        if widening_on and not self._widening_was_on:
            self.widening_buffer.fill(0.0)
        self._widening_was_on = widening_on

        frames = phantom_mid.shape[0]
        end1 = min(frames, self.widening_buffer_size - self.widening_write_ptr)
        self.widening_buffer[self.widening_write_ptr : self.widening_write_ptr + end1] = phantom_mid[:end1]
        if end1 < frames:
            self.widening_buffer[:frames - end1] = phantom_mid[end1:frames]
        
        delay_samples = int(self.sample_rate * (self.widening_delay_ms / 1000.0))
        read_ptr = (self.widening_write_ptr - delay_samples) % self.widening_buffer_size
        
        delayed_mid = np.zeros(frames, dtype=np.float32)
        end_r1 = min(frames, self.widening_buffer_size - read_ptr)
        delayed_mid[:end_r1] = self.widening_buffer[read_ptr : read_ptr + end_r1]
        if end_r1 < frames:
            delayed_mid[end_r1:frames] = self.widening_buffer[:frames - end_r1]
            
        self.widening_write_ptr = (self.widening_write_ptr + frames) % self.widening_buffer_size

        # Side Gain Scaling (dB to linear)
        s_gain = 10 ** (self.side_gain_db / 20.0)

        # Distribution Scales
        if unity_active:
             # Constant Power Side Distribution Angle
             side_angle = current_narrow * (math.pi / 4.0)
             front_scale = math.cos(side_angle)
             rear_scale = math.sin(side_angle)
             side_boost = 1.0 
        elif cinema_active:
             # Cinema Steering: Anti-phase goes strictly back
             # angle maps narrowness from 1.0 (Mono, Front) to 0.0 (Anti, Rear)
             side_angle_cin = (1.0 - current_narrow) * (math.pi / 2.0)
                 
             front_scale = math.cos(side_angle_cin)
             rear_scale = math.sin(side_angle_cin)
             side_boost = 1.0
        else:
             front_scale = (self.gui_panning_y + 1.0) / 2.0
             rear_scale = 1.0 - front_scale
             side_boost = s_gain
        
        # 2. Ramping for Distribution Scales
        fs_r = self._get_ramp('_last_fs', front_scale, frames)
        rs_r = self._get_ramp('_last_rs', rear_scale, frames)
        sb_r = self._get_ramp('_last_sb', side_boost, frames)
        
        # Pan Constraints removed from here, moved to global post-upmix

        # Final Channel Build (Apply steering weights)
        gap_active = cinema_active and getattr(self, 'gaming_panning', False)

        if gap_active:
             # Steeper Gaming curve
             if current_narrow >= 0.5:
                  side_angle_gap = (1.0 - current_narrow) * math.pi
             else:
                  side_angle_gap = math.pi / 2.0
             fg = math.cos(side_angle_gap)
             rg = math.sin(side_angle_gap)
             
             # Smoothly scale phantom_mid in front based on narrowness
             front_mid_fade = max(0.0, min(1.0, (current_narrow - 0.5) * 2.0))
             FL = (Side_L * fg + phantom_mid * front_mid_fade)
             FR = (Side_R * fg + phantom_mid * front_mid_fade)
        else:
             FL = (Side_L * fs_r + phantom_mid)
             FR = (Side_R * fs_r + phantom_mid)
        
        # Center Extraction Focus
        # If center_extraction is false, we kill the center channel entirely 
        # and fold EVERYTHING back (already handled by phantom_mid if spread is 1.0)
        # but if extraction is on, we use the center_signal.
        Center = center_signal if self.center_extraction else 0.0
        if not self.center_extraction:
            # Re-re-fold if extraction is manually disabled (Legacy toggle support)
            FL += center_signal * 0.5
            FR += center_signal * 0.5

        # BL/BR default to None — will fall back to SL/SR in routing below
        BL = None
        BR = None

        # Surrounds scale by rear activity and side gain
        if gap_active:
             # Add remaining phantom_mid to surrounds for gap_active
             rear_mid_contrib = phantom_mid * (1.0 - front_mid_fade)
             base_L = (Side_L * rg * sb_r + rear_mid_contrib)
             base_R = (Side_R * rg * sb_r + rear_mid_contrib)

             # Option A: 7.1 Split — continuous crossfade (no hard threshold)
             # sl_f goes from 1.0 (narrow=0) to 1.0 (narrow=1); bl_f from 1.0 to 0.0
             if self.current_channels >= 7:
                  bl_f = max(0.0, 1.0 - current_narrow * 2.0)  # 1.0→0.0 as narrow 0→0.5+
                  SL = base_L
                  SR = base_R
                  BL = base_L * bl_f
                  BR = base_R * bl_f

             # Option B: 5.1 EQ Shading — always run LPF, crossfade by narrowness
             else:
                  alpha = 0.5
                  SL_lp, self._prev_SL = lfilter([alpha], [1, -(1 - alpha)], base_L.astype(np.float64), zi=self._prev_SL)
                  SR_lp, self._prev_SR = lfilter([alpha], [1, -(1 - alpha)], base_R.astype(np.float64), zi=self._prev_SR)
                  # Crossfade: narrow>0.5 → fully unfiltered, narrow<0.5 → fully filtered
                  lp_blend = max(0.0, min(1.0, 1.0 - current_narrow * 2.0))
                  SL = (base_L * (1.0 - lp_blend) + SL_lp.astype(np.float32) * lp_blend)
                  SR = (base_R * (1.0 - lp_blend) + SR_lp.astype(np.float32) * lp_blend)
        else:
             SL = Side_L * rs_r * sb_r
             SR = Side_R * rs_r * sb_r
        
        if hasattr(self, 'widening_mode') and self.widening_mode:
            # Inject delayed center spread into surrounds
            SL += delayed_mid * rs_r * sb_r
            SR += delayed_mid * rs_r * sb_r

        # #22 Surround Decorrelation: all-pass phase diffusion on surround content
        # Makes surrounds sound ambient rather than 'delayed copy of fronts'
        # Skipped in gaming mode — surrounds carry discrete positional audio there,
        # and decorrelation would smear spatial precision.
        if getattr(self, 'surround_decorrelation', False) and not gap_active:
            SL, SR = self._apply_surround_decorrelation(SL, SR)
            if BL is not None:
                BL, BR = self._apply_surround_decorrelation(BL, BR)

        # Consistent Sub Level: Attenuate redirected L+R by 10dB (0.316)
        LFE = (L + R) * 0.316
        
        # Apply Safe Dynamic EQ if enabled
        # Adds up to +4dB (1.58x) boost at extreme low volumes, tapering to 0dB at 0.5 peak
        if getattr(self, 'dynamic_eq', False):
            # We use a linear ramp that maxes at +4dB boost
            # dyn_eq_level is 0.0 to 1.0 (average peak)
            safe_level = max(getattr(self, 'dyn_eq_level', 0.0), np.max(np.abs(src_block)))
            boost_factor = 1.0 + (1.0 - min(1.0, safe_level * 2.0)) * 0.58
            ramp = self._get_ramp('_last_up_dyn', boost_factor, LFE.shape[0])
            LFE *= ramp

        out_block[:, 0] = FL
        out_block[:, 1] = FR
        if self.current_channels > 2: 
            # Apply Stage Equalizer Weighting (-3dB)
            c_weight = 0.707 if self.stage_equalizer else 1.0
            out_block[:, 2] = Center * c_weight
            
        if self.current_channels > 3: out_block[:, 3] = LFE if not self.lfe_muted else 0.0
        
        # Surround Replication
        for i in range(4, self.current_channels):
            if i == 4: out_block[:, 4] = SL
            elif i == 5: out_block[:, 5] = SR
            elif i == 6: out_block[:, 6] = BL if BL is not None else SL
            elif i == 7: out_block[:, 7] = BR if BR is not None else SR
            else:
                if i % 2 == 0: out_block[:, i] = SL
                else: out_block[:, i] = SR

        # ─── Post-Routing Global Effects ───
        
        # Apply Global Radar Panning or Headphone Tracking Counter-Rotation
        is_hp_tracking = getattr(self, 'headphone_tracking', False)
        if getattr(self, 'independent_panning', False) or is_hp_tracking:
            panning_x = self.gui_panning_x
            if is_hp_tracking:
                panning_x += getattr(self, '_dynamic_pan_offset', 0.0)
                panning_x = max(-1.0, min(1.0, panning_x)) # Clamp to bounds
                
            bal_l = 1.0 - max(0, panning_x)
            bal_r = 1.0 - max(0, -panning_x)
            
            # Apply to all lefts and all rights based on channel format
            out_block[:, 0] *= bal_l
            out_block[:, 1] *= bal_r
            if out_ch > 4:
                out_block[:, 4] *= bal_l
                out_block[:, 5] *= bal_r
            if out_ch > 6:
                out_block[:, 6] *= bal_l
                out_block[:, 7] *= bal_r

        # ─── Post-Upmix Effects Pipeline ───
        self._apply_post_effects(out_block, out_block.shape[1], skip_subsynth=force_passthrough, src_block=src_block)

        # True-Peak Limiter (All channels) - Replaces the legacy soft-clipper
        out_block = self._apply_true_peak_limiter(out_block)

        return out_block

    def output_callback(self, outdata, frames, time_info, status):
        if status: print(f"Output Warning: {status}")
        
        src_block = None
        with self.buffer_lock:
            # If we don't have enough data, fill with silence
            if self.available_samples < frames:
                outdata.fill(0)
                return

            # assemble full contiguous block BEFORE upmix (critical for OLA continuity)
            # Use pre-allocated buffer to avoid allocation under lock
            end1 = min(frames, self.ring_buffer_size - self.read_ptr)
            src_block = self._src_blk_pre[:frames, :self.max_input_ch]
            
            if end1 >= frames:
                src_block[:] = self.buffer[self.read_ptr : self.read_ptr + frames]
            else:
                src_block[:end1] = self.buffer[self.read_ptr : self.read_ptr + end1]
                src_block[end1:] = self.buffer[:frames - end1]
            
            # Update pointers immediately while under lock
            self.read_ptr = (self.read_ptr + frames) % self.ring_buffer_size
            self.available_samples -= frames
            
            # Aggressive Latency Control (Watermark-based)
            if self.available_samples > self.target_latency_samples + 512:
                diff = self.available_samples - self.target_latency_samples
                self.read_ptr = (self.read_ptr + diff) % self.ring_buffer_size
                self.available_samples -= diff

        # Now process OUTSIDE the lock
        # Note: src_block is a view into the pre-allocated buffer. 
        # Upmix processes it and writes to its own pre-allocated output buffer.
        outdata[:] = self._apply_upmix(src_block, out_ch=outdata.shape[1])

        # Capture mono downmix for AEC reference (outside buffer_lock would race,
        # but this is a simple read of outdata we just wrote — safe here)
        # AEC reference capture removed

    def run(self):
        print(f"Monitoring '{self.input_name}' -> '{self.output_name}'")
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()
        
        # Start tray icon in a separate thread
        if self.tray_icon:
            self.tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
            self.tray_thread.start()

        if self.simulation:
            print("RUNNING IN SIMULATION MODE")
            self.current_channels = 2
        else:
            self.multi_ch_target = self.max_output_ch
            print(f"Detected hardware: {self.sample_rate}Hz, {self.max_output_ch} channels max")
            self.open_input_stream()

            # Start with the hardware's native channel count for stability
            target_channels = self.max_output_ch if not self.simulation else 8
            self.open_output_stream(target_channels)

            # Start mic processing streams if enabled and devices configured
            if getattr(self, 'mic_enabled', False):
                if getattr(self, 'mic_name', ''):
                    self.open_mic_stream()
                if getattr(self, 'mic_output_name', ''):
                    self.open_mic_output_stream()
        
        sim_phase: float = 0.0
        try:
            while not self.stop_event.is_set():
                time.sleep(CHECK_INTERVAL)
                
                if self.simulation:
                    # Simulate signal switching every 10 seconds
                    sim_phase += CHECK_INTERVAL
                    if int(sim_phase / 10) % 2 == 1:
                        self.is_multichannel_detected = True
                        self.last_active_time = time.time()
                    else:
                        if time.time() - self.last_active_time > STABILITY_WINDOW:
                            self.is_multichannel_detected = False
                
                # Upmixing logic: Lock to max channels for stability.
                # Switching hardware channel counts mid-session often causes 
                # [PaErrorCode -9998] or 1-second "Switching..." silences.
                target_channels = self.max_output_ch if not self.simulation else 8
                
                if target_channels != self.current_channels:
                    # If we are in simulation, just set the value
                    if self.simulation:
                        print(f"SIMULATION: Switching to {target_channels} channels")
                        self.current_channels = target_channels
                    else:
                        try:
                            self.open_output_stream(target_channels)
                        except Exception as e:
                            print(f"Stream error: {e}")
                            # Fallback: keep trying with current channels
                            self.current_channels = target_channels 
                
                if self.peak_hit:
                    print("\n[CLIP WARNING] Signal hit 0dB! Try running with --gain 0.9")
                    self.peak_hit = False
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

def set_startup(enabled=True):
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = "XPDSS_AudioAnalyzer"
    # Ensure we use the absolute path to the script and current python executable
    script_path = os.path.abspath(__file__)
    cmd = f'"{sys.executable}" "{script_path}" --minimized'
    
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
        if enabled:
            winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, cmd)
            print(f"Startup enabled: {cmd}")
        else:
            try:
                winreg.DeleteValue(key, app_name)
                print("Startup disabled.")
            except FileNotFoundError:
                print("Startup was not enabled.")
        winreg.CloseKey(key)
    except Exception as e:
        print(f"Failed to modify startup registry: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Audio Channel Bypass")
    parser.add_argument("--input", default="CABLE Output", help="Input device name (Virtual)")
    parser.add_argument("--output", default="marantz-AVR", help="Output device name (AVR)")
    parser.add_argument("--sim", action="store_true", help="Enable simulation mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug print of head coordinates")

    parser.add_argument("--gain", type=float, default=1.0, help="Gain factor (e.g. 0.9 for -1dB)")
    parser.add_argument("--startup", choices=['on', 'off'], help="Enable/Disable Windows startup")
    parser.add_argument("--minimized", action="store_true", help="Start minimized to tray")
    parser.add_argument("--blocksize", type=int, default=0, help="Buffer block size (0 for auto)")
    parser.add_argument("--latency", default="low", help="Streaming latency ('low', 'high', or seconds)")
    parser.add_argument("--upmix", action="store_true", default=True, help="Enable Upmixer for 2-ch content")
    parser.add_argument("--upmix-gain", type=float, default=0.707, help="Upmix steering intensity (default 0.707)")
    parser.add_argument("--exclusive", action="store_true", help="Use WASAPI Exclusive mode (lowest latency, bypasses EQ APO)")
    parser.add_argument("--gui", action="store_true", help="Show graphical interface")
    args = parser.parse_args()

    if args.startup:
        set_startup(args.startup == 'on')
        sys.exit(0)

    # Need sounddevice now
    import sounddevice as sd

    app = AudioBypass(args.input, args.output, latency=args.latency, simulation=args.sim, gain=args.gain, start_minimized=args.minimized, blocksize=args.blocksize, exclusive_mode=args.exclusive, debug=args.debug)
    app.upmix_enabled = args.upmix
    app.upmix_steering_intensity = args.upmix_gain
    
    if args.gui:
        import gui_components
        # Run audio in background thread if GUI is active
        audio_thread = threading.Thread(target=app.run, daemon=True)
        audio_thread.start()
        
        # Launch GUI (blocks until window closed)
        gui_components.launch_gui(app)
    else:
        app.run()
