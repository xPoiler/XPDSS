import os
import math
from scipy.signal import lfilter, butter, sosfilt, sosfilt_zi, iirnotch
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from scipy.signal import firwin2
import ctypes
import sys
import threading
import queue
import time
import argparse
import numpy as np
import json
from adaptive_control import ADAPTIVE_PARAMS, AdaptiveController
from adaptive_renderer import AdaptiveFeatureEngine
from spectral_upmix import STEER_CINEMA, STEER_GAMING, STEER_OFF, SpectralUpmixer
from xpdss_state import (
    apply_adaptive_state_to_object,
    apply_profile_to_object,
    apply_secondary_output_state_to_object,
    build_adaptive_state_from_object,
    build_profile_from_object,
    build_secondary_output_state_from_object,
)

try:
    import sounddevice as sd
except Exception:  # PortAudio missing (e.g. headless test runs)
    sd = None

IS_WINDOWS = sys.platform == 'win32'

# Constants for detection
SURROUND_SILENCE_FLOOR = 10 ** (-70 / 20)  # center/surround RMS above this = real content
STABILITY_WINDOW = 3.0
ANALYSIS_STRIDE = 5     # every Nth input block goes to the analysis thread
CHECK_INTERVAL = 0.2
BLOCK_SIZE = 0           # 0 allows WASAPI to choose the most stable hardware buffer

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

KERNEL32 = USER32 = None
SetWindowLongPtr = GetWindowLongPtr = None
TASKBAR_LIST = None

if IS_WINDOWS:
    from ctypes import wintypes

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
        except Exception:
            pass
        return None

    TASKBAR_LIST = get_taskbar_list()


def _frac_delay(sig, hist, d, frac):
    """Fractional delay of `sig` by d + frac samples, using `hist` (older samples) as prefix.

    Linear interpolation between taps d and d+1. `hist` must hold >= d + 1 samples.
    """
    n = sig.shape[0]
    ext = np.concatenate((hist, sig))
    start = hist.shape[0] - d
    a = ext[start:start + n]
    b = ext[start - 1:start - 1 + n]
    return a * (1.0 - frac) + b * frac


def _push_history(hist, sig):
    """Return the newest len(hist) samples of hist + sig."""
    h = hist.shape[0]
    if sig.shape[0] >= h:
        return sig[-h:].copy()
    return np.concatenate((hist[sig.shape[0]:], sig))


class AudioBypass:
    def __init__(self, input_name, output_name, latency='high', simulation=False, gain=1.0, start_minimized=False, blocksize=0, exclusive_mode=False, debug=False, config_file=None):
        self.debug = debug

        self.input_name = input_name
        self.output_name = output_name
        self.exclusive_mode = exclusive_mode
        self.latency = latency
        self.simulation = simulation
        # Audio health counters (shown in the GUI): which kind of dropout, if
        # any, happened - output late (processing missed the sound card's
        # deadline), input overflow, input buffer empty (the virtual cable
        # delivered late), slow blocks, hard resyncs.
        self.io_stats = {'out_underflow': 0, 'in_overflow': 0, 'buffer_empty': 0,
                         'slow_blocks': 0, 'resync': 0}
        if not simulation:
            # The audio callbacks are Python: by default another thread (GUI,
            # analysis) can hold the interpreter for up to 5 ms before the
            # callback gets to run. 1 ms keeps callbacks on time.
            sys.setswitchinterval(0.001)
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
        
        self.config_file = config_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.config_lock = threading.Lock()
        # Serializes stream open/close between the GUI, calibration worker and run() loop
        self.stream_lock = threading.RLock()
        self.webcam_name = ""
        self.webcam_index = 0
        # Device names must be known before the streams are sized; the full
        # profile is applied later by load_config().
        try:
            with open(self.config_file, 'r') as f:
                d = json.load(f)
            self.input_name = d.get("input_device", input_name)
            self.output_name = d.get("output_device", output_name)
        except Exception:
            self.input_name = input_name
            self.output_name = output_name

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
        self.ring_buffer_size = int(self.sample_rate * 0.25) # 250ms safety wrap
        # Input reserve: starts at 10 ms and grows by 5 ms (up to 80 ms) each
        # time the buffer runs dry, so it settles at what the virtual cable's
        # delivery pattern actually needs (VB-Cable hands over irregular
        # 256/512-sample chunks; a fixed 10 ms ran dry every few seconds).
        self.target_latency_samples = int(self.sample_rate * 0.010)
        self._target_max_samples = int(self.sample_rate * 0.080)
        self._in_primed = False
        self.buffer = np.zeros((self.ring_buffer_size, self.max_input_ch), dtype=np.float32)
        self.write_ptr = 0
        self.read_ptr = 0
        self.available_samples = 0
        self.buffer_lock = threading.Lock()
        
        # Pre-allocate for analysis
        self.pool_size = 10
        self.buffer_pool = [np.zeros((8192, self.max_input_ch), dtype=np.float32) for _ in range(self.pool_size)]
        self.pool_index = 0
        
        # Two spare pool slots: the slot being refilled can never be one that is
        # still queued or currently being analysed.
        self.analysis_queue = queue.Queue(maxsize=self.pool_size - 2)
        self._drift_fill_ema = None
        self._callback_error_count = 0
        self._last_callback_error = None
        self.stop_event = threading.Event()
        self.analysis_thread = None
        self.input_stream = None
        self.output_stream = None
        self.secondary_output_stream = None
        self.secondary_output_device_id = None
        self.secondary_output_rate = self.sample_rate
        self.secondary_output_channels = 2
        self.secondary_output_max_ch = 0
        self.secondary_output_buffer_lock = threading.Lock()
        self.secondary_output_ring_buffer_size = int(self.sample_rate * 0.25)
        self.secondary_output_buffer = np.zeros((self.secondary_output_ring_buffer_size, self.secondary_output_channels), dtype=np.float32)
        self.secondary_output_write_ptr = 0
        self.secondary_output_read_ptr = 0
        self.secondary_output_available_samples = 0
        self.secondary_output_target_latency_samples = int(self.sample_rate * 0.040)
        self._secondary_output_read_pre = np.zeros((8192, self.secondary_output_channels), dtype=np.float32)
        self._secondary_output_hp_sos = None
        self._secondary_output_lp_sos = None
        self._secondary_output_hp_zi = {}
        self._secondary_output_lp_zi = {}
        self._secondary_output_hist = {}
        self._secondary_drift_fill_ema = None
        self._secondary_output_primed = False
        apply_secondary_output_state_to_object(self, {})
        # Adaptive renderer: one on/off switch (saved); live values and pins are runtime only
        self.adaptive_enabled = False
        self.adaptive_snapshot = {}
        self.adaptive_values = {}      # live amounts per slider while adaptive (or its glide back) runs
        self.adaptive_pinned = set()   # sliders the user dragged while adaptive was on
        self.adaptive_engine = AdaptiveFeatureEngine(self.sample_rate)
        self.adaptive_controller = AdaptiveController()
        
        # Profile-scoped settings (upmix, enhancements, head tracking geometry...)
        # live in xpdss_state.PROFILE_DEFAULTS and are applied below by
        # _apply_profile_dict() / load_config(). Only runtime state goes here.
        self.current_channels = 0
        self.failed_channel_counts = set()

        # Multi-Profile Output Management
        self.current_mode = "speakers" # "speakers" or "headphones"
        self.speakers_config = {}
        self.headphones_config = {}

        self.dyn_eq_level = 0.0 # Track master volume smoothed (Dynamic EQ)
        self.bypass_all = False
        # Output headroom (dB, global): every channel in every mode (bypass
        # included) leaves this much lower, so the center of an upmixed loud
        # master fits without the limiter ducking everything. Same cut for
        # all channels: the LFE/mains balance and the passthrough/upmix match
        # are unchanged; raise the AVR volume once instead.
        self.output_headroom_db = 4.0

        self.correlation_smooth = 0.0 # 0.0 (Wide) to 1.0 (Mono)

        # --- Real-time Pre-allocations ---
        self._src_blk_pre = np.zeros((8192, self.max_input_ch), dtype=np.float32)
        self._upmix_out_buf = np.zeros((8192, 8), dtype=np.float32) # Standard 7.1/8ch max

        # Widening Mode (Center Spread bleed to surrounds with delay)
        self._widening_was_on = False  # Edge detect for buffer zero-fill
        self.widening_delay_ms = 20.0
        self.widening_buffer_size = max(4096, int(self.sample_rate))
        self.widening_buffer = np.zeros(self.widening_buffer_size, dtype=np.float32)
        self.widening_write_ptr = 0

        # #5 Vocal Lock
        self._vl_prev_vocal_score = 0.0  # Smoothed vocal detection score

        # #9 Spectral Tilt Correction
        self._tilt_update_cnt = 0
        self._tilt_taps = None
        self._tilt_filter_zi = {}

        # #11 Subharmonic Synthesizer
        self._sub_filter_zi = {}
        self._sub_out_sos = self._sub_hp_sos = self._sub_vocal_sos = None
        self._sub_drive_sos = None

        # #13 Harmonic Bass Exciter (PBE)
        self.harmonic_gain = 0.7

        # #14 Fletcher-Munson Loudness Compensation (driven by the Dynamic EQ toggle)
        self._last_loudness_db = None
        self._loud_b_l = self._loud_a_l = self._loud_b_h = self._loud_a_h = None
        self._loudness_filter_zi = {}

        # Output peak limiter (always on, final safety stage)
        self._limiter_buffer = None

        # Head tracking runtime
        self.head_tracker = None

        # Dynamics / enhancement runtime state
        self._agc_gain_smooth = 1.0
        self._clarity_gain_smooth = 0.0
        self._clarity_mud_sos = None
        self._clarity_peak_sos = None
        self._clarity_zi = {}
        self._room_delay_buf = None
        self._room_delay_ptr = 0
        self._room_lp_sos = None
        self._room_lp_zi = {}
        self._air_hp_sos = None
        self._air_hp_zi = {}

        # ─── Program Level Estimator (always-on, no toggle) ──────────────
        # Very slow RMS tracker that all dynamics processors use as a reference
        # so their thresholds adapt to whatever volume the signal arrives at.
        self._program_level_db = -20.0  # design-center reference level
        self._program_level_smooth = None  # None = seed from first measurement
        self._plr_reference_db = -20.0  # the level all thresholds were designed for

        self.gui_window = None
        self._gui_requested = threading.Event()

        # Initialize defaults for Profile Safety
        self._apply_profile_dict({})
        # Shared filter bank: must exist even on a first run without config.json
        self._init_multiband_filters()

        # Load persistent settings
        self.load_config()
        self.last_mode = None

        # Set process priority to High on Windows
        if IS_WINDOWS:
            try:
                import psutil
                psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)
                print("Process priority set to HIGH")
            except Exception:
                try:
                    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080) # HIGH_PRIORITY_CLASS
                    print("Process priority set to HIGH (via ctypes)")
                except Exception:
                    print("Warning: Could not set process priority.")

        if not simulation:
            print(f"Detected hardware: {self.sample_rate}Hz, {self.max_output_ch} channels max")

        self.last_active_time = time.time()
        self.is_multichannel_detected = False

        if self.start_minimized:
            self.hide_console()

        self._setup_tray()

    def _setup_tray(self):
        try:
            import pystray
            from PIL import Image
        except Exception as e:
            print(f"Tray unavailable: {e}")
            self.tray_icon = None
            return
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
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
        if not KERNEL32:
            return
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
            # Reveal existing window (always marshalled onto the Qt thread)
            from PySide6.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(self.gui_window, "bring_to_front", Qt.QueuedConnection)
        else:
            # Qt must run on the main thread: main() waits for this request.
            self._gui_requested.set()

    def show_console(self):
        if not KERNEL32:
            return
        hwnd = KERNEL32.GetConsoleWindow()
        if hwnd:
            # Re-add to Taskbar if we are showing (optional, but requested for visibility)
            if TASKBAR_LIST:
                TASKBAR_LIST.contents.lpVtbl.contents.AddTab(TASKBAR_LIST, hwnd)

            USER32.ShowWindow(hwnd, SW_SHOW)
            self.console_hidden = False

    def stop(self, icon=None, item=None):
        if self.stop_event.is_set():
            return
        print("Stopping application...")
        self.stop_event.set()
        self.save_config()
        if self.tray_icon:
            try: self.tray_icon.stop()
            except Exception: pass

        self.stop_head_tracker()
        with self.stream_lock:
            for attr in ('input_stream', 'output_stream'):
                stream = getattr(self, attr)
                if stream:
                    try: stream.stop(); stream.close()
                    except Exception: pass
                    setattr(self, attr, None)
            self.close_secondary_output_stream()

        # Quit the Qt event loop too, otherwise the process keeps running
        if self.gui_window is not None:
            try:
                from PySide6.QtCore import QMetaObject, Qt
                from PySide6.QtWidgets import QApplication
                qt_app = QApplication.instance()
                if qt_app is not None:
                    QMetaObject.invokeMethod(qt_app, "quit", Qt.QueuedConnection)
            except Exception as e:
                print(f"Could not quit GUI: {e}")

    @staticmethod
    def _is_virtual_cable(device_name):
        name = device_name.lower()
        return 'cable' in name or 'vb-audio' in name or 'voicemeeter' in name

    def _find_device(self, name, is_input=True, allow_fallback=True):
        if sd is None:
            raise RuntimeError("sounddevice/PortAudio is not available.")
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

        def _has_dir(d):
            return d['max_input_channels'] > 0 if is_input else d['max_output_channels'] > 0

        def _result(i, d):
            return i, (d['max_input_channels'] if is_input else d['max_output_channels']), d['default_samplerate']

        candidates = [(i, d) for i, d in enumerate(devices) if d['hostapi'] == wasapi_idx and _has_dir(d)]

        # Exact name match first, then substring match
        wanted = (name or "").strip().lower()
        if wanted:
            for i, d in candidates:
                if d['name'].strip().lower() == wanted:
                    return _result(i, d)
            for i, d in candidates:
                if wanted in d['name'].lower():
                    return _result(i, d)

        kind = 'Input' if is_input else 'Output'
        if not allow_fallback:
            raise ValueError(f"{kind} device '{name}' not found.")

        # Fallback: first available WASAPI device. For outputs never pick a
        # virtual cable: Windows plays INTO the cable, so sending our output
        # there would create a feedback loop.
        print(f"Warning: Device '{name}' not found. Falling back to another WASAPI device.")
        for i, d in candidates:
            if not is_input and self._is_virtual_cable(d['name']):
                continue
            print(f"Fallback match: '{d['name']}'")
            return _result(i, d)

        raise ValueError(f"Error: No usable WASAPI {kind} devices found.")

    def _analysis_worker(self):
        while not self.stop_event.is_set():
            try:
                # Get data from queue with timeout
                data = self.analysis_queue.get(timeout=0.1)
                
                # Perform Peak calculation (lighter than RMS)
                # Perform analysis on channels
                # data is (frames, channels)
                frames, chs = data.shape
                if chs < 2:
                    continue

                peak = np.max(np.abs(data), axis=0)
                if chs <= 2:
                    # Stereo sources still feed GUI metering, width tracking, and
                    # adaptive analysis below; they simply are not discrete surround.
                    self.is_multichannel_detected = False
                is_discrete = self._block_has_discrete_surround(data)

                # Require two consecutive discrete analysis blocks before switching
                # in (a single stray block never triggers); staying in is refreshed
                # by every discrete block and released after STABILITY_WINDOW.
                if is_discrete:
                    self._discrete_hits = getattr(self, '_discrete_hits', 0) + 1
                    if self._discrete_hits >= 2 or self.is_multichannel_detected:
                        self.last_active_time = time.time()
                        self.is_multichannel_detected = True
                else:
                    self._discrete_hits = 0

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
                    except Exception: pass

                try:
                    self._adaptive_step(data, frames * ANALYSIS_STRIDE / self.sample_rate)
                except Exception as adaptive_error:
                    print(f"Adaptive analysis error: {adaptive_error}")

                if time.time() - self.last_active_time > STABILITY_WINDOW:
                    self.is_multichannel_detected = False
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {e}")

    @staticmethod
    def _block_has_discrete_surround(data):
        """True when a (frames, channels) input block carries real discrete
        center/surround content.

        A stereo source on a multichannel cable leaves the center/surround
        channels digitally silent, so any real signal (RMS > -70 dBFS, above
        dither and noise floors) counts, however quiet relative to the fronts.
        Channels that are just an L+R / L-R derivative (a third-party matrix
        upmix, correlation 0.95-0.999) do not count. The LFE is ignored.
        """
        chs = data.shape[1]
        if chs <= 2:
            return False
        rms = np.sqrt(np.mean(np.square(data, dtype=np.float64), axis=0))
        live = [i for i in (2, 4, 5, 6, 7) if i < chs and rms[i] > SURROUND_SILENCE_FLOOR]
        if not live:
            return False
        L = data[:, 0]
        R = data[:, 1]
        fronts_silent = max(rms[0], rms[1]) < SURROUND_SILENCE_FLOOR
        for idx in live:
            # Signal with silent fronts: no upmix can create that
            if fronts_silent:
                return True
            ref = (L + R) if idx == 2 else (L - R)
            if not np.any(ref):
                return True
            corr = np.abs(np.corrcoef(data[:, idx], ref)[0, 1])
            # 0.95-0.999: a derived (matrixed) channel. > 0.999 is an exact
            # mirror, which a discrete test file can contain.
            if np.isnan(corr) or corr < 0.95 or corr > 0.999:
                return True
        return False

    def get_active_channels(self, data):
        # This is no longer used in the callback; moved to worker
        pass

    def get_adaptive_snapshot(self):
        return dict(getattr(self, 'adaptive_snapshot', {}))

    # ─── Adaptive renderer ──────────────────────────────────────────────
    # While adaptive is on, the amount sliders of the enhancement effects
    # follow the input (see adaptive_control). The profile attributes keep
    # the user's own values, which are what gets saved; the live values sit
    # in `adaptive_values`. A slider the user drags is pinned to its manual
    # value until adaptive is switched off and on again.

    def _manual_adaptive_values(self):
        return {key: float(getattr(self, key)) for key in ADAPTIVE_PARAMS}

    def _live_param(self, key):
        """Value an effect should use right now: adaptive unless pinned/off."""
        if key not in self.adaptive_pinned:
            value = self.adaptive_values.get(key)
            if value is not None:
                return value
        return getattr(self, key)

    def is_adaptive_driven(self, key):
        return bool(self.adaptive_enabled and key in ADAPTIVE_PARAMS and key not in self.adaptive_pinned)

    def set_adaptive_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self.adaptive_enabled:
            return
        self.adaptive_pinned = set()
        if enabled:
            # Start from the user's values so the sliders glide, not jump
            self.adaptive_controller.reset(self._manual_adaptive_values())
            self.adaptive_values = self.adaptive_controller.values()
        self.adaptive_enabled = enabled
        self.save_config()

    def set_manual_param(self, key, value):
        """A user edit of a slider. While adaptive is on, it pins that slider."""
        setattr(self, key, value)
        if self.adaptive_enabled and key in ADAPTIVE_PARAMS:
            self.adaptive_pinned = self.adaptive_pinned | {key}

    def _adaptive_step(self, data, dt):
        """Analysis-thread step: update measurements, then the live values."""
        snapshot = self.adaptive_engine.analyze_block(data, dt=dt)
        if snapshot is not None:
            self.adaptive_snapshot = snapshot.to_dict()
        if self.adaptive_enabled:
            if not self.adaptive_values:
                self.adaptive_controller.reset(self._manual_adaptive_values())
            # Silence holds the values (no drifting during pauses)
            fresh = self.adaptive_engine.last_block_active
            self.adaptive_values = self.adaptive_controller.update(
                self.adaptive_snapshot if fresh else None, dt,
                output_mode=getattr(self, 'current_mode', 'speakers'))
        elif self.adaptive_values:
            # Switched off: glide back to the user's values, then hand over
            done = self.adaptive_controller.glide_to(self._manual_adaptive_values(), dt)
            self.adaptive_values = {} if done else self.adaptive_controller.values()

    def _valid_sos(self, sos):
        """True when a SciPy SOS filter is initialized and safe for sosfilt."""
        try:
            arr = np.asarray(sos)
            return arr.ndim == 2 and arr.shape[1] == 6 and arr.shape[0] > 0
        except Exception:
            return False

    def _ensure_sos_filters(self, *attrs):
        """Initialize shared SOS filters on demand; never let real-time DSP crash.

        Some adaptive paths can enable DSP modules even when their manual toggle
        was off at startup.  Those modules still need the shared filter bank that
        older code only initialized lazily from other feature paths.
        """
        tracked_attrs = tuple(attrs)
        previous_signatures = {
            attr: tuple(np.asarray(getattr(self, attr, None)).shape)
            if self._valid_sos(getattr(self, attr, None)) else None
            for attr in tracked_attrs
        }
        if all(previous_signatures[attr] is not None for attr in tracked_attrs):
            return True
        try:
            self._init_multiband_filters()
        except Exception as exc:
            last_msg = getattr(self, '_last_filter_init_error', None)
            msg = str(exc)
            if msg != last_msg:
                print(f"Filter init error: {exc}")
                self._last_filter_init_error = msg
            return False

        changed = []
        for attr in tracked_attrs:
            current = getattr(self, attr, None)
            if not self._valid_sos(current):
                return False
            current_signature = tuple(np.asarray(current).shape)
            if previous_signatures.get(attr) != current_signature:
                changed.append(attr)
        if changed:
            self._reset_sos_dependents(*changed)
        return True

    def _reset_sos_dependents(self, *sos_attrs):
        reset_map = {
            '_clarity_mud_sos': ('_clarity_mud_zi',),
            '_clarity_peak_sos': ('_clarity_zi',),
            '_room_lp_sos': ('_room_lp_zi', '_room_delay_buf', '_room_delay_ptr'),
            '_air_hp_sos': ('_air_hp_zi',),
        }
        for sos_attr in sos_attrs:
            for dep in reset_map.get(sos_attr, ()):
                current = getattr(self, dep, None)
                if isinstance(current, dict):
                    current.clear()
                elif isinstance(current, np.ndarray):
                    setattr(self, dep, None)
                elif dep.endswith('_ptr'):
                    setattr(self, dep, 0)
                else:
                    setattr(self, dep, {} if dep.endswith('_zi') else None)

    def _ensure_sos_zi(self, store, key, sos):
        zi = store.get(key)
        expected_shape = (np.asarray(sos).shape[0], 2)
        if not isinstance(zi, np.ndarray) or zi.shape != expected_shape:
            store[key] = sosfilt_zi(sos) * 0.0
        return store[key]

    def _reset_secondary_output_runtime(self):
        channels = max(1, int(getattr(self, 'secondary_output_channels', 2) or 1))
        self.secondary_output_ring_buffer_size = int(self.sample_rate * 0.25)
        self.secondary_output_target_latency_samples = int(self.sample_rate * 0.040)
        self.secondary_output_buffer = np.zeros((self.secondary_output_ring_buffer_size, channels), dtype=np.float32)
        self.secondary_output_write_ptr = 0
        self.secondary_output_read_ptr = 0
        self.secondary_output_available_samples = 0
        self._secondary_output_read_pre = np.zeros((8192, channels), dtype=np.float32)
        self._secondary_output_hist.clear()
        self._secondary_output_hp_zi.clear()
        self._secondary_output_lp_zi.clear()
        self._secondary_drift_fill_ema = None
        self._secondary_output_primed = False

    def _resolve_secondary_output_device(self):
        self.secondary_output_device_id = None
        self.secondary_output_max_ch = 0
        self.secondary_output_rate = self.sample_rate
        if not getattr(self, 'secondary_output_device', ''):
            return False
        try:
            # Never fall back to another device: the shaker bus must not end up
            # on the main speakers.
            dev_id, max_ch, dev_rate = self._find_device(self.secondary_output_device, is_input=False, allow_fallback=False)
        except Exception as e:
            print(f"Secondary output unavailable: {e}")
            return False
        self.secondary_output_device_id = dev_id
        self.secondary_output_max_ch = max_ch
        self.secondary_output_rate = dev_rate
        self.secondary_output_channels = 2 if max_ch >= 2 else 1
        return True

    def close_secondary_output_stream(self):
        if self.secondary_output_stream:
            try:
                self.secondary_output_stream.stop()
                self.secondary_output_stream.close()
            except Exception:
                pass
            self.secondary_output_stream = None

    @staticmethod
    def _wasapi_settings(exclusive=False, auto_convert=False):
        """Build WasapiSettings, tolerating sounddevice versions without auto_convert."""
        if sd is None or not hasattr(sd, 'WasapiSettings'):
            return None
        if not exclusive and not auto_convert:
            return None
        try:
            return sd.WasapiSettings(exclusive=exclusive, auto_convert=auto_convert)
        except TypeError:
            return sd.WasapiSettings(exclusive=exclusive) if exclusive else None

    def open_secondary_output_stream(self):
        if self.simulation:
            return
        with self.stream_lock:
            self.close_secondary_output_stream()
            if not getattr(self, 'secondary_output_enabled', False):
                return
            if not self._resolve_secondary_output_device():
                return
            with self.secondary_output_buffer_lock:
                self._reset_secondary_output_runtime()
            # The shaker bus is rendered at the main sample rate. If the shaker
            # device runs at another rate, let WASAPI convert it.
            needs_convert = int(self.secondary_output_rate) != int(self.sample_rate)
            try:
                self.secondary_output_stream = sd.OutputStream(
                    device=self.secondary_output_device_id,
                    samplerate=self.sample_rate,
                    blocksize=self.blocksize,
                    dtype='float32',
                    channels=self.secondary_output_channels,
                    callback=self.secondary_output_callback,
                    latency='low',
                    extra_settings=self._wasapi_settings(auto_convert=needs_convert),
                )
                self.secondary_output_stream.start()
                print(f"Secondary Output active: {self.secondary_output_channels} channels -> {self.secondary_output_device}")
            except Exception as e:
                print(f"Failed to open secondary output stream: {e}")
                self.secondary_output_stream = None

    def refresh_secondary_output_filters(self):
        self._init_multiband_filters()
        self._secondary_output_hp_zi.clear()
        self._secondary_output_lp_zi.clear()

    def refresh_secondary_output_stream(self):
        self.close_secondary_output_stream()
        if getattr(self, 'secondary_output_enabled', False):
            self.open_secondary_output_stream()

    def set_secondary_output_enabled(self, enabled):
        self.secondary_output_enabled = bool(enabled)
        self.save_config()
        self.refresh_secondary_output_stream()

    def set_secondary_output_device(self, device_name):
        self.secondary_output_device = device_name or ""
        self.save_config()
        self.refresh_secondary_output_stream()

    def _apply_secondary_output_delay(self, mono_block, key='mono'):
        delay_samples = int(max(0.0, getattr(self, 'secondary_output_delay_ms', 0.0)) * self.sample_rate / 1000.0)
        if delay_samples <= 0:
            self._secondary_output_hist.pop(key, None)
            return mono_block
        hist = self._secondary_output_hist.get(key)
        if hist is None or hist.shape[0] != delay_samples:
            resized = np.zeros(delay_samples, dtype=np.float32)
            if hist is not None:
                keep = min(hist.shape[0], delay_samples)
                resized[-keep:] = hist[-keep:]
            hist = resized
        ext = np.concatenate((hist, np.asarray(mono_block, dtype=np.float32)))
        n = mono_block.shape[0]
        self._secondary_output_hist[key] = ext[-delay_samples:].copy()
        return ext[:n]

    def _filter_secondary_output_mono(self, mono_block, key='mono'):
        block = np.asarray(mono_block, dtype=np.float32).copy()
        if self._secondary_output_hp_sos is not None:
            zi = self._ensure_sos_zi(self._secondary_output_hp_zi, key, self._secondary_output_hp_sos)
            block, self._secondary_output_hp_zi[key] = sosfilt(self._secondary_output_hp_sos, block, zi=zi)
        if self._secondary_output_lp_sos is not None:
            zi = self._ensure_sos_zi(self._secondary_output_lp_zi, key, self._secondary_output_lp_sos)
            block, self._secondary_output_lp_zi[key] = sosfilt(self._secondary_output_lp_sos, block, zi=zi)
        return block

    def _build_secondary_output_block(self, speaker_block, src_block=None):
        if speaker_block is None or speaker_block.size == 0:
            return None
        frames = speaker_block.shape[0]
        ch = speaker_block.shape[1]
        source_mode = getattr(self, 'secondary_output_source_mode', 'lfe_or_bass_mix')
        mono_sum = getattr(self, 'secondary_output_mono_sum', True)
        gain = float(10.0 ** (getattr(self, 'secondary_output_gain_db', 0.0) / 20.0))

        if source_mode == 'front_bass_mix' or ch <= 3:
            base_left = speaker_block[:, 0]
            base_right = speaker_block[:, 1] if ch > 1 else speaker_block[:, 0]
        else:
            lfe = speaker_block[:, 3] if ch > 3 else speaker_block[:, 0]
            if ch > 1:
                stereo_mix = 0.5 * (speaker_block[:, 0] + speaker_block[:, 1])
            else:
                stereo_mix = speaker_block[:, 0]
            base_left = lfe + (0.25 * stereo_mix)
            base_right = lfe + (0.25 * stereo_mix)

        if mono_sum or self.secondary_output_channels == 1:
            mono = 0.5 * (base_left + base_right)
            mono = self._filter_secondary_output_mono(mono, key='mono')
            mono = self._apply_secondary_output_delay(mono, key='mono')
            mono *= gain
            mono = np.clip(mono, -1.0, 1.0)
            if self.secondary_output_channels == 1:
                return mono.reshape(frames, 1)
            return np.repeat(mono.reshape(frames, 1), self.secondary_output_channels, axis=1)

        left = self._filter_secondary_output_mono(base_left, key='left')
        right = self._filter_secondary_output_mono(base_right, key='right')
        left = self._apply_secondary_output_delay(left, key='left')
        right = self._apply_secondary_output_delay(right, key='right')
        block = np.column_stack([left, right]).astype(np.float32)
        block *= gain
        return np.clip(block, -1.0, 1.0)

    def _enqueue_secondary_output_block(self, block):
        if block is None or block.size == 0:
            return
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        frames, channels = block.shape
        with self.secondary_output_buffer_lock:
            if self.secondary_output_buffer.shape[1] != channels or self.secondary_output_buffer.shape[0] != self.secondary_output_ring_buffer_size:
                self.secondary_output_channels = channels
                self._reset_secondary_output_runtime()
            size = self.secondary_output_ring_buffer_size
            if frames >= size:
                block = block[-size:]
                frames = block.shape[0]
            overflow = max(0, self.secondary_output_available_samples + frames - size)
            if overflow:
                self.secondary_output_read_ptr = (self.secondary_output_read_ptr + overflow) % size
                self.secondary_output_available_samples -= overflow
            end1 = min(frames, size - self.secondary_output_write_ptr)
            self.secondary_output_buffer[self.secondary_output_write_ptr:self.secondary_output_write_ptr + end1] = block[:end1]
            if end1 < frames:
                self.secondary_output_buffer[:frames - end1] = block[end1:]
            self.secondary_output_write_ptr = (self.secondary_output_write_ptr + frames) % size
            self.secondary_output_available_samples += frames

    # ─── Clock-drift compensation ──────────────────────────────────────
    # Input (virtual cable) and outputs run on independent clocks. Instead of
    # dropping or skipping whole chunks when they drift apart, each block
    # consumes one sample more or less and is resampled back to `frames`.
    @staticmethod
    def _drift_adjust(ema, fill_after, target, frames):
        ema = float(fill_after) if ema is None else ema * 0.98 + float(fill_after) * 0.02
        band = max(32, frames // 4)
        if ema > target + band:
            return ema, 1
        if ema < target - band:
            return ema, -1
        return ema, 0

    @staticmethod
    def _resample_block(src, frames, out):
        n_in = src.shape[0]
        if n_in == frames:
            out[:frames] = src
            return out[:frames]
        pos = np.arange(frames, dtype=np.float64) * (n_in / frames)
        idx = np.arange(n_in, dtype=np.float64)
        for ch in range(src.shape[1]):
            out[:frames, ch] = np.interp(pos, idx, src[:, ch])
        return out[:frames]

    @staticmethod
    def _ring_read(buffer, read_ptr, count, dest):
        size = buffer.shape[0]
        end1 = min(count, size - read_ptr)
        dest[:end1] = buffer[read_ptr:read_ptr + end1, :dest.shape[1]]
        if end1 < count:
            dest[end1:count] = buffer[:count - end1, :dest.shape[1]]

    def secondary_output_callback(self, outdata, frames, time_info, status):
        if status:
            print(f"Secondary Output Warning: {status}")
        try:
            out_ch = outdata.shape[1]
            with self.secondary_output_buffer_lock:
                size = self.secondary_output_ring_buffer_size
                target = self.secondary_output_target_latency_samples
                available = self.secondary_output_available_samples
                # Prime to the target latency before playing (and after an underrun)
                if not self._secondary_output_primed:
                    if available < target + frames:
                        outdata.fill(0)
                        return
                    self._secondary_output_primed = True
                excess = available - frames - target
                if excess > max(2048, 4 * frames):
                    self.secondary_output_read_ptr = (self.secondary_output_read_ptr + excess) % size
                    self.secondary_output_available_samples -= excess
                    available = self.secondary_output_available_samples
                    self._secondary_drift_fill_ema = None
                self._secondary_drift_fill_ema, adj = self._drift_adjust(
                    self._secondary_drift_fill_ema, available - frames, target, frames)
                n_in = frames + adj
                if available < n_in:
                    self._secondary_output_primed = False
                    outdata.fill(0)
                    return
                if self._secondary_output_read_pre.shape[0] < n_in + 1 or self._secondary_output_read_pre.shape[1] != out_ch:
                    self._secondary_output_read_pre = np.zeros((max(n_in + 1, 8192), out_ch), dtype=np.float32)
                raw = self._secondary_output_read_pre[:n_in, :out_ch]
                self._ring_read(self.secondary_output_buffer, self.secondary_output_read_ptr, n_in, raw)
                self.secondary_output_read_ptr = (self.secondary_output_read_ptr + n_in) % size
                self.secondary_output_available_samples -= n_in
                if adj == 0:
                    outdata[:] = raw
                else:
                    self._resample_block(raw, frames, outdata)
        except Exception as e:
            outdata.fill(0)
            self._report_callback_error("secondary output", e)

    def _report_callback_error(self, where, exc):
        """Log callback exceptions (rate-limited) instead of letting them kill the stream."""
        self._callback_error_count += 1
        msg = f"{type(exc).__name__}: {exc}"
        now = time.time()
        last = self._last_callback_error
        if last is None or last[0] != msg or now - last[1] > 5.0:
            import traceback
            print(f"[{where} callback error #{self._callback_error_count}] {msg}")
            traceback.print_exc()
            self._last_callback_error = (msg, now)
        # Most DSP failures come from stale state after a format change: rebuild it.
        self._reset_dsp_state()

    # Lazily created DSP state that depends on sample rate, block size or channel count
    _DSP_STATE_ATTRS = (
        '_limiter_buffer', '_limiter_gr_zi', '_limiter_req_hist', '_limiter_hold_hist', '_tilt_taps', '_tilt_hist',
        '_fdw_lo_sos', '_fdw_hi_sos', '_fdw_zi',
        '_ext_xfeed_sos', '_ext_delay_buf',
        '_exciter_lp_sos', '_exciter_dc_zi',
        '_decorr_ap_states', '_tr_fast_zi', '_ud_ref_db',
        '_trans_hist', '_trans_lp_zi_L', '_trans_delay_smooth_L',
        '_ht_cs_L', '_ht_delay_buf', '_shadow_lp_zi', '_dist_bass_zi', '_bound_lp_zi_L', '_abs_lp_zi_L',
        '_sub_period_est', '_sub_need_smooth', '_sub_last_avoid',
        '_vl_voice_mask', '_dlg_bp_sos', '_dlg_bp_zi', '_air_post_zi',
        '_rs_out_pre', '_spectral', '_delay_hist', '_spectral_was_active',
        '_bass_split_sos', '_amb_shelf_sos', '_cls_stats', '_tilt_mono_hist', '_tilt_taps_designed',
    )

    def _reset_dsp_state(self):
        for attr in self._DSP_STATE_ATTRS:
            if attr in self.__dict__:
                delattr(self, attr)
        self._limiter_buffer = None
        self._tilt_taps = None
        self._tilt_update_cnt = 0
        for store in ('_tilt_filter_zi', '_sub_filter_zi', '_loudness_filter_zi', '_clarity_zi',
                      '_room_lp_zi', '_air_hp_zi'):
            setattr(self, store, {})
        if hasattr(self, '_clarity_mud_zi'):
            self._clarity_mud_zi = {}
        self._room_delay_buf = None
        self._room_delay_ptr = 0
        self._last_loudness_db = None
        self._upmix_out_buf = np.zeros((8192, 8), dtype=np.float32)
        for key in [k for k in self.__dict__ if k.startswith('_ramp_buf_')]:
            delattr(self, key)

    def open_input_stream(self):
        with self.stream_lock:
            if self.input_stream: return
            try:
                # Re-allocate buffers if channel count or sample rate changed
                with self.buffer_lock:
                    ring_size = int(self.sample_rate * 0.25)
                    if self.buffer.shape[1] != self.max_input_ch or self.buffer.shape[0] != ring_size:
                        print(f"[Stream] Resizing input buffer to {self.max_input_ch} channels")
                        self.ring_buffer_size = ring_size
                        self.target_latency_samples = int(self.sample_rate * 0.010)
                        self.buffer = np.zeros((self.ring_buffer_size, self.max_input_ch), dtype=np.float32)
                        self.buffer_pool = [np.zeros((8192, self.max_input_ch), dtype=np.float32) for _ in range(self.pool_size)]
                        self._src_blk_pre = np.zeros((8192, self.max_input_ch), dtype=np.float32)
                        self.pool_index = 0
                    self.write_ptr = 0
                    self.read_ptr = 0
                    self.available_samples = 0
                    self._drift_fill_ema = None
                    self._in_primed = False

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
                self.input_stream = None
                print(f"Failed to open input stream: {e}")

    def open_output_stream(self, channels, silent=False):
        with self.stream_lock:
            if channels in self.failed_channel_counts:
                return

            previous_channels = self.current_channels
            try:
                # 1. Close current stream
                if self.output_stream:
                    try:
                        self.output_stream.stop()
                        self.output_stream.close()
                    except Exception:
                        pass
                    self.output_stream = None

                # 2. Try to open new one. current_channels is set before start()
                #    so the first callbacks already see the right layout.
                self.current_channels = channels
                stream = sd.OutputStream(
                    device=self.output_device_id,
                    samplerate=self.sample_rate,
                    blocksize=self.blocksize,
                    dtype='float32',
                    channels=channels,
                    callback=self.output_callback,
                    latency='low',
                    extra_settings=self._wasapi_settings(exclusive=self.exclusive_mode),
                )
                stream.start()
                self.output_stream = stream
                if not silent:
                    mode_str = "(Exclusive)" if self.exclusive_mode else "(Shared)"
                    print(f"Output Stream active: {channels} channels {mode_str}")

            except Exception as e:
                self.current_channels = previous_channels
                self.failed_channel_counts.add(channels)
                # Fall back to Stereo if we were trying Surround
                if channels != 2:
                    print(f"Could not open {channels}-channel output ({e}); falling back to stereo.")
                    self.open_output_stream(2, silent=True)
                elif not silent:
                    print(f"CRITICAL ERROR: Could not open any output stream on device {self.output_device_id}: {e}")

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

        # Dialogue presence band (1.5-4 kHz) for Dialog Focus
        self._dlg_bp_sos = butter(2, [1500.0 / (sr / 2), min(0.95, 4000.0 / (sr / 2))], btype='bandpass', output='sos')

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

        # Secondary bass-shaker output filters
        shaker_hp = min(0.95, max(5.0, getattr(self, 'secondary_output_highpass_hz', 20.0)) / (sr / 2.0))
        shaker_lp = min(0.95, max(getattr(self, 'secondary_output_highpass_hz', 20.0) + 5.0, getattr(self, 'secondary_output_lowpass_hz', 80.0)) / (sr / 2.0))
        if shaker_hp < shaker_lp:
            self._secondary_output_hp_sos = butter(2, shaker_hp, btype='high', output='sos')
            self._secondary_output_lp_sos = butter(2, shaker_lp, btype='low', output='sos')

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

        # Amplitude compensation, physical 1/r (attenuate only: never boost, to
        # prevent clipping). The delay alignment below already fixes
        # localisation; a stronger level cut (the old exponent was 2.5)
        # pushed the image to the opposite side.
        comp_L = min(1.0, (dist_L / current_base_dist))
        comp_R = min(1.0, (dist_R / current_base_dist))
        comp_C = min(1.0, (dist_C / current_base_dist_C))

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
                comp_SL = min(1.0, (dist_SL / R_spk))
                comp_SR = min(1.0, (dist_SR / R_spk))
                _comp_keys += ['_ht_cs_SL','_ht_cs_SR']
                _comp_vals += [comp_SL, comp_SR]
                _delay_keys += ['_ht_ds_SL','_ht_ds_SR']
                _delay_vals += [max(0.0, (max_dist - dist_SL) / cm_per_sample),
                                max(0.0, (max_dist - dist_SR) / cm_per_sample)]
            if out_ch > 6:
                comp_BL = min(1.0, (dist_BL / R_spk))
                comp_BR = min(1.0, (dist_BR / R_spk))
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
                setattr(self, k, getattr(self, k, v) * 0.30 + v * 0.70)
            for k, v in zip(_delay_keys, _delay_vals):
                setattr(self, k, getattr(self, k, v) * 0.95 + v * 0.05)
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

        # Windows order: 7.1 = FL FR C LFE BL BR SL SR, 5.1 surrounds on 4/5
        side_ch = (6, 7) if out_ch >= 8 else (4, 5)
        back_ch = (4, 5)
        if is_gaming and not is_hp:
            if out_ch > 4:
                ramp_SL = self._get_ramp('_last_ht_comp_SL', getattr(self, '_ht_cs_SL', 1.0), frames)
                ramp_SR = self._get_ramp('_last_ht_comp_SR', getattr(self, '_ht_cs_SR', 1.0), frames)
                out_block[:frames, side_ch[0]] *= ramp_SL
                out_block[:frames, side_ch[1]] *= ramp_SR
            if out_ch > 6:
                ramp_BL = self._get_ramp('_last_ht_comp_BL', getattr(self, '_ht_cs_BL', 1.0), frames)
                ramp_BR = self._get_ramp('_last_ht_comp_BR', getattr(self, '_ht_cs_BR', 1.0), frames)
                out_block[:frames, back_ch[0]] *= ramp_BL
                out_block[:frames, back_ch[1]] *= ramp_BR
                
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

        # (No distance-dependent stage width: upmixed source positions must not
        # move when the listener does.)
        S_WIDTH = getattr(self, 'speaker_width_cm', 60.0)
        u_z = max(10.0, float(user_z)) # Avoid divide by zero

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

            # (No air-absorption treble boost: at 1-2 m air absorption is
            # ~0.1 dB/m even at 10 kHz, so a distance-dependent boost only
            # changed the tonal balance.)

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
                    _delay_ch_map[side_ch[0]] = '_ht_ds_SL'
                    _delay_ch_map[side_ch[1]] = '_ht_ds_SR'
                if out_ch > 6:
                    _delay_ch_map[back_ch[0]] = '_ht_ds_BL'
                    _delay_ch_map[back_ch[1]] = '_ht_ds_BR'
            any_delay = any(getattr(self, v, 0.0) > 0.01 for ch, v in _delay_ch_map.items() if ch < out_ch)

            if any_delay:
                if not hasattr(self, '_ht_delay_buf') or self._ht_delay_buf.shape[0] < frames + 2048 or self._ht_delay_buf.shape[1] != out_ch:
                    self._ht_delay_buf = np.zeros((max(8192, frames + 2048), out_ch), dtype=np.float32)
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

        current_time = time.time()
        if getattr(self, 'debug', False) and (not hasattr(self, '_last_ht_log_time') or (current_time - getattr(self, '_last_ht_log_time', 0)) >= 1.0):
             self._last_ht_log_time = current_time
             yaw_deg = np.degrees(yaw)
             print(f"[{current_time:.2f}][HT-PAN] X:{user_x:+.1f}cm Z:{user_z:.1f}cm | Yaw: {yaw_deg:+.1f}° | Comp L:{comp_L:.2f}x R:{comp_R:.2f}x")

    # ─── #19 Declipping ───────────────────────────────────────────────────
    def _apply_declip(self, block):
        """Detect hard-clipped runs and reconstruct them with a cubic spline.

        Peaks may end up above 0 dBFS; the output limiter brings them back down
        smoothly (no per-block renormalisation, which caused gain jumps).
        """
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
            try:
                restored = CubicSpline(indices[good], sig[good].astype(np.float64))(indices[clip_mask])
            except ValueError:
                continue
            # Keep the clipped sign and never go below the clip level; cap overshoot.
            clipped = sig[clip_mask]
            restored = np.sign(clipped) * np.clip(np.abs(restored), np.abs(clipped), 2.0)
            sig[clip_mask] = restored

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

        # Serial loop at decimated rate, on plain Python floats per channel
        # (same IEEE double arithmetic as the former numpy version, so the
        # result is bit-identical, without per-sample numpy call overhead)
        dec_env = np.empty_like(dec_in)
        if n_dec == 0:
            return np.empty((0, n_ch), dtype=abs_block.dtype), zi.copy()
        # coefficients may be scalars or one per channel (limiter groups)
        att_v = np.broadcast_to(np.asarray(att_d, dtype=np.float64).reshape(-1), (n_ch,))
        rel_v = np.broadcast_to(np.asarray(rel_d, dtype=np.float64).reshape(-1), (n_ch,))
        prev_out = np.empty((1, n_ch), dtype=np.result_type(zi.dtype, np.float64))
        for ch in range(n_ch):
            att_d = float(att_v[ch])
            rel_d = float(rel_v[ch])
            att_c = 1.0 - att_d
            rel_c = 1.0 - rel_d
            p = float(zi[0, ch])
            col = dec_in[:, ch].tolist()
            out_col = [0.0] * n_dec
            for i, v in enumerate(col):
                if v > p:
                    p = att_d * p + att_c * v
                else:
                    p = rel_d * p + rel_c * v
                out_col[i] = p
            dec_env[:, ch] = out_col
            prev_out[0, ch] = p
        prev = prev_out

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
        n_samp = block.shape[0]

        # Use reference block for envelope detection if available. Detection
        # is linked across channels (peak of all of them) and one gain is
        # applied to every channel: per-channel gains moved sounds between
        # the speakers on every hit (the center and the fronts boosted at
        # different moments).
        detect = ref_block if ref_block is not None else block
        abs_block = np.max(np.abs(detect), axis=1, keepdims=True).astype(np.float64)

        if not hasattr(self, '_tr_fast_zi') or self._tr_fast_zi.shape[1] != 1:
            self._tr_fast_zi = np.zeros((1, 1), dtype=np.float64)
            self._tr_slow_zi = np.zeros((1, 1), dtype=np.float64)
            self._tr_gain_zi = np.ones((1, 1), dtype=np.float64)

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
        transient_amount = self._live_param('transient_amount')
        max_boost = 10.0 ** (transient_amount / 20.0)
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
        # Stereo level of the program: mean power of L/R; with a discrete
        # multichannel input, the power of all main channels scaled to a stereo
        # equivalent (LFE excluded), so 5.1 and stereo read alike.
        mains = [c for c in range(min(block.shape[1], 8)) if c != 3]
        power = float(np.sum(np.mean(np.asarray(block[:, mains], dtype=np.float64) ** 2, axis=0))) / 2.0
        rms = np.sqrt(power + 1e-12)
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
        n_samp = block.shape[0]
        n_ch = 1   # linked: one level and one gain for all channels (per-channel
                   # gains pumped the quieter surrounds against the fronts)
        if n_samp < 64:
            return

        max_boost_db = self._live_param('micro_dyn_amount')

        # ── Lazy-init state ──
        if not hasattr(self, '_ud_ref_db') or len(self._ud_ref_db) != n_ch:
            self._ud_ref_db = np.full(n_ch, -100.0, dtype=np.float64)
            self._ud_prev_gain_db = np.zeros(n_ch, dtype=np.float64)
            self._ud_initialized = False

        # ── Per-channel block RMS in dB ──
        rms = np.sqrt(np.atleast_1d(np.mean(block.astype(np.float64) ** 2) * block.shape[1]) + 1e-12)
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
    def _apply_surround_decorrelation(self, sig_L, sig_R, bank='classic', stages=6):
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
            # The right channel needs a DIFFERENT set of stages: cascaded
            # all-pass filters commute, so merely reversing the order of the
            # same (delay, coeff) pairs produced an identical filter and no
            # decorrelation at all.
            self._decorr_delays_R = [int(sr * d) for d in
                                     [0.00037, 0.00083, 0.00127, 0.00149, 0.00211, 0.00263]]
            self._decorr_coeffs_R = [0.57, 0.63, 0.50, 0.61, 0.56, 0.64]
            # Per-bank, per-channel state: each stage has a delay buffer
            self._decorr_ap_states = {}
        if bank not in self._decorr_ap_states:
            self._decorr_ap_states[bank] = {}
            for ch_tag, delays in [('L', self._decorr_delays), ('R', self._decorr_delays_R)]:
                self._decorr_ap_states[bank][ch_tag] = [{} for _ in delays]
        states = self._decorr_ap_states[bank]

        def _process_channel(sig, ch_tag):
            # Each stage: y[n] = -g*x[n] + x[n-D] + g*y[n-D]  (Schroeder all-pass).
            # The feedback only links samples D apart, so the recursion is D
            # interleaved first-order filters: reshape to rows of D samples and
            # run one first-order lfilter down the rows (O(n), no Python loop).
            out = np.asarray(sig, dtype=np.float64)
            n = out.shape[0]
            delays = self._decorr_delays if ch_tag == 'L' else self._decorr_delays_R
            coeffs = self._decorr_coeffs if ch_tag == 'L' else self._decorr_coeffs_R
            for i, (delay, coeff) in enumerate(list(zip(delays, coeffs))[:stages]):
                state = states[ch_tag][i]
                if 'x_hist' not in state:
                    state['x_hist'] = np.zeros(delay)
                    state['y_hist'] = np.zeros(delay)
                x_ext = np.concatenate((state['x_hist'], out))
                u = -coeff * out + x_ext[:n]
                rows = -(-n // delay)
                u_rows = np.zeros(rows * delay)
                u_rows[:n] = u
                y = lfilter([1.0], [1.0, -coeff], u_rows.reshape(rows, delay), axis=0,
                            zi=(coeff * state['y_hist'])[None, :])[0].ravel()[:n]
                state['x_hist'] = x_ext[-delay:].copy()
                state['y_hist'] = np.concatenate((state['y_hist'], y))[-delay:].copy()
                out = y
            return out.astype(np.float32)

        # Use different delay orderings for L vs R to maximize decorrelation
        out_L = _process_channel(sig_L, 'L')
        out_R = _process_channel(sig_R, 'R')
        return out_L, out_R

    # ─── #20 Frequency-Dependent Stereo Width ─────────────────────────────
    def _apply_freq_dep_width(self, block):
        """Mono bass (<200 Hz), untouched mids, widened highs (>8 kHz).

        Works on Mid/Side: the side signal is high-passed at 200 Hz (so bass
        becomes mono: about -21 dB of side at 60 Hz) and 40 % of its >8 kHz
        content is added back on top. Mono content passes bit-exact.
        """
        if block.shape[1] < 2:
            return
        sr = self.sample_rate

        if not hasattr(self, '_fdw_lo_sos'):
            self._fdw_lo_sos = butter(2, 200.0 / (sr / 2), btype='high', output='sos')
            self._fdw_hi_sos = butter(2, min(0.95, 8000.0 / (sr / 2)), btype='high', output='sos')
            self._fdw_zi = {}

        L = block[:, 0].astype(np.float64)
        R = block[:, 1].astype(np.float64)
        mid = 0.5 * (L + R)
        side = 0.5 * (L - R)

        zi = self._ensure_sos_zi(self._fdw_zi, 'side_lo', self._fdw_lo_sos)
        side_no_bass, self._fdw_zi['side_lo'] = sosfilt(self._fdw_lo_sos, side, zi=zi)
        zi = self._ensure_sos_zi(self._fdw_zi, 'side_hi', self._fdw_hi_sos)
        side_air, self._fdw_zi['side_hi'] = sosfilt(self._fdw_hi_sos, side, zi=zi)
        side_out = side_no_bass + 0.4 * side_air

        block[:, 0] = mid + side_out
        block[:, 1] = mid - side_out

    # ─── #21 Headphone Externalization ────────────────────────────────────
    def _apply_hp_externalization(self, block):
        """Crossfeed + early reflections + pinna notch for out-of-head imaging."""
        if block.shape[1] < 2:
            return
        sr = self.sample_rate
        frames = block.shape[0]

        if hasattr(self, '_ext_delay_buf') and self._ext_delay_buf.shape[0] < int(sr * 0.05) + frames:
            self._ext_delay_buf = np.zeros((int(sr * 0.05) + frames, 2), dtype=np.float32)
            self._ext_delay_ptr = 0

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
            buf_sz = int(sr * 0.05) + frames
            self._ext_delay_buf = np.zeros((buf_sz, 2), dtype=np.float32)
            self._ext_delay_ptr = 0
            self._ext_reflections = [
                (int(sr * 0.006), 0.15),
                (int(sr * 0.013), 0.10),
                (int(sr * 0.021), 0.06),
            ]

        L = block[:, 0].copy()
        R = block[:, 1].copy()

        # 1. Crossfeed: ADD the opposite channel through a head-shadow LP
        #    (-10.5 dB), with a direct-path trim so mono bass stays ~unity.
        xf_L, self._ext_xfeed_zi_L = sosfilt(self._ext_xfeed_sos, R, zi=self._ext_xfeed_zi_L)
        xf_R, self._ext_xfeed_zi_R = sosfilt(self._ext_xfeed_sos, L, zi=self._ext_xfeed_zi_R)
        block[:, 0] = L * 0.82 + xf_L * 0.30
        block[:, 1] = R * 0.82 + xf_R * 0.30

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
        if not self._ensure_sos_filters('_clarity_mud_sos', '_clarity_peak_sos'):
            return
        
        # 1. Detect mud activity (L/R only for efficiency)
        detect_ch = min(n_ch, 2)
        
        if not hasattr(self, '_clarity_mud_zi'):
            self._clarity_mud_zi = {}
            
        # Detect energy in 200-500Hz band
        mud_sig = np.zeros((frames, detect_ch), dtype=np.float32)
        for ch in range(detect_ch):
            zi = self._ensure_sos_zi(self._clarity_mud_zi, ch, self._clarity_mud_sos)
            mud_sig[:, ch], self._clarity_mud_zi[ch] = sosfilt(self._clarity_mud_sos, block[:, ch], zi=zi)
            
        mud_rms = np.sqrt(np.mean(mud_sig**2) + 1e-12)
        bb_rms = np.sqrt(np.mean(block[:, :detect_ch]**2) + 1e-12)
        
        # Ratio of mud to broadband
        ratio = mud_rms / bb_rms
        
        # Threshold: if mud is more than 40% of broadband energy (log scale roughly)
        threshold = 0.40
        sensitivity = 2.0 # Response aggressiveness
        
        clarity_amount = self._live_param('clarity_amount')
        max_blend = np.clip(clarity_amount / 12.0, 0.08, 0.55)
        target_attenuation = 0.0
        if ratio > threshold:
            target_attenuation = np.clip((ratio - threshold) * sensitivity, 0.0, max_blend)
            
        # Smooth the gain changes (tau = 80ms)
        alpha = 1.0 - np.exp(-frames / (sr * 0.08))
        self._clarity_gain_smooth = self._clarity_gain_smooth * (1.0 - alpha) + target_attenuation * alpha
        
        if self._clarity_gain_smooth > 0.001:
            for ch in range(n_ch):
                zi = self._ensure_sos_zi(self._clarity_zi, ch, self._clarity_peak_sos)
                notched_sig, self._clarity_zi[ch] = sosfilt(self._clarity_peak_sos, block[:, ch], zi=zi)
                
                # Dynamic blend: out = sig * (1 - blend) + notched * blend
                block[:, ch] = block[:, ch] * (1.0 - self._clarity_gain_smooth) + notched_sig * self._clarity_gain_smooth

    def _apply_virtual_room(self, block):
        """Virtual Room: early reflections (4-18 ms) for spaciousness.

        Real early reflections reach the listener from the side and rear walls,
        not from the speaker that plays the direct sound. On surround layouts
        the reflections of the front stage are therefore played from the
        sides (early, 4.3 / 7.1 ms) and backs (later, 11.4 / 17.8 ms); nothing
        is added to the fronts, the center or the LFE (same-speaker echoes
        only colour the sound). On stereo outputs they go into L/R.
        """
        if not getattr(self, 'virtual_room', False):
            return
        if not self._ensure_sos_filters('_room_lp_sos'):
            return

        sr = self.sample_rate
        frames = block.shape[0]
        n_ch = block.shape[1]
        mix = self._live_param('virtual_room_mix')

        # Source: the front stage (center folded in at -6 dB)
        src = block[:, :2].astype(np.float32).copy()
        if n_ch > 2:
            src += 0.5 * block[:, 2:3]

        buf_sz = int(sr * 0.05) + frames
        if self._room_delay_buf is None or self._room_delay_buf.shape[0] < buf_sz or self._room_delay_buf.shape[1] != 2:
            self._room_delay_buf = np.zeros((buf_sz, 2), dtype=np.float32)
            self._room_delay_ptr = 0
        buf_len = self._room_delay_buf.shape[0]
        w_idx = (self._room_delay_ptr + np.arange(frames)) % buf_len
        self._room_delay_buf[w_idx, :] = src
        self._room_delay_ptr = (self._room_delay_ptr + frames) % buf_len

        def tap(ms, swap):
            r_idx = (w_idx - int(ms * sr / 1000.0)) % buf_len
            refl = self._room_delay_buf[r_idx, :]
            return refl[:, ::-1] if swap else refl

        if n_ch >= 6:
            side = 0.45 * tap(4.3, False) + 0.35 * tap(7.1, True)
            back = 0.25 * tap(11.4, False) + 0.15 * tap(17.8, True)
            if n_ch >= 8:
                targets = {(6, 7): side, (4, 5): back}
            else:
                targets = {(4, 5): side + back}
        else:
            front = (0.45 * tap(4.3, False) + 0.35 * tap(7.1, True)
                     + 0.25 * tap(11.4, False) + 0.15 * tap(17.8, True))
            targets = {(0, 1): front}

        # Damping (wall absorption) and mix
        for (cl, cr), refl in targets.items():
            for ch, sig in ((cl, refl[:, 0]), (cr, refl[:, 1])):
                zi = self._ensure_sos_zi(self._room_lp_zi, ch, self._room_lp_sos)
                damped, self._room_lp_zi[ch] = sosfilt(self._room_lp_sos, sig, zi=zi)
                block[:, ch] += (0.5 * mix) * damped

    def _apply_air_exciter(self, block):
        """Harmonic Air Exciter: Restores/Enhances 10kHz+ detail using non-linear saturation."""
        if not getattr(self, 'air_exciter', False):
            return
        if not self._ensure_sos_filters('_air_hp_sos'):
            return
            
        n_ch = block.shape[1]
        n_sec = np.asarray(self._air_hp_sos).shape[0]
        # 1. High-pass filter (all channels in one call: same per-channel
        #    arithmetic as one call per channel)
        zi = self._air_hp_zi.get('all')
        if not isinstance(zi, np.ndarray) or zi.shape != (n_sec, 2, n_ch):
            zi = np.zeros((n_sec, 2, n_ch))
        hp, self._air_hp_zi['all'] = sosfilt(self._air_hp_sos, block, axis=0, zi=zi)
        excited = hp.astype(block.dtype)

        # 2. Harmonics: smooth, bounded odd-order term (no discontinuity)
        harmonics = 0.25 * np.tanh(2.0 * excited) ** 3

        # 2b. High-pass the harmonics again so products aliased back below
        #     ~10 kHz are removed; the linear air band keeps a single HP so its
        #     phase still adds constructively to the dry signal.
        if not hasattr(self, '_air_post_zi'):
            self._air_post_zi = {}
        zi = self._air_post_zi.get('all')
        if not isinstance(zi, np.ndarray) or zi.shape != (n_sec, 2, n_ch):
            zi = np.zeros((n_sec, 2, n_ch))
        post, self._air_post_zi['all'] = sosfilt(self._air_hp_sos, harmonics, axis=0, zi=zi)
        excited += post.astype(block.dtype)

        # 3. Mix
        amount = self._live_param('air_amount')
        block += excited * amount

    def _apply_post_effects(self, out_block, out_ch, skip_subsynth=False, src_block=None, is_passthrough=False):
        """Apply post-upmix effects pipeline to the output block."""
        frames = out_block.shape[0]

        # ── Program Level Estimator (always runs, measures only) ──
        # Measured on the input: after the upmix, L/R alone miss the center
        # (several dB low), which made level-adaptive effects (Dynamic EQ,
        # transient gate...) think the music was quiet, and read differently
        # in passthrough and upmix.
        self._update_program_level(src_block if src_block is not None else out_block)

        # (#19 Declipping runs on the input, see _apply_upmix: after the upmix a
        # center above 0.99 is legitimate, not clipping)

        # #17 Loudness Normalization (AGC)
        if getattr(self, 'loudness_norm', False):
            self._apply_loudness_norm(out_block)

        # #14 Fletcher-Munson Loudness Compensation (Dynamic EQ toggle)
        if getattr(self, 'dynamic_eq', False):
            self._apply_loudness_comp(out_block, self._program_level_db)
        elif self._loudness_filter_zi:
            self._loudness_filter_zi = {}

        # #24 Dynamic Clarity (De-Mudding)
        if getattr(self, 'clarity_dsp', False):
            self._apply_clarity_dsp(out_block)

        # #25 Virtual Room (Early Reflections)
        if getattr(self, 'virtual_room', False):
            self._apply_virtual_room(out_block)

        # #26 Harmonic Air Exciter
        if getattr(self, 'air_exciter', False):
            self._apply_air_exciter(out_block)

        # #16 Dialogue presence boost on the center channel (skipped when vocal_lock
        #      is active since vocal_lock already applies its own adaptive boost).
        #      Always called on surround layouts so the boost ramps in/out.
        if out_ch > 2:
            focus_on = getattr(self, 'dialogue_focus', False) and not getattr(self, 'vocal_lock', False)
            self._apply_dialogue_focus(out_block, focus_on)

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
        # Tilt: its linear-phase FIR delays by 127 samples, so switching it
        # crossfades over one block (an instant switch jumped in time: click).
        # While off, its input history keeps running so it resumes cleanly.
        tilt_on = bool(self.spectral_tilt)
        tilt_was = getattr(self, '_tilt_was_on', False)
        if tilt_on or tilt_was:
            dry = out_block[:frames].copy() if tilt_on != tilt_was else None
            self._apply_spectral_tilt(out_block, frames, out_ch)
            if dry is not None:
                w = np.linspace(0.0, 1.0, frames, dtype=np.float32)[:, None]
                if not tilt_on:
                    w = 1.0 - w
                out_block[:frames] = dry * (1.0 - w) + out_block[:frames] * w
        else:
            self._tilt_keep_history(out_block, frames, out_ch)
        self._tilt_was_on = tilt_on

        # #23 Micro-Dynamics / Upward Compression (lifts quiet detail)
        # Snapshot before micro-dyn so transient restore can measure original dynamics
        micro_dyn_active = getattr(self, 'micro_dynamics', False)
        transient_active = getattr(self, 'transient_restore', False)
        pre_micro_dyn = out_block.copy() if (micro_dyn_active and transient_active) else None
        if micro_dyn_active:
            self._apply_micro_dynamics(out_block)

        # #18 Transient Restoration (late — enhances final dynamics)
        # When micro-dyn is active, measure envelopes from pre-micro-dyn signal
        # so transient detection isn't fooled by the upward compression
        if transient_active:
            self._apply_transient_restore(out_block, ref_block=pre_micro_dyn)

        # #12 Transaural Widening (crosstalk cancellation of the L/R pair).
        # Only without a center channel: with one, a source panned between L
        # and C would be cancelled on half of its image only, smearing it. The
        # front stage width then comes from the real speakers instead. Faded
        # in/out so switching never clicks.
        xtc_target = 1.0 if (getattr(self, 'transaural_widening', False)
                             and not getattr(self, '_center_in_use', False)) else 0.0
        if xtc_target > 0.0 or getattr(self, '_last_xtc_mix', 0.0) > 0.0:
            if not hasattr(self, '_last_xtc_mix'):
                self._last_xtc_mix = 0.0
            dry = out_block[:, :2].copy()
            self._apply_transaural_widener(out_block, frames)
            mix = self._get_ramp('_last_xtc_mix', xtc_target, frames)[:, np.newaxis]
            out_block[:, :2] = dry * (1.0 - mix) + out_block[:, :2] * mix

        # #21 Headphone Externalization (after transaural, for HP mode)
        if getattr(self, 'hp_externalization', False):
            self._apply_hp_externalization(out_block)

        # #1 Independent Head Tracking Volume Panning (MUST RUN LAST for Absolute Room Alignment)
        self._apply_head_tracking_panning(out_block, frames, out_ch, is_passthrough=is_passthrough)

    def _apply_transaural_widener(self, out_block, frames):
        """Recursive crosstalk cancellation (XTC) with optional head-tracked geometry.

        Per ear:  out_L = L - ff_L * D_L[LP(R)] - fb * D_L[out_R]
                  out_R = R - ff_R * D_R[LP(L)] - fb * D_R[out_L]
        where D_x is a fractional delay. Substituting one equation into the
        other turns the cross-feedback into a comb recursion on each output
        alone, which scipy's lfilter runs in C (the previous per-sample Python
        loop cost >10% of a core). Outputs of previous blocks enter through an
        explicit history, so the result stays exact while delays change with
        head tracking.
        """
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

        hist_len = 512
        if d_L + d_R + 3 > hist_len:
            return
        if not hasattr(self, '_trans_hist'):
            self._trans_hist = {k: np.zeros(hist_len, dtype=np.float64)
                                for k in ('L_lp', 'R_lp', 'out_L', 'out_R')}
        h = self._trans_hist

        L = out_block[:N, 0].astype(np.float64)
        R = out_block[:N, 1].astype(np.float64)
        if not hasattr(self, '_trans_lp_zi_L'):
            self._trans_lp_zi_L = sosfilt_zi(self._trans_lp_sos) * 0.0
            self._trans_lp_zi_R = sosfilt_zi(self._trans_lp_sos) * 0.0

        L_lp, self._trans_lp_zi_L = sosfilt(self._trans_lp_sos, L, zi=self._trans_lp_zi_L)
        R_lp, self._trans_lp_zi_R = sosfilt(self._trans_lp_sos, R, zi=self._trans_lp_zi_R)

        # Feed-forward cancellation terms
        u_L = L - gain_ff_L * _frac_delay(R_lp, h['R_lp'], d_L, frac_L)
        u_R = R - gain_ff_R * _frac_delay(L_lp, h['L_lp'], d_R, frac_R)

        # Feedback: out_L = u_L - fb*D_L[out_R],  out_R = u_R - fb*D_R[out_L].
        # Contributions of outputs from previous blocks are taken from the real
        # history (exact even when the delays just changed); the in-block part
        # of the cross-feedback becomes a comb recursion with kernel D_L*D_R.
        silent = np.zeros(N)
        w_R = u_R - gain_fb * _frac_delay(silent, h['out_L'], d_R, frac_R)
        w_L = u_L - gain_fb * _frac_delay(silent, h['out_R'], d_L, frac_L)
        in_L = u_L - gain_fb * _frac_delay(w_R, h['out_R'], d_L, frac_L)
        in_R = u_R - gain_fb * _frac_delay(w_L, h['out_L'], d_R, frac_R)

        D = d_L + d_R
        k0 = (1.0 - frac_L) * (1.0 - frac_R)
        k1 = (1.0 - frac_L) * frac_R + frac_L * (1.0 - frac_R)
        k2 = frac_L * frac_R
        fb2 = gain_fb * gain_fb
        a = np.zeros(D + 3)
        a[0] = 1.0
        a[D] = -fb2 * k0
        a[D + 1] = -fb2 * k1
        a[D + 2] = -fb2 * k2
        out_L = lfilter([1.0], a, in_L)
        out_R = lfilter([1.0], a, in_R)

        h['L_lp'] = _push_history(h['L_lp'], L_lp)
        h['R_lp'] = _push_history(h['R_lp'], R_lp)
        h['out_L'] = _push_history(h['out_L'], out_L)
        h['out_R'] = _push_history(h['out_R'], out_R)

        out_block[:N, 0] = out_L
        out_block[:N, 1] = out_R

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



    def _tilt_keep_history(self, out_block, frames, out_ch):
        hist_len = 254
        hist = getattr(self, '_tilt_hist', None)
        if hist is None or hist.shape != (hist_len, out_ch):
            hist = self._tilt_hist = np.zeros((hist_len, out_ch), dtype=np.float32)
        if frames >= hist_len:
            hist[:] = out_block[frames - hist_len:frames, :out_ch]
        else:
            hist[:-frames] = hist[frames:]
            hist[-frames:] = out_block[:frames, :out_ch]

    TILT_FRAME = 480   # analysis frame (samples): fixed, whatever the callback block size

    def _apply_spectral_tilt(self, out_block, frames, out_ch):
        """Adaptive Equalization with Overlap-and-Add Carry Buffers.

        Block-size independent: the spectrum is analysed on a fixed frame of
        the latest TILT_FRAME samples, smoothing runs in time (not per block),
        and the FIR always runs (its 127-sample delay never comes and goes;
        variable WASAPI block sizes used to make it click).
        """
        N = frames
        if N < 1:
            return
        A = self.TILT_FRAME
        k_blocks = N / float(A)           # how many reference blocks this one is worth

        # 1. Mono contour of the latest A samples (FL, FR, C, SL, SR; no LFE)
        chans = [ch for ch in (0, 1, 2, 4, 5) if ch < out_ch]
        mono_now = np.zeros(N, dtype=np.float32)
        for ch in chans:
            mono_now += out_block[:N, ch]
        if chans:
            mono_now /= len(chans)
        hist = getattr(self, '_tilt_mono_hist', None)
        if hist is None or hist.shape != (A,):
            hist = self._tilt_mono_hist = np.zeros(A, dtype=np.float32)
        if N >= A:
            hist[:] = mono_now[N - A:]
        else:
            hist[:-N] = hist[N:]
            hist[-N:] = mono_now
        mono = hist

        padded_mono = getattr(self, '_tilt_padded_pre', None)
        if padded_mono is None or padded_mono.shape != (2 * A,):
            padded_mono = self._tilt_padded_pre = np.zeros(2 * A, dtype=np.float32)
        padded_mono.fill(0)
        padded_mono[:A] = mono

        fft_mono = np.fft.rfft(padded_mono)
        mag = np.abs(fft_mono)
        silent = np.max(mag) < 1e-6

        if not silent:
            self._tilt_update_gain(mag, k_blocks)
        num_taps = 255
        if self._tilt_taps is None:
            # nothing analysed yet: a pure delay keeps the timing consistent
            self._tilt_taps = np.zeros(num_taps)
            self._tilt_taps[num_taps // 2] = 1.0
        taps = self._tilt_taps
        self._tilt_filter_block(out_block, N, out_ch, taps, k_blocks)

    def _tilt_update_gain(self, mag, k_blocks):
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
        
        # Gentle treble roll-off above 3kHz (-0.5dB per octave)
        treble_mask = freqs > 3000
        octaves = np.log2(freqs[treble_mask] / 3000.0)
        target[treble_mask] = 10 ** (-0.5 * octaves / 20.0)
        
        # Regularized Inversion (Soft Waterfilling) to avoid boosting noise floor
        reg = 0.02 * np.max(smooth_mag)  # 2% of max magnitude peak
        gain_raw = target / (smooth_mag + reg)

        
        # Normalize to prevent scale shifts
        gain_raw /= (np.mean(gain_raw) + 1e-6)
        
        if not hasattr(self, '_tilt_filter_zi') or not isinstance(self._tilt_filter_zi, dict):
             self._tilt_filter_zi = {}
             
        # Smoothed gain curve; reset if the block size (and so the bin count) changed
        prev_gain = self._tilt_filter_zi.get('_tilt_gain_smooth')
        if prev_gain is None or prev_gain.shape != gain_raw.shape:
            self._tilt_filter_zi['_tilt_gain_smooth'] = gain_raw.copy()
        else:
            a_s = 0.995 ** k_blocks
            self._tilt_filter_zi['_tilt_gain_smooth'] = prev_gain * a_s + gain_raw * (1.0 - a_s)

        # Prevent treble harshness by tapered maximum boost clamp
        max_gain = np.ones_like(freqs) * 2.0  # Max +6dB absolute boost
        treble_boost_mask = freqs > 1500
        # Taper allowed boost from 1500Hz to 2500Hz down to 1.0 (0dB boost)
        max_gain[treble_boost_mask] = np.interp(freqs[treble_boost_mask], [1500, 2500], [2.0, 1.0])
        max_gain[freqs > 2500] = 1.0 # Strictly prevent ANY boost above 2.5kHz to eliminate harshness

        gain = np.clip(self._tilt_filter_zi['_tilt_gain_smooth'], 0.3, max_gain)

        # ─── Design FIR taps only every 32 reference blocks to save CPU ───
        num_taps = 255
        period = 32 * self.TILT_FRAME
        if self._tilt_taps is None or getattr(self, '_tilt_taps_designed', False) is False or self._tilt_update_cnt >= period:
            norm_freqs = np.linspace(0, 1.0, len(gain))
            self._tilt_taps = firwin2(num_taps, norm_freqs, gain)
            self._tilt_taps_designed = True
            self._tilt_update_cnt = 0 if self._tilt_update_cnt < period else self._tilt_update_cnt - period
        self._tilt_update_cnt += k_blocks * self.TILT_FRAME

    def _tilt_filter_block(self, out_block, N, out_ch, taps, k_blocks):
        num_taps = taps.shape[0]
        # 4. Filter ALL channels (the linear-phase FIR delays everything by
        #    127 samples, so every channel must pass through it to stay aligned).
        #    Convolution runs on explicit input history, so swapping taps never
        #    corrupts the filter state.
        hist_len = num_taps - 1
        if not hasattr(self, '_tilt_hist') or self._tilt_hist.shape != (hist_len, out_ch):
            self._tilt_hist = np.zeros((hist_len, out_ch), dtype=np.float32)

        filtered_all = np.empty((N, out_ch), dtype=np.float32)
        energy_in = energy_out = 1e-12
        for ch in range(out_ch):
            sig = out_block[:N, ch].copy()
            ext = np.concatenate((self._tilt_hist[:, ch], sig))
            self._tilt_hist[:, ch] = ext[-hist_len:]
            filtered_all[:, ch] = np.convolve(ext, taps, mode='valid')[:N]
            if ch != 3:
                energy_in += float(np.sum(sig ** 2))
                energy_out += float(np.sum(filtered_all[:, ch].astype(np.float64) ** 2))

        # Slow level lock: compensates the EQ's overall gain without
        # block-rate pumping (heavily smoothed, limited to +/-6 dB). One scale
        # for all channels: per-channel scales re-balanced the speakers
        # against each other (sounds drifting between center and fronts).
        inst_scale = float(np.clip(np.sqrt(energy_in / energy_out), 0.5, 2.0))
        prev_scale = self._tilt_filter_zi.get('tilt_prev_scale', inst_scale)
        a_l = 0.97 ** k_blocks
        scale_fact = prev_scale * a_l + inst_scale * (1.0 - a_l) if energy_in > 1e-8 else prev_scale
        ramp = np.linspace(prev_scale, scale_fact, N, dtype=np.float32)
        out_block[:N, :out_ch] = filtered_all * ramp[:, None]
        self._tilt_filter_zi['tilt_prev_scale'] = scale_fact

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

        # 4. Voice Avoidance — asymmetric attack/release so voice onset ducks synth fast
        v_pow = np.mean(vocal_sig**2)
        d_pow = np.mean(drive_sig**2)
        sr = self.sample_rate
        frames = N

        # Vocal energy: fast attack (10ms) so voice is caught immediately, slow release (300ms)
        v_atk  = 1.0 - math.exp(-frames / (sr * 0.010))
        v_rel  = 1.0 - math.exp(-frames / (sr * 0.300))
        v_a    = v_atk if v_pow > self._sub_filter_zi['v_avg'] else v_rel
        self._sub_filter_zi['v_avg'] = self._sub_filter_zi['v_avg'] * (1.0 - v_a) + v_pow * v_a

        # Drive energy: slow attack (100ms), fast release (50ms) so ratio rises quickly
        # when bass disappears and voice takes over
        d_atk  = 1.0 - math.exp(-frames / (sr * 0.100))
        d_rel  = 1.0 - math.exp(-frames / (sr * 0.050))
        d_a    = d_atk if d_pow > self._sub_filter_zi['d_avg'] else d_rel
        self._sub_filter_zi['d_avg'] = self._sub_filter_zi['d_avg'] * (1.0 - d_a) + d_pow * d_a

        vocal_energy = math.sqrt(self._sub_filter_zi['v_avg'])
        drive_energy = math.sqrt(self._sub_filter_zi['d_avg'])

        ratio = vocal_energy / (drive_energy + 1e-6)
        # Flat threshold: vocal filter is always above drive band so ratio is meaningful
        avoid_thresh = 0.85
        avoid_slope  = 1.5
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
        win = getattr(self, '_sub_spec_win', None)
        if win is None or win.shape[0] != len(buf):
            win = self._sub_spec_win = np.hanning(len(buf))
        spec = np.abs(np.fft.rfft(buf * win))
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
        """ISO 226:2003 inspired adaptive loudness compensation (Dynamic EQ toggle).

        `level` is the program level in dBFS (slow tracker): 0 dB of boost at
        -20 dBFS and above, rising to +8 dB of bass/treble shelf at -40 dBFS.
        """
        boost_db = float(np.clip((-20.0 - level) * 0.4, 0.0, 8.0))
        if boost_db < 0.1 and not self._loudness_filter_zi:
            return block

        # Simple Bass/Treble shelving filters
        sr = self.sample_rate
        # Only re-calculate coefficients if the boost changed by >0.25 dB
        if self._last_loudness_db is None or self._loud_b_l is None or abs(self._last_loudness_db - boost_db) > 0.25:
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
            self._exciter_dc_zi = np.zeros(1)


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
        #   Rectification produces a large DC component (0.64×A for full-wave).
        #   A stateful one-pole DC blocker (~5 Hz) removes it continuously;
        #   subtracting a per-block mean instead created block-rate steps.
        if not hasattr(self, '_exciter_dc_zi'):
            self._exciter_dc_zi = np.zeros(1)
        r = math.exp(-2.0 * math.pi * 5.0 / self.sample_rate)
        harmonics, self._exciter_dc_zi = lfilter([1.0, -1.0], [1.0, -r], harmonics, zi=self._exciter_dc_zi)

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
        """Look-ahead peak limiter: final safety stage on all channels.

        The gain is the running minimum of the required gain over a 1.5 ms
        look-ahead window (so it is already down when the peak arrives),
        with instant attack and a 60 ms release. All main channels share one
        gain (no image shift); the LFE channel (index 3) is limited on its
        own, so a sub-channel peak never turns down dialogue and music.
        The LFE gain is held for 25 ms (half a cycle at 20 Hz) so it never
        follows the bass waveform (no distortion on sustained overload), then
        released in 20 ms so following bass is not ducked (no pumping).
        Rebuilds itself when the channel count, block size or sample rate
        changes.
        """
        n, ch = block.shape
        if n == 0:
            return block
        threshold = 0.98
        look_ahead = max(2, 2 * int(0.0015 * self.sample_rate / 2))   # even
        if ch > 3:
            groups = [[c for c in range(ch) if c != 3], [3]]
            hold_ms = [0.0, 25.0]
            release_ms = [60.0, 20.0]
        else:
            groups = [list(range(ch))]
            hold_ms = [0.0]
            release_ms = [60.0]
        n_groups = len(groups)
        holds = [2 * int(h * self.sample_rate / 2000.0) for h in hold_ms]   # even sample counts
        max_hold = max(holds)
        if (self._limiter_buffer is None or self._limiter_buffer.shape != (look_ahead, ch)
                or getattr(self, '_limiter_req_hist', np.zeros((0, 0))).shape != (look_ahead, n_groups)
                or getattr(self, '_limiter_hold_hist', np.zeros((0, 0))).shape != (max_hold, n_groups)):
            self._limiter_buffer = np.zeros((look_ahead, ch), dtype=np.float32)
            self._limiter_req_hist = np.ones((look_ahead, n_groups), dtype=np.float64)
            self._limiter_hold_hist = np.ones((max_hold, n_groups), dtype=np.float64)
            self._limiter_gr_zi = np.zeros((1, n_groups), dtype=np.float64)

        combined = np.concatenate((self._limiter_buffer, block), axis=0)  # (la + n, ch)
        self._limiter_buffer = combined[n:].copy()

        new_abs = np.abs(combined[look_ahead:]).astype(np.float64)
        peak = np.stack([new_abs[:, g].max(axis=1) for g in groups], axis=1)      # (n, groups)
        req_new = np.minimum(1.0, threshold / np.maximum(peak, 1e-9))
        req = np.concatenate((self._limiter_req_hist, req_new))  # aligned with `combined`
        self._limiter_req_hist = req[n:].copy()

        # Output sample i is combined[i]; its gain must cover samples i..i+la
        # (forward min == centred min shifted by la/2, O(n))
        g_min = minimum_filter1d(req, size=look_ahead + 1, axis=0, mode='nearest')[look_ahead // 2:look_ahead // 2 + n]

        # Hold: each output sample also keeps the lowest gain of the last
        # `hold` samples (per group)
        if max_hold:
            g_ext = np.concatenate((self._limiter_hold_hist, g_min))
            self._limiter_hold_hist = g_ext[n:].copy()
            for gi, h in enumerate(holds):
                if h:
                    # trailing min over [i - h, i] == centred min shifted by h/2 (O(n))
                    seg = g_ext[max_hold - h:, gi]
                    g_min[:, gi] = minimum_filter1d(seg, size=h + 1, mode='nearest')[h // 2:h // 2 + n]

        gr = 1.0 - g_min
        if gr.max() <= 0.0 and self._limiter_gr_zi.max() < 1e-7:
            # Nothing to limit and the previous release has finished: unity
            # gain, no envelope work (the normal case)
            self._limiter_gr_zi = np.zeros_like(self._limiter_gr_zi)
            out = combined[:n].copy()
            np.clip(out, -1.0, 1.0, out=out)
            return out
        release = np.exp(-1.0 / (self.sample_rate * np.asarray(release_ms) / 1000.0)).reshape(1, -1)
        gr_env, self._limiter_gr_zi = self._asym_env_decimated(gr, 0.0, release, self._limiter_gr_zi)
        group_gain = 1.0 - np.maximum(gr_env, gr)

        gain = np.empty((n, ch), dtype=np.float32)
        for gi, g in enumerate(groups):
            gain[:, g] = group_gain[:, gi:gi + 1]
        out = combined[:n] * gain
        # Absolute safety (only reached by numerical edge cases)
        np.clip(out, -1.0, 1.0, out=out)
        return out

    def _apply_dialogue_focus(self, block, enabled=True):
        """Speech-presence boost (1.5-4 kHz, up to ~+3.5 dB) on the Center channel.

        The band gain is ramped so toggling (manually or by the adaptive
        renderer) never produces a step.
        """
        if block.shape[1] < 3:
            return block
        target = 0.5 if enabled else 0.0
        if not enabled and getattr(self, '_last_dlg_gain', 0.0) == 0.0:
            self._dlg_bp_zi = None
            return block
        if getattr(self, '_dlg_bp_sos', None) is None:
            self._init_multiband_filters()
        if not hasattr(self, '_last_dlg_gain'):
            self._last_dlg_gain = 0.0
        zi = getattr(self, '_dlg_bp_zi', None)
        if zi is None:
            zi = sosfilt_zi(self._dlg_bp_sos) * 0.0
        band, self._dlg_bp_zi = sosfilt(self._dlg_bp_sos, block[:, 2], zi=zi)
        ramp = self._get_ramp('_last_dlg_gain', target, block.shape[0])
        block[:, 2] += band * ramp
        return block

    def _apply_profile_dict(self, profile):
        """Hydrate current attributes from a profile dictionary structure."""
        apply_profile_to_object(self, profile or {})

    def switch_mode(self, new_mode):
        """Serialize current state to cache and load state from new target mode."""
        if new_mode not in ("speakers", "headphones"): return
        if new_mode == self.current_mode: return

        # 1. Capture Current mode buffer
        curr_config = build_profile_from_object(self)
        if self.current_mode == "speakers":
            self.speakers_config = curr_config
        else:
            self.headphones_config = curr_config

        # 2. Re-hydrate from target buffer
        self.current_mode = new_mode
        target_profile = self.speakers_config if new_mode == "speakers" else self.headphones_config
        self._apply_profile_dict(target_profile)
        self._sync_head_tracker()
        self.save_config()

        # 3. Request GUI re-sync (switch_mode is called from the GUI thread)
        if getattr(self, 'gui_window', None):
            if hasattr(self.gui_window, 'sync_ui_from_backend'):
                self.gui_window.sync_ui_from_backend()
            if hasattr(self.gui_window, 'update_slider_states'):
                self.gui_window.update_slider_states()

    def save_config(self, path=None):
        target = path or self.config_file

        # 1. Update Active state into current cached mode dictionary
        active_config = build_profile_from_object(self)

        if self.current_mode == "speakers":
            self.speakers_config = active_config
        else:
            self.headphones_config = active_config

        adaptive_state = build_adaptive_state_from_object(self)
        secondary_output_state = build_secondary_output_state_from_object(self)

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
            "output_headroom_db": float(getattr(self, "output_headroom_db", 4.0)),
            "secondary_output": secondary_output_state,
            **adaptive_state,
        }
        # Atomic write: a crash mid-save can never leave a truncated config
        tmp = f"{target}.tmp"
        try:
            with self.config_lock:
                with open(tmp, 'w') as f:
                    json.dump(data, f, indent=4)
                os.replace(tmp, target)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def load_config(self, path=None):
        """Load settings. With an explicit `path` (GUI import) the machine-specific
        device selections are kept, since the streams are not reopened."""
        target = path or self.config_file
        is_import = path is not None
        if not os.path.exists(target):
            return
        try:
            with open(target, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            if not is_import:
                # Keep the unreadable file for inspection instead of silently overwriting it
                try:
                    os.replace(target, target + ".corrupt")
                    print(f"Unreadable config moved to {target}.corrupt; using defaults.")
                except Exception:
                    pass
            return
        try:
            self.current_mode = data.get("current_mode", "speakers")
            if self.current_mode not in ("speakers", "headphones"):
                self.current_mode = "speakers"
            self.speakers_config = data.get("speakers", {}) or {}
            self.headphones_config = data.get("headphones", {}) or {}
            if not is_import:
                self.input_name = data.get("input_device", self.input_name)
                self.output_name = data.get("output_device", self.output_name)
                self.webcam_name = data.get("webcam_device", getattr(self, "webcam_name", ""))
                self.webcam_index = data.get("webcam_index", getattr(self, "webcam_index", 0))
            self.bypass_all = data.get("bypass_all", getattr(self, "bypass_all", False))
            try:
                self.output_headroom_db = max(0.0, min(12.0, float(data.get("output_headroom_db", getattr(self, "output_headroom_db", 4.0)))))
            except (TypeError, ValueError):
                self.output_headroom_db = 4.0
            apply_adaptive_state_to_object(self, {k: v for k, v in data.items() if k in build_adaptive_state_from_object(self)})

            secondary_output_state = data.get("secondary_output", {})
            if not secondary_output_state:
                secondary_output_state = {
                    key: data[key]
                    for key in (
                        "secondary_output_enabled",
                        "secondary_output_device",
                        "secondary_output_gain_db",
                        "secondary_output_lowpass_hz",
                        "secondary_output_highpass_hz",
                        "secondary_output_delay_ms",
                        "secondary_output_source_mode",
                        "secondary_output_mono_sum",
                    )
                    if key in data
                }
            if is_import:
                secondary_output_state.pop("secondary_output_device", None)
            apply_secondary_output_state_to_object(self, {**build_secondary_output_state_from_object(self), **secondary_output_state})

            # --- Backward Compatibility ---
            # If loading an OLD flat format config.json, populate the speakers config with it!
            if "upmix_enabled" in data and not self.speakers_config:
                legacy_excluded = {
                    "current_mode", "speakers", "headphones", "input_device", "output_device",
                    "webcam_device", "webcam_index", "bypass_all", "output_headroom_db", "adaptive_enabled", "adaptive_monitor_enabled",
                    "adaptive_observe_only", "adaptive_mode", "adaptive_profile", "adaptive_compare_bypass",
                    "secondary_output", "secondary_output_enabled", "secondary_output_device",
                    "secondary_output_gain_db", "secondary_output_lowpass_hz", "secondary_output_highpass_hz",
                    "secondary_output_delay_ms", "secondary_output_source_mode", "secondary_output_mono_sum",
                }
                self.speakers_config = {k: data[k] for k in data if k not in legacy_excluded}

            # Load active profile state values
            active_profile = self.speakers_config if self.current_mode == "speakers" else self.headphones_config
            self._apply_profile_dict(active_profile)

            print(f"Config loaded from {target}")

            # Start/Stop tracker thread for loaded modes
            self._sync_head_tracker()
            self.refresh_secondary_output_filters()
        except Exception as e:
            print(f"Failed to apply config: {e}")

    def _sync_head_tracker(self):
        """Run the camera tracker only while a tracking mode is enabled."""
        if getattr(self, 'head_tracking', False) or getattr(self, 'headphone_tracking', False):
            self.start_head_tracker()
        else:
            self.stop_head_tracker()

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
        if getattr(self, 'head_tracker', None):
            self.head_tracker.stop()
            self.head_tracker = None
            print("Head Tracker offline.")

    def _reset_runtime_after_format_change(self, old_rate):
        """Rebuild everything that depends on sample rate / channel layout."""
        self.failed_channel_counts.clear()
        self._reset_dsp_state()
        if int(old_rate) != int(self.sample_rate):
            for attr in ('_sub_analysis_buf', '_sub_phase_acc', '_sub_env_state', '_sub_confidence',
                         '_ext_xfeed_zi_L', '_exciter_hp_sos', '_exciter_tame_sos'):
                if attr in self.__dict__:
                    delattr(self, attr)
            self.widening_buffer_size = max(4096, int(self.sample_rate))
            self.widening_buffer = np.zeros(self.widening_buffer_size, dtype=np.float32)
            self.widening_write_ptr = 0
            self.adaptive_engine = AdaptiveFeatureEngine(self.sample_rate)
            self._program_level_smooth = None
            self._rebuild_sub_drive_filter()
        self.refresh_secondary_output_filters()

    def change_devices(self, in_name, out_name, secondary_out_name=None):
        """Switch devices. Raises (with streams restored on the previous devices
        when possible) if the requested devices cannot be found."""
        with self.stream_lock:
            # Resolve first, without fallback, so a typo can't route audio elsewhere
            in_dev = self._find_device(in_name, is_input=True, allow_fallback=False)
            out_dev = self._find_device(out_name, is_input=False, allow_fallback=False)

            self.input_name = in_name
            self.output_name = out_name
            if secondary_out_name is not None:
                self.secondary_output_device = secondary_out_name
            self.save_config()

            # Stop streams securely
            for attr in ('input_stream', 'output_stream'):
                stream = getattr(self, attr)
                if stream:
                    try: stream.stop(); stream.close()
                    except Exception: pass
                    setattr(self, attr, None)
            self.close_secondary_output_stream()
            time.sleep(0.2)

            old_rate = self.sample_rate
            self.input_device_id, self.max_input_ch, self.input_rate = in_dev
            self.output_device_id, self.max_output_ch, self.output_rate = out_dev
            self.sample_rate = int(self.output_rate)
            self._reset_runtime_after_format_change(old_rate)

            # Reboot streams
            self.current_channels = 0
            self.open_input_stream()
            self.open_output_stream(self.max_output_ch)
            self.refresh_secondary_output_stream()

    # ════════════════════════════════════════════════════════════════════
    #  BINAURAL CALIBRATION
    # ════════════════════════════════════════════════════════════════════

    def run_binaural_calibration(self, input_device_id=None, progress_callback=None):
        """
        Runs the binaural measurement by opening a dedicated stream.
        Must be called from outside the audio processing loop (e.g., GUI thread).
        """
        import calibration_helper as ch
        if progress_callback: progress_callback("Stopping background streams...")

        in_id = input_device_id if input_device_id is not None else self.input_device_id

        with self.stream_lock:
            # 1. Close streams temporarily to release the device
            for attr in ('input_stream', 'output_stream'):
                stream = getattr(self, attr)
                if stream:
                    try: stream.stop(); stream.close()
                    except Exception: pass
                    setattr(self, attr, None)
            time.sleep(0.3) # Wait for hardware release

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
                itd, k, peaks = ch.calculate_transaural_params(L_FL, R_FL, L_FR, R_FR, sweep, sample_rate=self.sample_rate)

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
        if status:
            print(f"Input Warning: {status}")
            if getattr(status, 'input_overflow', False):
                self.io_stats['in_overflow'] += 1
        try:
            # Apply gain
            if self.gain != 1.0:
                indata[:frames, :] *= self.gain

            # Clip detection (reported by run() loop)
            if not self.peak_hit and np.max(np.abs(indata[:frames])) >= 0.999:
                self.peak_hit = True

            # Throttled Analysis
            self.callback_count += 1
            if self.callback_count % ANALYSIS_STRIDE == 0:
                buf = self.buffer_pool[self.pool_index]
                f = min(frames, buf.shape[0])
                if buf.shape[1] == indata.shape[1]:
                    np.copyto(buf[:f, :], indata[:f, :])
                    try:
                        self.analysis_queue.put_nowait(buf[:f, :])
                        self.pool_index = (self.pool_index + 1) % self.pool_size
                    except queue.Full:
                        pass

            # Write to Ring Buffer
            with self.buffer_lock:
                size = self.ring_buffer_size
                data = indata[:frames]
                if data.shape[1] != self.buffer.shape[1]:
                    return
                if frames >= size:
                    data = data[-size:]
                    frames = size
                # On overflow drop the OLDEST samples: advance the read pointer
                # too, so (write_ptr - read_ptr) always equals available_samples.
                overflow = max(0, self.available_samples + frames - size)
                if overflow:
                    self.read_ptr = (self.read_ptr + overflow) % size
                    self.available_samples -= overflow

                end1 = min(frames, size - self.write_ptr)
                self.buffer[self.write_ptr : self.write_ptr + end1] = data[:end1]
                if end1 < frames:
                    self.buffer[:frames - end1] = data[end1:frames]

                self.write_ptr = (self.write_ptr + frames) % size
                self.available_samples += frames
        except Exception as e:
            self._report_callback_error("input", e)

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

    ROUTE_FADE_SECONDS = 0.05

    def _apply_upmix(self, src_block, out_ch=None):
        """
        Route the input to the output layout (passthrough or upmix), then run
        the post-effects pipeline and the output limiter.
        src_block: (frames, input channels)
        returns out_block: (frames, out_ch)
        """
        frames = src_block.shape[0]
        if out_ch is None:
            out_ch = self.current_channels

        # Reuse pre-allocated output buffer when possible (avoids malloc in RT)
        _buf = getattr(self, '_upmix_out_buf', None)
        if _buf is None or _buf.shape[0] < frames or _buf.shape[1] != out_ch:
            _buf = np.zeros((max(frames, 8192), out_ch), dtype=np.float32)
            self._upmix_out_buf = _buf
        out_block = _buf[:frames]
        out_block[:] = 0.0

        # 0. Master Bypass Mode: no DSP, but channels the output lacks are
        #    still downmixed rather than dropped.
        if getattr(self, 'bypass_all', False):
            limit = min(src_block.shape[1], out_ch)
            out_block[:, :limit] = src_block[:, :limit]
            self._downmix_extra_channels(src_block, out_block, out_ch)
            self._apply_output_headroom(out_block)
            return out_block

        # Declipping repairs the source, so it runs on the input: only there is
        # a flat run at full scale really clipping.
        if getattr(self, 'declip', False):
            src_block = src_block.copy()
            self._apply_declip(src_block)

        # 1. Passthrough: real multichannel detected, stereo output, or upmix off
        force_passthrough = self.is_multichannel_detected and not self.manual_upmix
        is_passthrough = force_passthrough or out_ch <= 2 or not self.upmix_enabled

        # Spectral upmix engine (optional, A/B against the classic one). While
        # it is selected, passthrough is delayed by the same latency so that
        # switching between the two never jumps in time.
        spectral = self._spectral_selected(out_ch)
        if is_passthrough:
            mode = 'pass_d' if spectral else 'pass'
        else:
            mode = 'spectral' if spectral else 'upmix'

        if is_passthrough and self.is_multichannel_detected:
            if self.last_mode != "PASSTHROUGH_DISCRETE":
                print("\n[MODE] PASSTHROUGH (Full Discrete Surround)")
                self.last_mode = "PASSTHROUGH_DISCRETE"
        elif is_passthrough:
            self.last_mode = "PASSTHROUGH_STEREO"
        elif self.last_mode != mode:
            engine = "Spectral" if mode == 'spectral' else "Classic"
            print(f"\n[MODE] UPMIXING ({engine} - {out_ch} ch)")
            self.last_mode = mode

        # 2. Mode changes are crossfaded so switching never clicks
        fade_len = max(1, int(self.ROUTE_FADE_SECONDS * self.sample_rate))
        prev_mode = getattr(self, '_route_mode', None) or mode
        if mode != prev_mode:
            if getattr(self, '_route_fade_from', None) == mode:
                # Flipped back mid-fade: reverse the fade from where it is
                self._route_fade_pos = fade_len - self._route_fade_pos
            else:
                self._route_fade_pos = 0
            self._route_fade_from = prev_mode
        self._route_mode = mode

        # The spectral engine runs once per block whenever the current route,
        # or the one being faded out, needs it.
        spectral_modes = ('spectral', 'pass_d')
        if mode in spectral_modes or getattr(self, '_route_fade_from', None) in spectral_modes:
            self._prepare_spectral_block(src_block, out_ch)
        else:
            self._spectral_was_active = False

        self._route(mode, src_block, out_block, out_ch)

        if getattr(self, '_route_fade_from', None) is not None:
            fb = getattr(self, '_route_fade_buf', None)
            if fb is None or fb.shape[0] < frames or fb.shape[1] != out_ch:
                fb = self._route_fade_buf = np.zeros((max(frames, 8192), out_ch), dtype=np.float32)
            other = fb[:frames]
            other[:] = 0.0
            self._route(self._route_fade_from, src_block, other, out_ch)
            w = np.clip((self._route_fade_pos + np.arange(1, frames + 1)) / fade_len, 0.0, 1.0).astype(np.float32)
            out_block *= w[:, np.newaxis]
            out_block += other * (1.0 - w)[:, np.newaxis]
            self._route_fade_pos += frames
            if self._route_fade_pos >= fade_len:
                self._route_fade_from = None

        # ─── Post-Routing Global Effects ───
        # Radar balance, or headphone-tracking counter-rotation (applies in
        # both modes: stereo outputs always take the passthrough route)
        is_hp_tracking = getattr(self, 'headphone_tracking', False)
        if getattr(self, 'independent_panning', False) or is_hp_tracking:
            panning_x = self.gui_panning_x if getattr(self, 'independent_panning', False) else 0.0
            if is_hp_tracking:
                panning_x += getattr(self, '_dynamic_pan_offset', 0.0)
            panning_x = max(-1.0, min(1.0, panning_x))

            bal_l = self._get_ramp('_last_bal_l', 1.0 - max(0.0, panning_x), frames)
            bal_r = self._get_ramp('_last_bal_r', 1.0 - max(0.0, -panning_x), frames)

            # Apply to all lefts and all rights based on channel format
            out_block[:, 0] *= bal_l
            out_block[:, 1] *= bal_r
            if out_ch > 5:
                out_block[:, 4] *= bal_l
                out_block[:, 5] *= bal_r
            if out_ch > 7:
                out_block[:, 6] *= bal_l
                out_block[:, 7] *= bal_r

        # Is a real center channel in use? (XTC only makes sense without one)
        self._center_in_use = out_ch > 2 and (
            self.is_multichannel_detected if is_passthrough else bool(self.center_extraction))

        # ─── Post-Upmix Effects Pipeline ───
        self._apply_post_effects(out_block, out_ch, skip_subsynth=force_passthrough,
                                 src_block=src_block, is_passthrough=is_passthrough)

        # Output headroom, then the peak limiter (all channels)
        self._apply_output_headroom(out_block)
        return self._apply_true_peak_limiter(out_block)

    def _apply_output_headroom(self, out_block):
        gain = 10.0 ** (-max(0.0, float(getattr(self, 'output_headroom_db', 0.0))) / 20.0)
        if gain != 1.0 or getattr(self, '_last_headroom', 1.0) != 1.0:
            out_block *= self._get_ramp('_last_headroom', gain, out_block.shape[0])[:, np.newaxis]

    def _route(self, mode, src_block, out_block, out_ch):
        if mode == 'pass':
            self._route_passthrough(src_block, out_block, out_ch)
        elif mode == 'pass_d':
            self._route_passthrough(self._spec_delayed_src, out_block, out_ch)
        elif mode == 'spectral':
            self._route_spectral(out_block, out_ch)
        else:
            self._route_upmix(src_block, out_block, out_ch)

    # ─── Spectral upmix engine ──────────────────────────────────────────

    def _spectral_selected(self, out_ch):
        return bool(getattr(self, 'spectral_upmix', False) and self.upmix_enabled and out_ch > 2)

    def _delay_block(self, key, block, delay):
        """Fixed integer delay with per-key history (works for 1-D or 2-D blocks)."""
        store = self.__dict__.setdefault('_delay_hist', {})
        shape = (delay,) + block.shape[1:]
        hist = store.get(key)
        if hist is None or hist.shape != shape:
            hist = np.zeros(shape, dtype=np.float32)
        if delay <= 0:
            return block
        ext = np.concatenate((hist, block.astype(np.float32, copy=False)))
        store[key] = ext[block.shape[0]:].copy()
        return ext[:block.shape[0]]

    def _prepare_spectral_block(self, src_block, out_ch=8):
        """Run the spectral analysis/resynthesis exactly once per block."""
        engine = getattr(self, '_spectral', None)
        if engine is None or engine.sample_rate != int(self.sample_rate):
            engine = self._spectral = SpectralUpmixer(self.sample_rate)
        if not getattr(self, '_spectral_was_active', False):
            # (Re)activated: start from a clean state, never from stale audio
            engine.reset()
            self._delay_hist = {}
            self._spectral_was_active = True
        width = self.width_factor
        # Cinema / gaming steering: front/back placement from L/R phase
        if getattr(self, 'cinema_steering', False):
            steering = STEER_GAMING if getattr(self, 'gaming_panning', False) else STEER_CINEMA
        else:
            steering = STEER_OFF
        self._spec_streams = engine.process(
            src_block[:, :2], center_spread=self.center_spread, width=width,
            center=self.center_extraction, steering=steering,
            split_sides_backs=out_ch >= 8,
            envelopment=getattr(self, 'envelopment', 0.0),
            wrap=getattr(self, 'wrap_amount', 0.0)).astype(np.float32)
        self._spec_delayed_src = self._delay_block('spec_src', src_block, engine.latency)

    def _route_spectral(self, out_block, out_ch):
        """Front stage from the re-panned direct sound, ambience (and, with
        cinema/gaming steering, rear point sources) to the surrounds.

        Windows channel order: FL FR C LFE BL BR SL SR (5.1: 4/5 = surrounds).
        """
        frames = out_block.shape[0]
        streams = self._spec_streams
        dl, dr, dc, al, ar, side_l, side_r, back_l, back_r = (streams[:, i] for i in range(9))
        d_src = self._spec_delayed_src

        # Ambience: additive, like the commercial upmixers - with the radar in
        # the middle the fronts keep all of it and the surrounds add a
        # decorrelated, delayed copy around the listener; towards the front
        # the surround copy fades out, towards the rear the fronts' share.
        # Surround level by Side Gain.
        f_amb_g, r_amb_g = self._ambience_front_rear()
        f_amb = self._get_ramp('_last_spec_famb', f_amb_g, frames)
        r_amb = self._get_ramp('_last_spec_ramb', r_amb_g * 10 ** (self.side_gain_db / 20.0), frames)

        out_block[:, 0] = dl + f_amb * al
        out_block[:, 1] = dr + f_amb * ar
        if out_ch > 2:
            out_block[:, 2] = dc * (0.707 if self.stage_equalizer else 1.0)

        # LFE: identical to the classic upmix (L+R at -10 dB), on the delayed
        # input so it stays aligned with the spectral output.
        if out_ch > 3:
            lfe = (d_src[:, 0] + d_src[:, 1]) * 0.316
            if getattr(self, 'dynamic_eq', False):
                safe_level = max(getattr(self, 'dyn_eq_level', 0.0), float(np.max(np.abs(d_src))))
                boost_factor = 1.0 + (1.0 - min(1.0, safe_level * 2.0)) * 0.58
                lfe = lfe * self._get_ramp('_last_spec_dyn', boost_factor, frames)
            out_block[:, 3] = 0.0 if self.lfe_muted else lfe

        if out_ch <= 4:
            return
        (amb_sl, amb_sr), amb_backs = self._surround_ambience((al * r_amb, ar * r_amb), out_ch, bank='spectral')
        # + steered rear point sources (cinema / gaming), time-aligned
        if out_ch >= 8:
            out_block[:, 6] = amb_sl + side_l
            out_block[:, 7] = amb_sr + side_r
            out_block[:, 4] = amb_backs[0] + back_l
            out_block[:, 5] = amb_backs[1] + back_r
        else:
            out_block[:, 4] = amb_sl + side_l
            out_block[:, 5] = amb_sr + side_r

    def _ambience_front_rear(self):
        """Front and surround ambience gains from the radar's Y.

        Additive in the middle (front 1, surround 1: the fronts keep their
        ambience and the surrounds add a decorrelated copy, as Dolby Surround
        / Neural:X do - a constant-power split made the fronts drier without
        adding envelopment), front only at the top, surround only at the bottom.
        """
        fade = (1.0 - (self.gui_panning_y + 1.0) / 2.0) * (math.pi / 2.0)
        return min(1.0, math.sqrt(2.0) * math.cos(fade)), min(1.0, math.sqrt(2.0) * math.sin(fade))

    def _bass_split(self, sig, key, cutoff_hz=200.0):
        """Stateful 2nd-order low-pass; (sig - result) is the complementary high part."""
        sos = getattr(self, '_bass_split_sos', None)
        if sos is None or getattr(self, '_bass_split_sr', None) != self.sample_rate:
            sos = self._bass_split_sos = butter(2, cutoff_hz / (self.sample_rate / 2.0), btype='low', output='sos')
            self._bass_split_sr = self.sample_rate
            self._bass_split_zi = {}
        zi = self._ensure_sos_zi(self._bass_split_zi, key, sos)
        low, self._bass_split_zi[key] = sosfilt(sos, sig, zi=zi)
        return low.astype(np.float32)

    def _surround_ambience(self, amb, out_ch, bank):
        """Ambience (one signal or an (L, R) pair) -> surround feeds.

        Psychoacoustics: ambience must not be a coherent copy of the fronts or
        of the other surrounds (comb filtering / phasiness), should arrive
        ~10 ms after the fronts (precedence keeps the image in front), and is
        duller than direct sound (gentle -2.5 dB shelf above ~7 kHz, which also
        hides residual sibilance leaking into the surrounds). On 7.1 the sides
        carry most of it (envelopment comes from lateral energy) and the backs
        a later, further-diffused copy; total power is preserved.

        Returns ((side_l, side_r), (back_l, back_r) or None).
        """
        if isinstance(amb, tuple):
            pair = np.stack(amb, axis=1).astype(np.float32)
        else:
            pair = np.stack((amb, amb), axis=1).astype(np.float32)
        pair = self._ambience_shelf(pair, bank)
        pair = self._delay_block(f'{bank}_haas', pair, int(0.010 * self.sample_rate))
        sl, sr = self._apply_surround_decorrelation(pair[:, 0], pair[:, 1], bank=bank)
        if out_ch < 8:
            return (sl, sr), None
        back = self._delay_block(f'{bank}_back', np.stack((sl, sr), axis=1), int(0.008 * self.sample_rate))
        bl, br = self._apply_surround_decorrelation(back[:, 0], back[:, 1], bank=f'{bank}_back', stages=3)
        return (0.8 * sl, 0.8 * sr), (0.6 * bl, 0.6 * br)

    def _ambience_shelf(self, pair, key, corner_hz=7000.0, cut_db=-2.5):
        """First-order high shelf: x - k * HP(x)."""
        sos = getattr(self, '_amb_shelf_sos', None)
        if sos is None or getattr(self, '_amb_shelf_sr', None) != self.sample_rate:
            sos = self._amb_shelf_sos = butter(1, min(0.95, corner_hz / (self.sample_rate / 2.0)), btype='high', output='sos')
            self._amb_shelf_sr = self.sample_rate
            self._amb_shelf_zi = {}
        k = 1.0 - 10 ** (cut_db / 20.0)
        out = np.empty_like(pair)
        for ch in range(pair.shape[1]):
            zkey = f'{key}_{ch}'
            zi = self._ensure_sos_zi(self._amb_shelf_zi, zkey, sos)
            hp, self._amb_shelf_zi[zkey] = sosfilt(sos, pair[:, ch], zi=zi)
            out[:, ch] = pair[:, ch] - k * hp
        return out

    def _downmix_extra_channels(self, src_block, out_block, out_ch):
        """Fold input channels the output layout lacks into the ones it has.

        Windows channel order: FL FR C LFE BL BR SL SR. Center and surrounds
        go in at -3 dB (ITU-style), the LFE at -10 dB into L/R on outputs
        without an LFE channel; on 5.1 outputs the 7.1 sides fold into 4/5.
        """
        in_ch = src_block.shape[1]
        for ch in range(out_ch, in_ch):
            sig = src_block[:, ch]
            if ch == 2:
                out_block[:, 0] += 0.707 * sig
                out_block[:, 1] += 0.707 * sig
            elif ch == 3:
                out_block[:, 0] += 0.316 * sig
                out_block[:, 1] += 0.316 * sig
            else:
                is_left = ch % 2 == 0
                if out_ch >= 6:
                    target = 4 if is_left else 5
                else:
                    target = 0 if is_left else 1
                out_block[:, target] += 0.707 * sig

    def _route_passthrough(self, src_block, out_block, out_ch):
        """Discrete channels 1:1, extra channels downmixed, bass redirect to LFE."""
        frames, in_ch = src_block.shape
        limit = min(in_ch, out_ch)
        out_block[:, :limit] = src_block[:, :limit]
        self._downmix_extra_channels(src_block, out_block, out_ch)

        # Double Bass: all mains summed and redirected to the LFE at -10 dB,
        # full band: band-limiting and sub/main alignment are done downstream
        # (room correction); filtering here shifted the sub's phase against the
        # mains and weakened the summed bass. Summing every main channel is
        # deliberate: a 5.1/7.1 mix
        # then sends the same bass to the sub as its stereo version does
        # through the upmix ((L+R) * 0.316), so the sub level is consistent
        # between the two modes. Peaks above full scale are handled by the
        # output limiter.
        if out_ch > 3:
            other_channels = [i for i in range(in_ch) if i != 3]
            if other_channels:
                mains_sum = np.sum(src_block[:, other_channels], axis=1) * 0.316
            else:
                mains_sum = np.zeros(frames, dtype=np.float32)
            lfe_original = src_block[:, 3] if in_ch > 3 else 0.0
            out_block[:, 3] = lfe_original + mains_sum

        # Apply Safe Dynamic EQ if enabled and LFE exists (Stereo passthrough won't have out_ch > 3)
        if out_ch > 3 and getattr(self, 'dynamic_eq', False):
            safe_level = max(getattr(self, 'dyn_eq_level', 0.0), np.max(np.abs(src_block)))
            boost_factor = 1.0 + (1.0 - min(1.0, safe_level * 2.0)) * 0.58
            ramp = self._get_ramp('_last_pass_dyn', boost_factor, frames)
            out_block[:, 3] *= ramp

    CLASSIC_MATCH_GAIN = 1.0  # spectral/classic level match, see end of _route_upmix
    # Statistics time constants (s): placement must not wander on music;
    # steering follows faster, gaming fastest (as in the spectral engine).
    # Each surround mixes in a little of the opposite side's ambience (30 deg:
    # 0.87 own - 0.5 other), so it is less coherent with its own front (comb
    # ripple 11 -> 8 dB on diffuse sound) while staying mostly left / right.
    CLASSIC_AMB_CROSS = math.radians(30.0)
    CLASSIC_TAU = {'static': 0.30, 'cinema': 0.07, 'gaming': 0.04}

    def _route_upmix(self, src_block, out_block, out_ch):
        """Classic (time-domain, broadband) correlation-driven upmix.

        L and R never leave their own fronts and nothing is ever played as an
        inverted copy of the other side:
          - the center takes the part that is common to L and R (correlation
            x balance: hard-panned or uncorrelated content gives no center),
            the fronts give up the same power (constant power);
          - the surrounds take the uncorrelated, balanced part (ambience),
            the correlated part cancelled out of each side, decorrelated and
            ~10 ms late (shared helper with the spectral engine);
          - with cinema/gaming steering, opposite-phase (matrix-encoded)
            content leaves the fronts and becomes a same-polarity rear source,
            panned by its L/R balance (7.1 gaming: sides vs backs by depth).
        Below ~200 Hz nothing is moved. One set of broadband gains per block
        from smoothed statistics: cheap, stable, but - unlike the spectral
        engine - it cannot separate sources playing at the same time.
        """
        L = src_block[:, 0]
        R = src_block[:, 1]
        frames = src_block.shape[0]
        L_lo = self._bass_split(L, 'classic_l')
        R_lo = self._bass_split(R, 'classic_r')
        Lh = L - L_lo
        Rh = R - R_lo

        cinema_active = bool(getattr(self, 'cinema_steering', False))
        gaming = cinema_active and bool(getattr(self, 'gaming_panning', False))
        mode = 'gaming' if gaming else ('cinema' if cinema_active else 'static')

        # Smoothed L/R statistics of the band above the bass handover
        now = np.array([np.dot(Lh, Lh), np.dot(Rh, Rh), np.dot(Lh, Rh)], dtype=np.float64) / max(frames, 1)
        st = getattr(self, '_cls_stats', None)
        if st is None:
            st = now
        else:
            k = 1.0 - math.exp(-frames / (self.CLASSIC_TAU[mode] * self.sample_rate))
            st = st + k * (now - st)
        self._cls_stats = st
        pll, prr, plr = float(st[0]), float(st[1]), float(st[2])
        geo = math.sqrt(pll * prr)
        if geo > 1e-12:
            rho = max(-1.0, min(1.0, plr / geo))
            bal = min(1.0, 2.0 * geo / (pll + prr))  # 1 = balanced, 0 = one-sided
        else:
            rho, bal = 0.0, 0.0                      # silence or hard-panned
        rho_pos, rho_neg = max(rho, 0.0), max(-rho, 0.0)

        # Center: common part, scaled by Center Spread (phantom vs center)
        spread = max(0.0, min(1.0, self.center_spread))
        use_center = bool(self.center_extraction) and out_ch > 2
        g_c = (1.0 - spread) * rho_pos * bal if use_center else 0.0

        # Rear (steering only): opposite-phase part
        rear = 0.0
        if cinema_active and out_ch > 4:
            rear = rho_neg if not gaming else max(0.0, min(1.0, (rho_neg - 0.25) / 0.5))
        rear *= bal                                  # hard-panned: no phase info

        # Ambience: uncorrelated and balanced; front/rear by the radar (static)
        amb = (1.0 - abs(rho)) * bal
        fade = (1.0 - (self.gui_panning_y + 1.0) / 2.0) * (math.pi / 2.0)
        s_gain = 10 ** (self.side_gain_db / 20.0)
        f_amb_g, r_amb_g = self._ambience_front_rear()      # additive (see _ambience_front_rear)
        amb_front = 1.0 - amb * (1.0 - f_amb_g)            # gain on L/R: ambience share kept in front
        amb_rear = amb * r_amb_g * s_gain if out_ch > 4 else 0.0
        if out_ch <= 4:
            amb_front = 1.0

        rear = min(1.0, rear)
        k_front = math.sqrt(max(0.0, 1.0 - g_c * g_c)) * math.sqrt(1.0 - rear) * amb_front
        alpha = rho_pos                              # cancels the correlated part out of each surround feed

        kf_r = self._get_ramp('_last_cls_kf', k_front, frames)
        gc_r = self._get_ramp('_last_cls_gc', g_c * math.sqrt(2.0), frames)
        al_r = self._get_ramp('_last_cls_alpha', alpha, frames)
        ar_r = self._get_ramp('_last_cls_amb', amb_rear / math.sqrt(1.0 + alpha * alpha), frames)

        Mid_h = (Lh + Rh) * 0.5
        FL = L_lo + Lh * kf_r
        FR = R_lo + Rh * kf_r
        # Width (front stage, 1.0 = as mixed)
        w = max(0.5, min(2.0, self.width_factor))
        if abs(w - 1.0) > 1e-6:
            fm = (FL + FR) * 0.5
            fs = (FL - FR) * 0.5
            FL, FR = fm + w * fs, fm - w * fs
        Center = Mid_h * gc_r

        # Widening: delayed copy of the mid into the surround ambience
        widening_on = bool(getattr(self, 'widening_mode', False))
        if widening_on and not self._widening_was_on:
            self.widening_buffer.fill(0.0)
        self._widening_was_on = widening_on
        end1 = min(frames, self.widening_buffer_size - self.widening_write_ptr)
        self.widening_buffer[self.widening_write_ptr: self.widening_write_ptr + end1] = Mid_h[:end1]
        if end1 < frames:
            self.widening_buffer[:frames - end1] = Mid_h[end1:frames]
        delay_samples = int(self.sample_rate * (self.widening_delay_ms / 1000.0))
        read_ptr = (self.widening_write_ptr - delay_samples) % self.widening_buffer_size
        delayed_mid = np.zeros(frames, dtype=np.float32)
        end_r1 = min(frames, self.widening_buffer_size - read_ptr)
        delayed_mid[:end_r1] = self.widening_buffer[read_ptr: read_ptr + end_r1]
        if end_r1 < frames:
            delayed_mid[end_r1:frames] = self.widening_buffer[:frames - end_r1]
        self.widening_write_ptr = (self.widening_write_ptr + frames) % self.widening_buffer_size

        sides = None   # (left, right) for the side pair / 5.1 surrounds
        backs = None   # (left, right) for the 7.1 back pair
        if out_ch > 4:
            amb_l = (Lh - al_r * Rh) * ar_r
            amb_r = (Rh - al_r * Lh) * ar_r
            cb, sb = math.cos(self.CLASSIC_AMB_CROSS), math.sin(self.CLASSIC_AMB_CROSS)
            amb_l, amb_r = cb * amb_l - sb * amb_r, cb * amb_r - sb * amb_l
            if widening_on:
                wm = delayed_mid * (0.5 * math.sin(fade) * s_gain)
                amb_l = amb_l + wm
                amb_r = amb_r + wm
            sides, backs = self._surround_ambience((amb_l, amb_r), out_ch, bank='classic')

            # Steered rear point source: same polarity, panned by L/R balance,
            # time-aligned with the fronts and not diffused (positional)
            th = math.atan2(math.sqrt(prr), math.sqrt(pll)) if (pll + prr) > 1e-12 else math.pi / 4
            g_rear = math.sqrt(2.0 * rear)
            if out_ch >= 8:
                if gaming:
                    behind = max(0.0, min(1.0, (rho_neg - 0.5) / 0.5)) * (math.pi / 2)
                    g_side, g_back = math.cos(behind), math.sin(behind)
                else:
                    g_side, g_back = 0.8, 0.6
            else:
                g_side, g_back = 1.0, 0.0
            rl_s = self._get_ramp('_last_cls_rls', g_rear * math.cos(th) * g_side, frames)
            rr_s = self._get_ramp('_last_cls_rrs', g_rear * math.sin(th) * g_side, frames)
            S_d = (Lh - Rh) * 0.5
            sides = (sides[0] + S_d * rl_s, sides[1] + S_d * rr_s)
            if backs is not None:
                rl_b = self._get_ramp('_last_cls_rlb', g_rear * math.cos(th) * g_back, frames)
                rr_b = self._get_ramp('_last_cls_rrb', g_rear * math.sin(th) * g_back, frames)
                backs = (backs[0] + S_d * rl_b, backs[1] + S_d * rr_b)

        # Consistent Sub Level: L+R redirected at -10dB (0.316), full band (see passthrough)
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

        # Level match with the spectral engine (constant-power design: within
        # ~0.2 dB on diffuse / partly coherent material, ~+1 dB on dense
        # panned mixes), so an A/B is not decided by loudness. LFE untouched.
        g = self.CLASSIC_MATCH_GAIN
        out_block[:, 0] = FL * g
        out_block[:, 1] = FR * g
        if out_ch > 2:
            # Apply Stage Equalizer Weighting (-3dB)
            c_weight = 0.707 if self.stage_equalizer else 1.0
            out_block[:, 2] = Center * (c_weight * g)

        if out_ch > 3: out_block[:, 3] = LFE if not self.lfe_muted else 0.0

        # Surrounds, Windows order: FL FR C LFE BL BR SL SR (5.1: 4/5 = surrounds)
        if sides is not None:
            sides = (sides[0] * g, sides[1] * g)
            if out_ch >= 8:
                out_block[:, 6], out_block[:, 7] = sides
                if backs is not None:
                    out_block[:, 4], out_block[:, 5] = backs[0] * g, backs[1] * g
            else:
                out_block[:, 4], out_block[:, 5] = sides

    def _read_input_block(self, frames):
        """Pull `frames` samples from the input ring buffer (None on underrun).

        Applies clock-drift compensation: when the fill level drifts away
        from the latency target, one sample more/less is consumed and the
        block is resampled back to `frames` (inaudible, unlike the hard
        skips/dropouts a drifting virtual cable otherwise causes).
        """
        with self.buffer_lock:
            size = self.ring_buffer_size
            ch = self.buffer.shape[1]
            target = self.target_latency_samples

            # Hard resync after a long stall (stream restart, overflow): jump to
            # the newest audio BEFORE reading, so no stale block is played.
            excess = self.available_samples - frames - target
            if excess > max(2048, 4 * frames):
                self.io_stats['resync'] += 1
                self.read_ptr = (self.read_ptr + excess) % size
                self.available_samples -= excess
                self._drift_fill_ema = None

            # (Re)priming: after start or an underrun, wait until the reserve
            # is back before playing, instead of running at the edge (one
            # short gap rather than a stream of them).
            if not self._in_primed:
                if self.available_samples < target + frames:
                    return None
                self._in_primed = True
                self._drift_fill_ema = None

            self._drift_fill_ema, adj = self._drift_adjust(
                self._drift_fill_ema, self.available_samples - frames, target, frames)
            n_in = frames + adj
            if self.available_samples < n_in:
                # Underrun: the input arrived late. Grow the reserve and re-prime.
                self.io_stats['buffer_empty'] += 1
                self.target_latency_samples = min(self._target_max_samples,
                                                  target + int(self.sample_rate * 0.005))
                self._in_primed = False
                return None

            if self._src_blk_pre.shape[0] < n_in + 1 or self._src_blk_pre.shape[1] != ch:
                self._src_blk_pre = np.zeros((max(8192, 2 * n_in), ch), dtype=np.float32)
            raw = self._src_blk_pre[:n_in, :ch]
            self._ring_read(self.buffer, self.read_ptr, n_in, raw)

            self.read_ptr = (self.read_ptr + n_in) % size
            self.available_samples -= n_in

        if adj == 0:
            return raw
        rs = getattr(self, '_rs_out_pre', None)
        if rs is None or rs.shape[0] < frames or rs.shape[1] != ch:
            rs = self._rs_out_pre = np.zeros((max(8192, frames), ch), dtype=np.float32)
        return self._resample_block(raw, frames, rs)

    def output_callback(self, outdata, frames, time_info, status):
        if status:
            print(f"Output Warning: {status}")
            if getattr(status, 'output_underflow', False):
                self.io_stats['out_underflow'] += 1
        # Any exception escaping a sounddevice callback stops the stream for
        # good, so every failure is caught here and the block is silenced.
        t0 = time.perf_counter()
        try:
            src_block = self._read_input_block(frames)
            if src_block is None:
                outdata.fill(0)
                return
            outdata[:] = self._apply_upmix(src_block, out_ch=outdata.shape[1])
            if time.perf_counter() - t0 > frames / float(self.sample_rate):
                self.io_stats['slow_blocks'] += 1
            if getattr(self, 'secondary_output_enabled', False) and self.secondary_output_stream:
                self._enqueue_secondary_output_block(self._build_secondary_output_block(outdata, src_block=src_block))
        except Exception as e:
            outdata.fill(0)
            self._report_callback_error("output", e)

    def _check_stream_health(self):
        """Reopen streams that PortAudio stopped (device unplugged, driver reset...)."""
        if not self.stream_lock.acquire(blocking=False):
            return  # a device change / calibration is in progress
        try:
            if self.input_stream is not None and not self.input_stream.active:
                print("[Stream] Input stream stopped unexpectedly, reopening...")
                try: self.input_stream.close()
                except Exception: pass
                self.input_stream = None
                self.open_input_stream()
            if self.output_stream is not None and not self.output_stream.active:
                print("[Stream] Output stream stopped unexpectedly, reopening...")
                self.open_output_stream(self.current_channels or self.max_output_ch)
            if self.secondary_output_stream is not None and not self.secondary_output_stream.active:
                print("[Stream] Secondary output stopped unexpectedly, reopening...")
                self.refresh_secondary_output_stream()
        except Exception as e:
            print(f"Stream health check failed: {e}")
        finally:
            self.stream_lock.release()

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
            print(f"Detected hardware: {self.sample_rate}Hz, {self.max_output_ch} channels max")
            self.open_input_stream()

            # Start with the hardware's native channel count for stability
            self.open_output_stream(self.max_output_ch)
            self.refresh_secondary_output_stream()

        sim_phase: float = 0.0
        health_timer = 0.0
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
                    continue

                # Upmixing logic: Lock to max channels for stability.
                # Switching hardware channel counts mid-session often causes
                # [PaErrorCode -9998] or 1-second "Switching..." silences.
                # (open_output_stream returns immediately for counts that failed.)
                if self.max_output_ch != self.current_channels and self.stream_lock.acquire(blocking=False):
                    try:
                        self.open_output_stream(self.max_output_ch)
                    except Exception as e:
                        print(f"Stream error: {e}")
                    finally:
                        self.stream_lock.release()

                health_timer += CHECK_INTERVAL
                if health_timer >= 2.0:
                    health_timer = 0.0
                    self._check_stream_health()

                if self.peak_hit:
                    print("\n[CLIP WARNING] Input hit 0 dBFS! Lower the source volume or run with --gain 0.9")
                    self.peak_hit = False
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

def set_startup(enabled=True):
    import winreg
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


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
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
    upmix_group = parser.add_mutually_exclusive_group()
    upmix_group.add_argument("--upmix", dest="upmix", action="store_true", default=None,
                             help="Force the upmixer ON (default: use the saved setting)")
    upmix_group.add_argument("--no-upmix", dest="upmix", action="store_false",
                             help="Force the upmixer OFF (default: use the saved setting)")
    parser.add_argument("--upmix-gain", type=float, default=None, help=argparse.SUPPRESS)  # deprecated, no effect
    parser.add_argument("--exclusive", action="store_true", help="Use WASAPI Exclusive mode (lowest latency, bypasses EQ APO)")
    parser.add_argument("--gui", action="store_true", help="Show graphical interface")
    args = parser.parse_args()

    if args.startup:
        set_startup(args.startup == 'on')
        sys.exit(0)

    if sd is None and not args.sim:
        print("Error: sounddevice/PortAudio could not be loaded.")
        sys.exit(1)
    if args.upmix_gain is not None:
        print("Note: --upmix-gain is deprecated and has no effect.")

    app = AudioBypass(args.input, args.output, latency=args.latency, simulation=args.sim, gain=args.gain, start_minimized=args.minimized, blocksize=args.blocksize, exclusive_mode=args.exclusive, debug=args.debug)
    if args.upmix is not None:
        app.upmix_enabled = args.upmix

    # Audio runs in a background thread; the main thread is kept for Qt, which
    # must own the main thread. The GUI opens at start (--gui) or on demand
    # from the tray ("Show Mixer").
    audio_thread = threading.Thread(target=app.run, daemon=True)
    audio_thread.start()
    if args.gui:
        app._gui_requested.set()

    try:
        while not app.stop_event.is_set():
            if app._gui_requested.wait(0.2):
                import gui_components
                gui_components.launch_gui(app)  # blocks until app.stop() quits Qt
                break
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        audio_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
