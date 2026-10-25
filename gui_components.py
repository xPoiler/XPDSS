import sys
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QFrame, QGridLayout, QPushButton,
                             QMessageBox, QProgressDialog, QTabWidget, QScrollArea)
from PySide6.QtCore import Qt, QPointF, Signal, Slot, QTimer, QRectF, QThread
import sounddevice as sd
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QRadialGradient, QFont
from PySide6.QtMultimedia import QMediaDevices

class CalibrationWorker(QThread):
    progress_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, audio_app, input_device_id=None):
        super().__init__()
        self.audio_app = audio_app
        self.input_device_id = input_device_id

    def run(self):
        success, msg = self.audio_app.run_binaural_calibration(
            input_device_id=self.input_device_id,
            progress_callback=self.progress_signal.emit
        )
        self.finished_signal.emit(success, msg)

from PySide6.QtWidgets import QDialog, QComboBox

class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Binaural Calibration Setup")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        info = QLabel("Wear your binaural mics inside your ears and sit facing forward.\n\n"
                      "This will play a loud sine sweep to test triggers.\n"
                      "Select your **Binaural Microphone input** below:")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.combo = QComboBox()
        layout.addWidget(self.combo)

        # Populate with sounddevice inputs
        import sounddevice as sd
        try:
             devices = sd.query_devices()
             try:
                 default_input = sd.default.device[0]
             except Exception:
                 default_input = -1

             for i, d in enumerate(devices):
                 if d['max_input_channels'] > 0:
                     self.combo.addItem(f"{d['name']} ({d['max_input_channels']} in)", i)
                     if i == default_input:
                          self.combo.setCurrentIndex(self.combo.count() - 1)
        except Exception as e:
             self.combo.addItem(f"Error listing inputs: {e}")

        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Start Calibration")
        btn_cancel = QPushButton("Cancel")
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)
        layout.addLayout(btn_layout)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

class HaloPanningWidget(QFrame):
    panningChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(450, 450)
        self.puck_pos = QPointF(0, 0)
        self.is_dragging = False
        self.last_peaks = [0.0] * 8
        self.independent_panning = False

        self.color_cyan = QColor(0, 195, 255)
        self.color_bg = QColor(15, 15, 15)
        self.color_grid = QColor(40, 40, 40)

    def set_peaks(self, peaks):
        self.last_peaks = peaks
        self.update()

    def set_independent_panning(self, enabled):
        self.independent_panning = enabled
        if not enabled:
            self.puck_pos.setX(0)
            self.panningChanged.emit(0.0, self.puck_pos.y())
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        is_enabled = self.isEnabled()
        active_color = self.color_cyan if is_enabled else QColor(100, 100, 100)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 40

        painter.setBrush(self.color_bg)
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, 0, w, h)

        painter.setPen(QPen(self.color_grid if is_enabled else QColor(30, 30, 30), 1))
        for r_factor in [0.25, 0.5, 0.75, 1.0]:
            r = radius * r_factor
            painter.drawEllipse(QPointF(cx, cy), r, r)

        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            painter.drawLine(cx, cy, cx + radius * math.cos(rad), cy + radius * math.sin(rad))

        self.draw_speakers(painter, cx, cy, radius, is_enabled)

        puck_x = cx + self.puck_pos.x() * radius
        puck_y = cy - self.puck_pos.y() * radius
        puck_grad = QRadialGradient(puck_x, puck_y, 25)
        if is_enabled:
             puck_grad.setColorAt(0, QColor(0, 255, 255, 180))
             puck_grad.setColorAt(1, QColor(0, 255, 255, 0))
        else:
             puck_grad.setColorAt(0, QColor(100, 100, 100, 150))
             puck_grad.setColorAt(1, QColor(100, 100, 100, 0))

        painter.setBrush(puck_grad)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(puck_x, puck_y), 25, 25)
        painter.setBrush(Qt.white if is_enabled else QColor(130, 130, 130))
        painter.drawEllipse(QPointF(puck_x, puck_y), 6, 6)

        fade_y = int((1.0 - (self.puck_pos.y() + 1.0) / 2.0) * 100)
        painter.setPen(active_color)
        painter.setFont(QFont("Inter", 14, QFont.Medium))
        painter.drawText(QRectF(0, h-60, w, 40), Qt.AlignCenter, f"Fade: {fade_y}%")

    def draw_speakers(self, painter, cx, cy, radius, is_enabled=True):
        win = self.window()
        app = getattr(win, 'audio_app', None)
        num_ch = app.current_channels if app else 2

        if num_ch <= 6:
            speaker_map = {
                0: ("L", -135),  1: ("R", -45),   2: ("C", -90),
                4: ("Sl", 135),  5: ("Sr", 45)
            }
        else:
            speaker_map = {
                0: ("L", -135),  1: ("R", -45),   2: ("C", -90),
                # Windows 7.1 order: 4/5 = back pair, 6/7 = side pair
                4: ("Bl", 135),  5: ("Br", 45),
                6: ("Sl", 180),  7: ("Sr", 0)
            }

        painter.setFont(QFont("Inter", 10, QFont.Bold))
        label_dist = radius + 25
        active_color = self.color_cyan if is_enabled else QColor(100, 100, 100)

        for ch_idx in range(num_ch):
            if ch_idx in speaker_map:
                text, angle = speaker_map[ch_idx]
                rad = math.radians(angle)
                lx = cx + label_dist * math.cos(rad)
                ly = cy + label_dist * math.sin(rad)

                # Level halo from the analysis peaks (0 dBFS -> full halo)
                peak = self.last_peaks[ch_idx] if ch_idx < len(self.last_peaks) else 0.0
                if is_enabled and peak > 0.001:
                    level = max(0.0, min(1.0, 1.0 + 20.0 * math.log10(peak) / 60.0))
                    halo = QColor(active_color)
                    halo.setAlpha(int(60 + 140 * level))
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(halo, 3))
                    painter.drawEllipse(QPointF(lx, ly), 17 + 8 * level, 17 + 8 * level)

                painter.setBrush(active_color)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(lx, ly), 15, 15)
                painter.setPen(Qt.white if is_enabled else QColor(160, 160, 160))
                painter.drawText(QRectF(lx-15, ly-15, 30, 30), Qt.AlignCenter, text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.update_puck_from_mouse(event.position())

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.update_puck_from_mouse(event.position())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False

    def update_puck_from_mouse(self, pos):
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 40
        nx = (pos.x() - cx) / radius if self.independent_panning else 0.0
        ny = (cy - pos.y()) / radius
        dist = math.sqrt(nx*nx + ny*ny)
        if dist > 1.0:
            nx /= dist
            ny /= dist
        self.puck_pos.setX(nx); self.puck_pos.setY(ny)
        self.panningChanged.emit(self.puck_pos.x(), self.puck_pos.y())
        self.update()

class LabeledSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label, min_val, max_val, initial_val, suffix="", is_int=False, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label_txt = label
        self.suffix = suffix
        self.min_val = min_val
        self.max_val = max_val
        self.is_int = is_int

        self._value = float(initial_val)
        self._auto_state = None
        self.lbl_title = QLabel(f"{label}:")
        self.lbl_title.setStyleSheet("color: #888; font-size: 11px; font-weight: bold; min-width: 100px;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)

        norm_val = (initial_val - min_val) / (max_val - min_val)
        self.slider.setValue(int(norm_val * 100))

        from PySide6.QtWidgets import QLineEdit
        self.input_val = QLineEdit(self._format_val(initial_val))
        self.input_val.setStyleSheet("""
            QLineEdit {
                color: #00e5ff;
                font-size: 11px;
                font-weight: bold;
                background: transparent;
                border: none;
                min-width: 50px;
            }
            QLineEdit:focus {
                background: #222;
                border-bottom: 1px solid #00e5ff;
            }
        """)
        self.input_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.input_val.returnPressed.connect(self._on_input_changed)

        self.slider.valueChanged.connect(self._on_value_changed)

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.slider)
        layout.addWidget(self.input_val)

    def _format_val(self, val):
        if self.is_int:
            return f"{int(val)}{self.suffix}"
        return f"{val:.2f}{self.suffix}"

    def _on_value_changed(self, pos):
        val = self.min_val + (pos / 100.0) * (self.max_val - self.min_val)
        self._value = val
        self.input_val.setText(self._format_val(val))
        self.valueChanged.emit(val)

    def _on_input_changed(self):
        text = self.input_val.text().replace(self.suffix, "").strip()
        try:
            val = float(text)
            val = max(self.min_val, min(self.max_val, val))
            norm_val = (val - self.min_val) / (self.max_val - self.min_val)
            self.slider.blockSignals(True)
            self.slider.setValue(int(norm_val * 100))
            self.slider.blockSignals(False)
            self.input_val.setText(self._format_val(val))
            self._value = val
            self.valueChanged.emit(val)
            self.input_val.clearFocus()
        except ValueError:
            pos = self.slider.value()
            val = self.min_val + (pos / 100.0) * (self.max_val - self.min_val)
            self.input_val.setText(self._format_val(val))

    def current_value(self):
        return self._value

    def set_auto_state(self, state):
        """None (manual), 'auto' (moved by the adaptive renderer) or 'pinned'."""
        if state == self._auto_state:
            return
        self._auto_state = state
        suffix = {'auto': ' (auto)', 'pinned': ' (pinned)'}.get(state, '')
        color = {'auto': '#00e5ff', 'pinned': '#ffaa44'}.get(state, '#888')
        self.lbl_title.setText(f"{self.label_txt}{suffix}:")
        self.lbl_title.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold; min-width: 100px;")

    def set_value(self, val):
        self._value = float(val)
        norm_val = (val - self.min_val) / (self.max_val - self.min_val)
        self.slider.blockSignals(True)
        self.slider.setValue(int(norm_val * 100))
        self.slider.blockSignals(False)
        self.input_val.setText(self._format_val(val))


def _make_section_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #555; font-size: 9px; font-weight: bold; margin-top: 8px; margin-bottom: 2px; letter-spacing: 1px;")
    return lbl


class UpmixerWindow(QMainWindow):
    def __init__(self, audio_app):
        super().__init__()
        self.audio_app = audio_app

        # Settings are saved shortly after the last change (slider drags and
        # radar moves used to rewrite config.json on every tick).
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self.audio_app.save_config)

        self.audio_app.gui_window = self

        self.setWindowTitle("XPDSS PRO - Audio Analyzer")
        self.setMinimumSize(700, 550)
        self.setStyleSheet("""
            QMainWindow { background-color: #0c0c0c; }
            QSlider::groove:horizontal { border: 1px solid #2a2a2a; height: 3px; background: #1a1a1a; border-radius: 1px; }
            QSlider::sub-page:horizontal { background: #008899; }
            QSlider::handle:horizontal { background: #00e5ff; border: none; width: 10px; height: 10px; margin: -3.5px 0; border-radius: 5px; }
            QSlider::handle:horizontal:disabled { background: #3a3a3a; }
            QSlider::sub-page:horizontal:disabled { background: #222; }
            QLabel:disabled { color: #444; }
            QLineEdit:disabled { color: #444; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)

        title_lbl = QLabel("SURROUND ANALYSIS CONSOLE")
        title_lbl.setStyleSheet("color: #00e5ff; font-size: 16px; font-weight: bold; letter-spacing: 1px;")
        main_layout.addWidget(title_lbl)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { background: #111; color: #888; padding: 10px 20px; font-weight: bold; border: 1px solid #222; }
            QTabBar::tab:selected { background: #004455; color: #00e5ff; border: 1px solid #00e5ff; }
            QTabWidget::pane { border: 1px solid #333; top: -1px; }
        """)
        main_layout.addWidget(self.tabs, 1)

        # ═══════════════════════════════════════════════════════════════════
        #  TAB 1: MIXER & STEERING
        # ═══════════════════════════════════════════════════════════════════
        tab_mixer = QWidget()
        content_layout = QHBoxLayout(tab_mixer)
        content_layout.addStretch(1)

        self.radar = HaloPanningWidget()
        self.radar.panningChanged.connect(self.on_panning_changed)
        content_layout.addWidget(self.radar, 3)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)

        btn_style = """
            QPushButton { background: #222; color: #888; border-radius: 4px; padding: 12px; font-size: 11px; border: 1px solid #333; font-weight: bold; }
            QPushButton:checked { background: #004455; color: #00e5ff; border: 1px solid #00e5ff; }
            QPushButton:disabled { background: #1a1a1a; color: #444; border: 1px solid #222; }
        """
        btn_sm = btn_style.replace("padding: 12px;", "padding: 6px; font-size: 10px;")

        # ── Mode Selector ──
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(5)

        from PySide6.QtWidgets import QButtonGroup
        self.mode_group = QButtonGroup(self)

        self.btn_mode_spk = QPushButton("SPEAKERS")
        self.btn_mode_spk.setCheckable(True)
        self.btn_mode_spk.setChecked(getattr(self.audio_app, 'current_mode', 'speakers') == 'speakers')
        self.btn_mode_spk.setStyleSheet(btn_style + "padding: 8px;")

        self.btn_mode_hp = QPushButton("HEADPHONES")
        self.btn_mode_hp.setCheckable(True)
        self.btn_mode_hp.setChecked(getattr(self.audio_app, 'current_mode', 'speakers') == 'headphones')
        self.btn_mode_hp.setStyleSheet(btn_style + "padding: 8px;")

        self.mode_group.addButton(self.btn_mode_spk)
        self.mode_group.addButton(self.btn_mode_hp)
        mode_layout.addWidget(self.btn_mode_spk)
        mode_layout.addWidget(self.btn_mode_hp)
        right_panel.addLayout(mode_layout)

        self.btn_mode_spk.clicked.connect(lambda: self.audio_app.switch_mode("speakers"))
        self.btn_mode_hp.clicked.connect(lambda: self.audio_app.switch_mode("headphones"))

        # ── Master Controls ──
        self.btn_bypass = QPushButton("MASTER BYPASS")
        self.btn_bypass.setCheckable(True)
        self.btn_bypass.setChecked(getattr(audio_app, 'bypass_all', False))
        self.btn_bypass.setStyleSheet(btn_style.replace("#004455", "#440000").replace("#00e5ff", "#ff4444"))
        self.btn_bypass.toggled.connect(self.on_bypass_toggled)
        self.btn_bypass.setToolTip("Hard Bypass: Disables all DSP processing.")
        right_panel.addWidget(self.btn_bypass)

        self.btn_master = QPushButton("UPMIX MASTER")
        self.btn_master.setCheckable(True)
        self.btn_master.setChecked(audio_app.upmix_enabled)
        self.btn_master.setStyleSheet(btn_style.replace("#222", "#333"))
        self.btn_master.toggled.connect(lambda chk: (setattr(self.audio_app, "upmix_enabled", chk), self.request_save()))
        right_panel.addWidget(self.btn_master)

        self.btn_ind = QPushButton("INDEPENDENT PANNING")
        self.btn_ind.setCheckable(True)
        self.btn_ind.setStyleSheet(btn_sm)
        self.btn_ind.toggled.connect(self.toggle_independent)
        right_panel.addWidget(self.btn_ind)

        # ── Adaptive Renderer: one switch ──
        right_panel.addWidget(_make_section_label("ADAPTIVE RENDERER"))
        self.btn_adaptive = QPushButton("ADAPTIVE")
        self.btn_adaptive.setCheckable(True)
        self.btn_adaptive.setStyleSheet(btn_sm)
        self.btn_adaptive.setToolTip(
            "Moves the amount sliders of the effects you enabled (clarity, transient,\n"
            "micro-dyn, air, room mix) to suit the input, in real time.\n"
            "Never toggles effects or changes where sounds come from.\n"
            "Dragging a slider pins it; switch adaptive off and on to release pins.")
        self.btn_adaptive.toggled.connect(self.toggle_adaptive)
        right_panel.addWidget(self.btn_adaptive)

        self.lbl_adaptive = QLabel("")
        self.lbl_adaptive.setWordWrap(True)
        self.lbl_adaptive.setStyleSheet("color: #66d9ef; font-family: monospace; font-size: 10px; background: #101820; border: 1px solid #223344; padding: 6px;")
        right_panel.addWidget(self.lbl_adaptive)
        self.refresh_adaptive_ui()

        # ══════════════════════════════════════════════════════════════
        #  GROUPED TOGGLE BUTTONS (2-column grid per section)
        # ══════════════════════════════════════════════════════════════
        self.option_buttons = {}

        button_groups = [
            ("UPMIX ENGINE", [
                ("I/O (MANUAL)", "manual_upmix", "Force upmixing ON even for surround sources."),
                ("SPECTRAL UPMIX", "spectral_upmix",
                 "Frequency-domain upmix: each source is placed across L/C/R from its own\n"
                 "position in the mix, only ambience goes to the surrounds.\n"
                 "Off = classic upmix (A/B, matched in level): same placement rules but one set of\n"
                 "broadband gains, so it cannot separate sources playing at the same time.\n"
                 "Spectral adds 21 ms latency while selected.\n"
                 "Surround ambience in both engines is delayed ~10 ms, decorrelated\n"
                 "and mostly lateral (sides > backs on 7.1).\n"
                 "With CINEMA STEERING: opposite-phase (matrix-encoded) content becomes\n"
                 "rear point sources; with GAMING PAN too: sharper and faster, sides vs backs on 7.1.\n"
                 "Center Spread = phantom vs. dedicated center, Width = front stage width,\n"
                 "Side Gain = surround ambience level, radar Y = ambience front/rear balance."),
                ("STAGE EQUALIZER", "stage_equalizer", "Balance front/surround weight."),
                ("CENTER FOCUS", "center_extraction", "Extract dialogue to center speaker."),
                ("CINEMA STEERING", "cinema_steering",
                 "Opposite-phase (matrix-encoded) content leaves the fronts and plays\n"
                 "from the surrounds as a rear source; gradual front/back fade."),
                ("GAMING PAN", "gaming_panning",
                 "With CINEMA STEERING: decisive front/back decisions and ~2x faster\n"
                 "tracking. On 7.1 also splits rear sources between sides and backs."),
                ("LFE MUTE", "lfe_muted", "Mute the subwoofer output channel."),
            ]),
            ("AUDIO ENHANCE", [
                ("DYNAMIC EQ", "dynamic_eq", "Fletcher-Munson loudness compensation."),
                ("DIALOG FOCUS", "dialogue_focus", "Adaptive speech-presence emphasis for center/dialogue."),
                ("VOCAL LOCK", "vocal_lock", "Detect vocals and boost center clarity."),
                ("TILT CORRECT", "spectral_tilt", "Auto-correct spectral tilt toward warm target."),
                ("BASS EXCITER", "harmonic_exciter", "Psychoacoustic bass harmonics for small speakers."),
                ("SUBHARMONIC", "sub_synth", "Generate subharmonics with voice avoidance."),
                ("LOUDNESS NORM", "loudness_norm", "Slow AGC targeting -23 LUFS across all sources."),
                ("TRANSIENT", "transient_restore", "Restore transient punch on compressed masters."),
                ("DECLIP", "declip", "Detect and restore hard-clipped peaks."),
                ("MICRO DYN", "micro_dynamics", "Upward compression — lift quiet detail without touching peaks."),
                ("CLARITY DSP", "clarity_dsp", "Dynamic de-mudding: Attenuates boxy 200-500Hz resonances."),
                ("AIR EXCITER", "air_exciter", "Harmonic high-frequency detail enhancement."),
                ("FREQ-DEP WIDTH", "freq_dep_width", "Mono bass, normal mids, wide highs."),
            ]),
            ("SPATIAL AUDIO", [
                ("TRANSAURAL XTC", "transaural_widening", "Crosstalk cancellation for a wider stereo soundstage.\n"
                 "Only active when no center channel is in use (it would fight the\n"
                 "center's real phantom image); fades out automatically otherwise."),
                ("HEAD-SHADOW EQ", "head_shadow_eq", "Angle-dependent HF occlusion model."),
                ("3D TRACK (SPK)", "head_tracking", "Head tracking for speaker distance/delay."),
                ("3D TRACK (HP)", "headphone_tracking", "Head tracking for headphone counter-rotation."),
                ("HP EXTERNALIZE", "hp_externalization", "Crossfeed + reflections for out-of-head imaging."),
                ("3D PREVIEW", "head_tracking_preview", "Show head tracking camera preview."),
                ("VIRTUAL ROOM", "virtual_room", "Early reflections (4-18 ms) for spaciousness.\n"
                 "Surround layouts: played from the sides/backs like real wall\n"
                 "reflections, never from fronts, center or LFE. Stereo: into L/R."),
            ]),
        ]

        for section_label, btns in button_groups:
            right_panel.addWidget(_make_section_label(section_label))
            grid = QGridLayout()
            grid.setSpacing(4)
            for i, (text, attr, tooltip) in enumerate(btns):
                btn = QPushButton(text)
                btn.setCheckable(True)
                btn.setChecked(getattr(audio_app, attr, False))
                btn.setStyleSheet(btn_sm)
                btn.setToolTip(tooltip)
                btn.toggled.connect(self._make_toggle_handler(attr))
                self.option_buttons[attr] = btn
                grid.addWidget(btn, i // 2, i % 2)
            right_panel.addLayout(grid)

        # ── Quit ──
        self.btn_quit = QPushButton("QUIT APPLICATION")
        self.btn_quit.setStyleSheet(btn_style + "background: #3a1a1a; color: #ff6666; border: 1px solid #5a2a2a;")
        self.btn_quit.clicked.connect(self.audio_app.stop)
        right_panel.addWidget(self.btn_quit)

        # ══════════════════════════════════════════════════════════════
        #  PARAMETER SLIDERS (contextual visibility)
        # ══════════════════════════════════════════════════════════════
        right_panel.addWidget(_make_section_label("PARAMETERS"))

        self.sld_spread = LabeledSlider("Center Spread", 0.0, 100.0, audio_app.center_spread * 100.0, suffix="%", is_int=True)
        self.sld_spread.valueChanged.connect(lambda v: (setattr(self.audio_app, "center_spread", v / 100.0), self.request_save()))
        right_panel.addWidget(self.sld_spread)

        self.sld_side_gain = LabeledSlider("Side Gain", -6.0, 6.0, audio_app.side_gain_db, suffix=" dB")
        self.sld_side_gain.valueChanged.connect(lambda v: (setattr(self.audio_app, "side_gain_db", v), self.request_save()))
        right_panel.addWidget(self.sld_side_gain)

        self.sld_width = LabeledSlider("Width Factor", 0.5, 2.0, audio_app.width_factor, suffix="x")
        self.sld_width.valueChanged.connect(lambda v: (setattr(self.audio_app, "width_factor", v), self.request_save()))
        right_panel.addWidget(self.sld_width)

        self.sld_sub_gain = LabeledSlider("Sub Gain", 0.0, 5.0, audio_app.sub_gain, suffix="x")
        self.sld_sub_gain.valueChanged.connect(lambda v: (setattr(self.audio_app, "sub_gain", v), self.request_save()))
        right_panel.addWidget(self.sld_sub_gain)

        self.sld_sub_low = LabeledSlider("Sub Low Cutoff", 15.0, 60.0, getattr(audio_app, 'sub_low_cutoff', 20.0), suffix=" Hz")
        self.sld_sub_low.valueChanged.connect(lambda v: (setattr(self.audio_app, "sub_low_cutoff", v), self.request_save()))
        right_panel.addWidget(self.sld_sub_low)

        self.sld_sub_cutoff = LabeledSlider("Sub High Cutoff", 45.0, 200.0, getattr(audio_app, 'sub_drive_cutoff', 80.0), suffix=" Hz")
        self.sld_sub_cutoff.valueChanged.connect(lambda v: (setattr(self.audio_app, "sub_drive_cutoff", v), self.request_save()))
        right_panel.addWidget(self.sld_sub_cutoff)

        self.sld_transient = LabeledSlider("Transient Boost", 1.0, 12.0, getattr(audio_app, 'transient_amount', 4.0), suffix=" dB")
        self.sld_transient.valueChanged.connect(lambda v: self._on_adaptive_slider('transient_amount', v))
        right_panel.addWidget(self.sld_transient)

        self.sld_micro_dyn = LabeledSlider("Micro-Dyn Uplift", 1.0, 12.0, getattr(audio_app, 'micro_dyn_amount', 4.0), suffix=" dB")
        self.sld_micro_dyn.valueChanged.connect(lambda v: self._on_adaptive_slider('micro_dyn_amount', v))
        right_panel.addWidget(self.sld_micro_dyn)

        self.sld_clarity = LabeledSlider("Clarity Atten.", 1.0, 6.0, getattr(audio_app, 'clarity_amount', 4.0), suffix=" dB")
        self.sld_clarity.valueChanged.connect(lambda v: self._on_adaptive_slider('clarity_amount', v))
        right_panel.addWidget(self.sld_clarity)

        self.sld_room_mix = LabeledSlider("Room Mix", 0.0, 100.0, getattr(audio_app, 'virtual_room_mix', 0.25) * 100.0, suffix="%", is_int=True)
        self.sld_room_mix.valueChanged.connect(lambda v: self._on_adaptive_slider('virtual_room_mix', v / 100.0))
        right_panel.addWidget(self.sld_room_mix)

        self.sld_air = LabeledSlider("Air Gain", 0.0, 100.0, getattr(audio_app, 'air_amount', 0.3) * 100.0, suffix="%", is_int=True)
        self.sld_air.valueChanged.connect(lambda v: self._on_adaptive_slider('air_amount', v / 100.0))
        right_panel.addWidget(self.sld_air)

        self.sld_trans_delay = LabeledSlider("XTC Delay", 0.0, 30.0, getattr(audio_app, 'transaural_delay', 7), suffix=" smp", is_int=True)
        self.sld_trans_delay.valueChanged.connect(lambda v: (setattr(self.audio_app, "transaural_delay", int(v)), self.request_save()))
        right_panel.addWidget(self.sld_trans_delay)

        self.sld_trans_gain = LabeledSlider("XTC Gain", 0.0, 1.25, getattr(audio_app, 'transaural_gain', 0.35), suffix="x")
        self.sld_trans_gain.valueChanged.connect(lambda v: (setattr(self.audio_app, "transaural_gain", v), self.request_save()))
        right_panel.addWidget(self.sld_trans_gain)

        self.sld_trans_feedback = LabeledSlider("XTC Feedback", 0.0, 0.40, getattr(audio_app, 'transaural_feedback', 0.08), suffix="x")
        self.sld_trans_feedback.valueChanged.connect(lambda v: (setattr(self.audio_app, "transaural_feedback", v), self.request_save()))
        right_panel.addWidget(self.sld_trans_feedback)

        self.sld_speaker_width = LabeledSlider("Speaker Width", 20.0, 150.0, getattr(audio_app, 'speaker_width_cm', 60.0), suffix=" cm")
        self.sld_speaker_width.valueChanged.connect(lambda v: (setattr(self.audio_app, "speaker_width_cm", v), self.request_save()))
        right_panel.addWidget(self.sld_speaker_width)

        self.sld_camera_fov = LabeledSlider("Camera FOV", 30.0, 120.0, getattr(audio_app, 'camera_fov', 70.0), suffix="°")
        self.sld_camera_fov.valueChanged.connect(lambda v: (setattr(self.audio_app, "camera_fov", v), self.request_save()))
        right_panel.addWidget(self.sld_camera_fov)

        self.sld_speaker_dist = LabeledSlider("Baseline Distance", 30.0, 200.0, getattr(audio_app, 'speaker_distance_cm', 70.0), suffix=" cm")
        self.sld_speaker_dist.valueChanged.connect(lambda v: (setattr(self.audio_app, "speaker_distance_cm", v), self.request_save()))
        right_panel.addWidget(self.sld_speaker_dist)

        self.sld_room_width = LabeledSlider("Room Width", 2.0, 10.0, getattr(audio_app, 'room_width_m', 4.0), suffix=" m")
        self.sld_room_width.valueChanged.connect(lambda v: (setattr(self.audio_app, "room_width_m", v), self.request_save()))
        right_panel.addWidget(self.sld_room_width)

        self.update_slider_states()

        # ── Config Import/Export/Calibrate ──
        config_layout = QHBoxLayout()
        self.btn_import = QPushButton("IMPORT")
        self.btn_import.setStyleSheet(btn_style + "padding: 8px;")
        self.btn_import.clicked.connect(self.import_config)

        self.btn_export = QPushButton("EXPORT")
        self.btn_export.setStyleSheet(btn_style + "padding: 8px;")
        self.btn_export.clicked.connect(self.export_config)

        self.btn_calibrate = QPushButton("CALIBRATE XTC")
        self.btn_calibrate.setStyleSheet(btn_style + "padding: 8px;")
        self.btn_calibrate.setToolTip("Measure ITD / crosstalk with binaural in-ear microphones.")
        self.btn_calibrate.clicked.connect(self.start_calibration)

        config_layout.addWidget(self.btn_import)
        config_layout.addWidget(self.btn_export)
        config_layout.addWidget(self.btn_calibrate)
        right_panel.addLayout(config_layout)

        right_panel.addStretch()

        # Wrap in scroll area
        self.scroll_widget = QWidget()
        self.scroll_widget.setLayout(right_panel)
        self.scroll_widget.setStyleSheet("background-color: #0c0c0c;")
        self.scroll_widget.setMinimumWidth(300)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.scroll_widget)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(260)

        content_layout.addWidget(scroll, 1)
        self.tabs.addTab(tab_mixer, "MIXER & STEERING")

        # ═══════════════════════════════════════════════════════════════════
        #  TAB 2: DEVICE SETUP
        # ═══════════════════════════════════════════════════════════════════
        tab_setup = QWidget()
        setup_layout = QVBoxLayout(tab_setup)
        setup_layout.setSpacing(15)
        setup_layout.setContentsMargins(40, 40, 40, 40)

        lbl_in = QLabel("INPUT DEVICE (WASAPI)")
        lbl_in.setStyleSheet("color: #aaa; font-weight: bold;")
        self.combo_in = QComboBox()
        self.combo_in.setStyleSheet("QComboBox { background: #222; color: #00e5ff; padding: 5px; border: 1px solid #444; }")

        lbl_out = QLabel("OUTPUT DEVICE (WASAPI)")
        lbl_out.setStyleSheet("color: #aaa; font-weight: bold;")
        self.combo_out = QComboBox()
        self.combo_out.setStyleSheet("QComboBox { background: #222; color: #00e5ff; padding: 5px; border: 1px solid #444; }")

        lbl_secondary = QLabel("BASS SHAKER OUTPUT (WASAPI)")
        lbl_secondary.setStyleSheet("color: #aaa; font-weight: bold;")
        self.combo_secondary_out = QComboBox()
        self.combo_secondary_out.setStyleSheet("QComboBox { background: #222; color: #00e5ff; padding: 5px; border: 1px solid #444; }")

        lbl_cam = QLabel("WEBCAM DEVICE (3D TRACKING)")
        lbl_cam.setStyleSheet("color: #aaa; font-weight: bold;")
        self.combo_cam = QComboBox()
        self.combo_cam.setStyleSheet("QComboBox { background: #222; color: #00e5ff; padding: 5px; border: 1px solid #444; }")

        self.btn_apply_dev = QPushButton("APPLY DEVICES")
        self.btn_apply_dev.setStyleSheet(btn_style + "background: #003322; color: #00ffaa; border: 1px solid #00ffaa;")
        self.btn_apply_dev.clicked.connect(self.apply_devices)

        setup_layout.addWidget(lbl_in)
        setup_layout.addWidget(self.combo_in)
        setup_layout.addWidget(lbl_out)
        setup_layout.addWidget(self.combo_out)
        setup_layout.addWidget(lbl_secondary)
        setup_layout.addWidget(self.combo_secondary_out)
        setup_layout.addWidget(lbl_cam)
        setup_layout.addWidget(self.combo_cam)
        setup_layout.addWidget(self.btn_apply_dev)
        setup_layout.addStretch()

        self.tabs.addTab(tab_setup, "DEVICE SETUP")

        # ═══════════════════════════════════════════════════════════════════
        #  TAB 3: BASS SHAKER
        # ═══════════════════════════════════════════════════════════════════
        tab_secondary = QWidget()
        secondary_layout = QVBoxLayout(tab_secondary)
        secondary_layout.setSpacing(14)
        secondary_layout.setContentsMargins(40, 40, 40, 40)

        self.btn_secondary_enable = QPushButton("BASS SHAKER OUTPUT")
        self.btn_secondary_enable.setCheckable(True)
        self.btn_secondary_enable.setStyleSheet(btn_style + "padding: 10px;")
        self.btn_secondary_enable.toggled.connect(self.toggle_secondary_output)
        secondary_layout.addWidget(self.btn_secondary_enable)

        source_row = QHBoxLayout()
        source_lbl = QLabel("SOURCE")
        source_lbl.setStyleSheet("color: #aaa; font-weight: bold;")
        self.combo_secondary_source = QComboBox()
        self.combo_secondary_source.setStyleSheet("QComboBox { background: #222; color: #00e5ff; padding: 5px; border: 1px solid #444; }")
        self.combo_secondary_source.addItem("LFE OR BASS MIX", "lfe_or_bass_mix")
        self.combo_secondary_source.addItem("FRONT BASS MIX", "front_bass_mix")
        self.combo_secondary_source.currentIndexChanged.connect(self.on_secondary_source_changed)
        source_row.addWidget(source_lbl)
        source_row.addWidget(self.combo_secondary_source, 1)
        secondary_layout.addLayout(source_row)

        self.btn_secondary_mono = QPushButton("MONO SUM")
        self.btn_secondary_mono.setCheckable(True)
        self.btn_secondary_mono.setStyleSheet(btn_style.replace("padding: 12px;", "padding: 8px;"))
        self.btn_secondary_mono.toggled.connect(self.toggle_secondary_mono_sum)
        secondary_layout.addWidget(self.btn_secondary_mono)

        self.sld_secondary_gain = LabeledSlider("Gain", -18.0, 12.0, getattr(self.audio_app, 'secondary_output_gain_db', 0.0), " dB")
        self.sld_secondary_lowpass = LabeledSlider("Low-pass", 35.0, 120.0, getattr(self.audio_app, 'secondary_output_lowpass_hz', 80.0), " Hz")
        self.sld_secondary_highpass = LabeledSlider("High-pass", 5.0, 40.0, getattr(self.audio_app, 'secondary_output_highpass_hz', 20.0), " Hz")
        self.sld_secondary_delay = LabeledSlider("Delay", 0.0, 80.0, getattr(self.audio_app, 'secondary_output_delay_ms', 0.0), " ms")
        self.sld_secondary_gain.valueChanged.connect(lambda val: self.on_secondary_slider_changed('secondary_output_gain_db', val))
        self.sld_secondary_lowpass.valueChanged.connect(lambda val: self.on_secondary_slider_changed('secondary_output_lowpass_hz', val))
        self.sld_secondary_highpass.valueChanged.connect(lambda val: self.on_secondary_slider_changed('secondary_output_highpass_hz', val))
        self.sld_secondary_delay.valueChanged.connect(lambda val: self.on_secondary_slider_changed('secondary_output_delay_ms', val))
        secondary_layout.addWidget(self.sld_secondary_gain)
        secondary_layout.addWidget(self.sld_secondary_lowpass)
        secondary_layout.addWidget(self.sld_secondary_highpass)
        secondary_layout.addWidget(self.sld_secondary_delay)

        self.lbl_secondary_status = QLabel("Bass shaker bus follows the main XPDSS render and feeds a dedicated output stream.")
        self.lbl_secondary_status.setWordWrap(True)
        self.lbl_secondary_status.setStyleSheet("color: #66d9ef; font-size: 10px; background: #101820; border: 1px solid #223344; padding: 8px;")
        secondary_layout.addWidget(self.lbl_secondary_status)
        secondary_layout.addStretch()

        self.tabs.addTab(tab_secondary, "BASS SHAKER")
        self.refresh_devices()

        # Initial state sync
        self.sync_ui_from_backend()
        self.radar.puck_pos = QPointF(self.audio_app.gui_panning_x, self.audio_app.gui_panning_y)
        self.radar.set_independent_panning(self.audio_app.independent_panning)
        self.btn_ind.setChecked(self.audio_app.independent_panning)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(40)

        self.update_slider_states()

    def request_save(self):
        self._save_timer.start()

    @Slot()
    def bring_to_front(self):
        """Invoked (queued) from the tray thread: show and focus the window."""
        self.showNormal()
        self.raise_()
        self.activateWindow()

    # ── Toggle handler factory ──────────────────────────────────────────
    def _make_toggle_handler(self, attr):
        def handler(chk):
            setattr(self.audio_app, attr, chk)

            # Mutual exclusion for tracking modes
            if attr == 'head_tracking' and chk:
                if 'headphone_tracking' in self.option_buttons:
                     self.option_buttons['headphone_tracking'].blockSignals(True)
                     self.option_buttons['headphone_tracking'].setChecked(False)
                     setattr(self.audio_app, 'headphone_tracking', False)
                     self.option_buttons['headphone_tracking'].blockSignals(False)
            elif attr == 'headphone_tracking' and chk:
                if 'head_tracking' in self.option_buttons:
                     self.option_buttons['head_tracking'].blockSignals(True)
                     self.option_buttons['head_tracking'].setChecked(False)
                     setattr(self.audio_app, 'head_tracking', False)
                     self.option_buttons['head_tracking'].blockSignals(False)

            # Start/Stop tracker thread
            is_tracking_active = getattr(self.audio_app, 'head_tracking', False) or getattr(self.audio_app, 'headphone_tracking', False)
            if attr in ('head_tracking', 'headphone_tracking'):
                 if is_tracking_active:
                     if hasattr(self.audio_app, 'start_head_tracker'):
                         self.audio_app.start_head_tracker()
                 else:
                     if hasattr(self.audio_app, 'stop_head_tracker'):
                         self.audio_app.stop_head_tracker()

            elif attr == 'head_tracking_preview':
                if hasattr(self.audio_app, 'head_tracker') and self.audio_app.head_tracker:
                    self.audio_app.head_tracker.show_preview = chk

            self.request_save()
            if hasattr(self, 'update_slider_states'):
                self.update_slider_states()
        return handler

    def on_bypass_toggled(self, checked):
        setattr(self.audio_app, "bypass_all", checked)
        self.request_save()
        self.update_slider_states()

    def toggle_independent(self, checked):
        self.audio_app.independent_panning = checked
        self.radar.set_independent_panning(checked)
        self.request_save()

    def toggle_adaptive(self, checked):
        self.audio_app.set_adaptive_enabled(checked)
        self.refresh_adaptive_ui()

    def _on_adaptive_slider(self, attr, value):
        """User edit of an amount slider: pins it while adaptive is on."""
        self.audio_app.set_manual_param(attr, value)
        self.request_save()

    def toggle_secondary_output(self, checked):
        self.audio_app.set_secondary_output_enabled(checked)
        self.update_slider_states()

    def toggle_secondary_mono_sum(self, checked):
        self.audio_app.secondary_output_mono_sum = checked
        self.request_save()
        self.audio_app.refresh_secondary_output_stream()
        self.update_slider_states()

    def on_secondary_source_changed(self, _index):
        self.audio_app.secondary_output_source_mode = self.combo_secondary_source.currentData()
        self.request_save()

    def on_secondary_slider_changed(self, attr, value):
        setattr(self.audio_app, attr, value)
        if attr in ('secondary_output_lowpass_hz', 'secondary_output_highpass_hz'):
            self.audio_app.refresh_secondary_output_filters()
        self.request_save()

    def refresh_adaptive_ui(self):
        if hasattr(self, 'btn_adaptive'):
            self.btn_adaptive.blockSignals(True)
            self.btn_adaptive.setChecked(getattr(self.audio_app, 'adaptive_enabled', False))
            self.btn_adaptive.blockSignals(False)

    def sync_ui_from_backend(self):
        """Synchronize all GUI widget values from the active profile state."""
        app = self.audio_app

        is_spk = (getattr(app, 'current_mode', 'speakers') == 'speakers')
        if hasattr(self, 'btn_mode_spk'): self.btn_mode_spk.setChecked(is_spk)
        if hasattr(self, 'btn_mode_hp'): self.btn_mode_hp.setChecked(not is_spk)

        sliders_map = {
            'sld_spread': ('center_spread', lambda v: v * 100.0),
            'sld_side_gain': ('side_gain_db', None),
            'sld_width': ('width_factor', None),
            'sld_sub_gain': ('sub_gain', None),
            'sld_sub_low': ('sub_low_cutoff', None),
            'sld_sub_cutoff': ('sub_drive_cutoff', None),
            'sld_transient': ('transient_amount', None),
            'sld_micro_dyn': ('micro_dyn_amount', None),
            'sld_clarity': ('clarity_amount', None),
            'sld_room_mix': ('virtual_room_mix', lambda v: v * 100.0),
            'sld_air': ('air_amount', lambda v: v * 100.0),
            'sld_trans_delay': ('transaural_delay', None),
            'sld_trans_gain': ('transaural_gain', None),
            'sld_trans_feedback': ('transaural_feedback', None),
            'sld_speaker_width': ('speaker_width_cm', None),
            'sld_camera_fov': ('camera_fov', None),
            'sld_speaker_dist': ('speaker_distance_cm', None),
            'sld_room_width': ('room_width_m', None),
        }
        for sld_name, (attr, transform) in sliders_map.items():
            if hasattr(self, sld_name):
                val = getattr(app, attr, 0.0)
                if transform:
                    val = transform(val)
                getattr(self, sld_name).set_value(val)

        if hasattr(self, 'option_buttons'):
            for attr, btn in self.option_buttons.items():
                btn.blockSignals(True)
                btn.setChecked(getattr(app, attr, False))
                btn.blockSignals(False)

        if hasattr(self, 'btn_master'): self.btn_master.setChecked(app.upmix_enabled)
        if hasattr(self, 'btn_ind'): self.btn_ind.setChecked(app.independent_panning)
        if hasattr(self, 'btn_bypass'): self.btn_bypass.setChecked(getattr(app, 'bypass_all', False))
        if hasattr(self, 'btn_secondary_enable'):
            self.btn_secondary_enable.blockSignals(True)
            self.btn_secondary_enable.setChecked(getattr(app, 'secondary_output_enabled', False))
            self.btn_secondary_enable.blockSignals(False)
        if hasattr(self, 'btn_secondary_mono'):
            self.btn_secondary_mono.blockSignals(True)
            self.btn_secondary_mono.setChecked(getattr(app, 'secondary_output_mono_sum', True))
            self.btn_secondary_mono.blockSignals(False)
        if hasattr(self, 'combo_secondary_source'):
            idx = self.combo_secondary_source.findData(getattr(app, 'secondary_output_source_mode', 'lfe_or_bass_mix'))
            if idx >= 0:
                self.combo_secondary_source.blockSignals(True)
                self.combo_secondary_source.setCurrentIndex(idx)
                self.combo_secondary_source.blockSignals(False)
        if hasattr(self, 'sld_secondary_gain'): self.sld_secondary_gain.set_value(getattr(app, 'secondary_output_gain_db', 0.0))
        if hasattr(self, 'sld_secondary_lowpass'): self.sld_secondary_lowpass.set_value(getattr(app, 'secondary_output_lowpass_hz', 80.0))
        if hasattr(self, 'sld_secondary_highpass'): self.sld_secondary_highpass.set_value(getattr(app, 'secondary_output_highpass_hz', 20.0))
        if hasattr(self, 'sld_secondary_delay'): self.sld_secondary_delay.set_value(getattr(app, 'secondary_output_delay_ms', 0.0))
        self.refresh_adaptive_ui()

    def update_slider_states(self):
        app = self.audio_app
        current_mode = getattr(app, 'current_mode', 'speakers')
        is_spk = (current_mode == 'speakers')
        is_hp = (current_mode == 'headphones')

        is_cinema = getattr(app, 'cinema_steering', False)
        ch = app.max_output_ch
        is_stereo = (ch <= 2)
        is_bypass = getattr(app, 'bypass_all', False)

        # 1. Master Bypass disables everything
        if hasattr(self, 'btn_master'): self.btn_master.setDisabled(is_bypass)
        if hasattr(self, 'btn_ind'): self.btn_ind.setDisabled(is_bypass)
        if hasattr(self, 'btn_import'): self.btn_import.setDisabled(is_bypass)
        if hasattr(self, 'btn_export'): self.btn_export.setDisabled(is_bypass)
        if hasattr(self, 'btn_calibrate'): self.btn_calibrate.setDisabled(is_bypass)
        if hasattr(self, 'btn_adaptive'): self.btn_adaptive.setDisabled(is_bypass)

        if hasattr(self, 'option_buttons'):
            for b in self.option_buttons.values():
                b.setDisabled(is_bypass)

        all_sliders = ['sld_spread', 'sld_width', 'sld_side_gain', 'sld_sub_gain', 'sld_sub_cutoff',
                       'sld_sub_low', 'sld_transient', 'sld_micro_dyn', 'sld_clarity', 'sld_room_mix', 'sld_air',
                       'sld_trans_delay', 'sld_trans_gain', 'sld_trans_feedback',
                       'sld_speaker_width', 'sld_camera_fov', 'sld_speaker_dist', 'sld_room_width']
        for s_name in all_sliders:
            if hasattr(self, s_name):
                getattr(self, s_name).setDisabled(is_bypass)

        if hasattr(self, 'radar'):
            self.radar.setDisabled(is_bypass or is_stereo or is_hp)

        secondary_enabled = getattr(app, 'secondary_output_enabled', False)
        secondary_controls = [
            'combo_secondary_source',
            'sld_secondary_gain',
            'sld_secondary_lowpass',
            'sld_secondary_highpass',
            'sld_secondary_delay',
            'btn_secondary_mono',
        ]
        for name in secondary_controls:
            if hasattr(self, name):
                getattr(self, name).setDisabled(is_bypass or not secondary_enabled)

        if is_bypass:
            return

        # 3. Stereo output disables surround features
        if is_stereo:
            self.btn_master.setEnabled(False)
            self.btn_master.setText("UPMIX (REQ SURROUND)")
            surround_features = ['center_extraction', 'stage_equalizer', 'cinema_steering', 'gaming_panning', 'vocal_lock', 'lfe_muted', 'spectral_upmix']
            for f in surround_features:
                if f in self.option_buttons:
                    self.option_buttons[f].setEnabled(False)
        else:
            self.btn_master.setEnabled(True)
            self.btn_master.setText("UPMIX MASTER")

        if hasattr(self, 'btn_master'):
             self.btn_master.setVisible(is_spk)

        # 4. Mode-exclusive visibility
        speaker_only = ['transaural_widening', 'cinema_steering', 'gaming_panning', 'stage_equalizer', 'spectral_upmix',
                        'head_tracking', 'center_extraction', 'lfe_muted', 'vocal_lock', 'freq_dep_width']
        headphone_only = ['head_shadow_eq', 'headphone_tracking', 'hp_externalization']

        for f in speaker_only:
            if f in self.option_buttons:
                self.option_buttons[f].setVisible(is_spk)
        for f in headphone_only:
            if f in self.option_buttons:
                self.option_buttons[f].setVisible(is_hp)

        # Gaming requires Cinema
        if 'gaming_panning' in self.option_buttons:
            if is_spk and not is_stereo:
                 self.option_buttons['gaming_panning'].setEnabled(is_cinema)

        # 5. Cascading slider visibility
        is_sub = getattr(app, 'sub_synth', False)
        if hasattr(self, 'sld_sub_gain'): self.sld_sub_gain.setVisible(is_sub)
        if hasattr(self, 'sld_sub_low'): self.sld_sub_low.setVisible(is_sub)
        if hasattr(self, 'sld_sub_cutoff'): self.sld_sub_cutoff.setVisible(is_sub)

        is_transient = getattr(app, 'transient_restore', False)
        if hasattr(self, 'sld_transient'): self.sld_transient.setVisible(is_transient)

        is_micro_dyn = getattr(app, 'micro_dynamics', False)
        if hasattr(self, 'sld_micro_dyn'): self.sld_micro_dyn.setVisible(is_micro_dyn)

        is_clarity = getattr(app, 'clarity_dsp', False)
        if hasattr(self, 'sld_clarity'): self.sld_clarity.setVisible(is_clarity)

        is_room = getattr(app, 'virtual_room', False)
        if hasattr(self, 'sld_room_mix'): self.sld_room_mix.setVisible(is_room)

        is_air = getattr(app, 'air_exciter', False)
        if hasattr(self, 'sld_air'): self.sld_air.setVisible(is_air)

        is_trans = getattr(app, 'transaural_widening', False) and is_spk
        if hasattr(self, 'sld_trans_delay'): self.sld_trans_delay.setVisible(is_trans)
        if hasattr(self, 'sld_trans_gain'): self.sld_trans_gain.setVisible(is_trans)
        if hasattr(self, 'sld_trans_feedback'): self.sld_trans_feedback.setVisible(is_trans)

        is_any_tracking = getattr(app, 'head_tracking', False) or getattr(app, 'headphone_tracking', False)
        if hasattr(self, 'sld_speaker_width'): self.sld_speaker_width.setVisible(is_any_tracking and is_spk)
        if hasattr(self, 'sld_camera_fov'): self.sld_camera_fov.setVisible(is_any_tracking)
        if hasattr(self, 'sld_speaker_dist'): self.sld_speaker_dist.setVisible(is_any_tracking and is_spk)
        if hasattr(self, 'sld_room_width'): self.sld_room_width.setVisible(is_any_tracking and is_spk)

        if 'head_tracking_preview' in self.option_buttons:
             self.option_buttons['head_tracking_preview'].setVisible(is_any_tracking)

    def refresh_devices(self):
        self.combo_in.clear()
        self.combo_out.clear()
        self.combo_secondary_out.clear()
        self.combo_secondary_out.addItem("Disabled / None", "")

        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
            wasapi_idx = -1
            for i, h in enumerate(host_apis):
                if 'wasapi' in h['name'].lower():
                    wasapi_idx = i
                    break

            if wasapi_idx == -1: return

            for i, d in enumerate(devices):
                if d['hostapi'] == wasapi_idx:
                    if d['max_input_channels'] > 0:
                        self.combo_in.addItem(f"{d['name']} ({d['max_input_channels']} In)", d['name'])
                    if d['max_output_channels'] > 0:
                        self.combo_out.addItem(f"{d['name']} ({d['max_output_channels']} Out)", d['name'])
                        self.combo_secondary_out.addItem(f"{d['name']} ({d['max_output_channels']} Out)", d['name'])

            idx_in = self.combo_in.findData(self.audio_app.input_name)
            if idx_in >= 0: self.combo_in.setCurrentIndex(idx_in)

            idx_out = self.combo_out.findData(self.audio_app.output_name)
            if idx_out >= 0: self.combo_out.setCurrentIndex(idx_out)

            idx_secondary = self.combo_secondary_out.findData(getattr(self.audio_app, 'secondary_output_device', ''))
            if idx_secondary >= 0:
                self.combo_secondary_out.setCurrentIndex(idx_secondary)

            self.combo_cam.clear()
            self.combo_cam.addItem("Default (Index 0)", "")
            cameras = QMediaDevices.videoInputs()
            for c in cameras:
                self.combo_cam.addItem(c.description(), c.description())

            self.combo_cam.insertSeparator(self.combo_cam.count())
            self.combo_cam.addItem("Force Index 0", "idx:0")
            self.combo_cam.addItem("Force Index 1", "idx:1")
            self.combo_cam.addItem("Force Index 2", "idx:2")
            self.combo_cam.addItem("Force Index 3", "idx:3")

            current_webcam = getattr(self.audio_app, 'webcam_name', "")
            idx_cam = self.combo_cam.findData(current_webcam)
            if idx_cam >= 0: self.combo_cam.setCurrentIndex(idx_cam)
        except Exception as e:
            print("Could not query devices:", e)

    def apply_devices(self):
        in_name = self.combo_in.currentData()
        out_name = self.combo_out.currentData()
        secondary_out_name = self.combo_secondary_out.currentData()
        if in_name and out_name:
            self.btn_apply_dev.setText("APPLYING...")
            self.btn_apply_dev.setEnabled(False)
            QApplication.processEvents()

            try:
                self.audio_app.change_devices(in_name, out_name, secondary_out_name)

                cam_name = self.combo_cam.currentData()
                if cam_name and cam_name.startswith("idx:"):
                    cam_idx = int(cam_name.split(":")[1])
                else:
                    cam_idx = self.combo_cam.currentIndex() - 1
                    if cam_idx < 0: cam_idx = 0

                self.audio_app.webcam_name = cam_name
                self.audio_app.webcam_index = cam_idx
                self.request_save()

                if hasattr(self.audio_app, 'head_tracker') and self.audio_app.head_tracker:
                     self.audio_app.stop_head_tracker()
                     if hasattr(self.audio_app, 'start_head_tracker'):
                          self.audio_app.start_head_tracker()

                self.update_slider_states()
            except Exception as e:
                QMessageBox.critical(self, "Device Error", str(e))
                self.refresh_devices()

            self.btn_apply_dev.setText("APPLY DEVICES")
            self.btn_apply_dev.setEnabled(True)

    def on_panning_changed(self, x, y):
        self.audio_app.gui_panning_x = x
        self.audio_app.gui_panning_y = y
        self.request_save()

    def import_config(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Import Config", "", "JSON Files (*.json)")
        if path:
            self.audio_app.load_config(path)
            self.refresh_devices()
            self.radar.puck_pos = QPointF(self.audio_app.gui_panning_x, self.audio_app.gui_panning_y)
            self.btn_ind.setChecked(self.audio_app.independent_panning)
            self.radar.set_independent_panning(self.audio_app.independent_panning)
            self.sync_ui_from_backend()
            self.update_slider_states()
            self.update_gui()

    def export_config(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Export Config", "my_config.json", "JSON Files (*.json)")
        if path:
            self.audio_app.save_config(path)

    def update_gui(self):
        peaks = getattr(self.audio_app, 'last_peaks', [0.0]*8)
        self.radar.set_peaks(peaks)

        self._update_adaptive_display()

        if hasattr(self, 'lbl_secondary_status'):
            enabled = getattr(self.audio_app, 'secondary_output_enabled', False)
            device = getattr(self.audio_app, 'secondary_output_device', '') or 'Disabled / None'
            source = getattr(self.audio_app, 'secondary_output_source_mode', 'lfe_or_bass_mix').replace('_', ' ').upper()
            mode = 'MONO' if getattr(self.audio_app, 'secondary_output_mono_sum', True) else 'STEREO'
            self.lbl_secondary_status.setText(
                f"ENABLED: {'YES' if enabled else 'NO'}\n"
                f"device={device}\n"
                f"source={source}  mode={mode}\n"
                f"gain={getattr(self.audio_app, 'secondary_output_gain_db', 0.0):.2f} dB  lp={getattr(self.audio_app, 'secondary_output_lowpass_hz', 80.0):.1f} Hz  hp={getattr(self.audio_app, 'secondary_output_highpass_hz', 20.0):.1f} Hz  delay={getattr(self.audio_app, 'secondary_output_delay_ms', 0.0):.1f} ms"
            )

    # Amount sliders driven by the adaptive renderer: attr -> (slider, display transform)
    ADAPTIVE_SLIDERS = {
        'clarity_amount': ('sld_clarity', 1.0),
        'transient_amount': ('sld_transient', 1.0),
        'micro_dyn_amount': ('sld_micro_dyn', 1.0),
        'air_amount': ('sld_air', 100.0),
        'virtual_room_mix': ('sld_room_mix', 100.0),
    }

    def _update_adaptive_display(self):
        app = self.audio_app
        enabled = getattr(app, 'adaptive_enabled', False)
        pinned = getattr(app, 'adaptive_pinned', set())
        for attr, (sld_name, scale) in self.ADAPTIVE_SLIDERS.items():
            sld = getattr(self, sld_name, None)
            if sld is None:
                continue
            if enabled:
                sld.set_auto_state('pinned' if attr in pinned else 'auto')
            else:
                sld.set_auto_state(None)
            # Show the live value (also while gliding back after switching off)
            if attr not in pinned and not sld.slider.isSliderDown():
                live = app._live_param(attr)
                if abs(live * scale - sld.current_value()) > 1e-6:
                    sld.set_value(live * scale)

        if hasattr(self, 'lbl_adaptive'):
            if not enabled:
                self.lbl_adaptive.setVisible(False)
                return
            self.lbl_adaptive.setVisible(True)
            snap = app.get_adaptive_snapshot()
            if not snap:
                self.lbl_adaptive.setText("waiting for signal...")
                return
            dyn = snap.get('dynamic_range_db', float('nan'))
            dyn_txt = "measuring" if dyn != dyn else f"{dyn:.1f} dB"
            held = "" if app.adaptive_engine.last_block_active else "  (silence: holding)"
            self.lbl_adaptive.setText(
                f"dynamics {dyn_txt}  mud {snap.get('mud_score', 0):.2f}  air {snap.get('air_score', 0):.2f}\n"
                f"dialogue {snap.get('dialogue_score', 0):.2f}  spaciousness {snap.get('immersion_score', 0):.2f}{held}")

    def start_calibration(self):
        from PySide6.QtWidgets import QDialog
        dialog = CalibrationDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        selected_device = dialog.combo.currentData()
        if not isinstance(selected_device, int):
            return

        # No cancel button: the measurement cannot be interrupted safely mid-sweep
        progress = QProgressDialog("Initializing calibration...", None, 0, 0, self)
        progress.setWindowTitle("Calibrating...")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        worker = CalibrationWorker(self.audio_app, input_device_id=selected_device)
        self._calibration_worker = worker  # keep a reference while it runs

        def _update_progress(text):
            progress.setLabelText(text)

        def _finish(success, message):
            progress.close()
            if success:
                QMessageBox.information(self, "Success", f"Calibration Complete!\n\n{message}")
                if hasattr(self, 'sld_trans_delay'):
                    self.sld_trans_delay.set_value(self.audio_app.transaural_delay)
                if hasattr(self, 'sld_trans_gain'):
                    self.sld_trans_gain.set_value(self.audio_app.transaural_gain)

                if 'transaural_widening' in self.option_buttons:
                     self.option_buttons['transaural_widening'].setChecked(True)
            else:
                 QMessageBox.critical(self, "Calibration Failed", f"An error occurred:\n{message}")
            self._calibration_worker = None
            worker.deleteLater()

        worker.progress_signal.connect(_update_progress)
        worker.finished_signal.connect(_finish)
        worker.start()

    def closeEvent(self, event):
        if self._save_timer.isActive():
            self._save_timer.stop()
            self.audio_app.save_config()
        if getattr(self.audio_app, 'tray_icon', None) is None:
            # No tray icon to bring the window back: closing means quitting
            event.ignore()
            self.audio_app.stop()
            return
        # Keep running in the tray
        event.ignore()
        self.hide()

def launch_gui(audio_app):
    app = QApplication.instance() or QApplication(sys.argv)
    # The window hides on close (tray keeps running); quitting is explicit
    app.setQuitOnLastWindowClosed(False)
    window = UpmixerWindow(audio_app)
    window.show()
    return app.exec()
