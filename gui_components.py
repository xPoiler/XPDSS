import sys
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QFrame, QGridLayout, QPushButton,
                             QMessageBox, QProgressDialog, QTabWidget, QScrollArea)
from PySide6.QtCore import Qt, QPointF, Signal, QTimer, QRectF, QThread
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
             except:
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
                4: ("Sl", 180),  5: ("Sr", 0),
                6: ("Bl", 135),  7: ("Br", 45)
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
            self.valueChanged.emit(val)
            self.input_val.clearFocus()
        except ValueError:
            pos = self.slider.value()
            val = self.min_val + (pos / 100.0) * (self.max_val - self.min_val)
            self.input_val.setText(self._format_val(val))

    def set_value(self, val):
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
        self.audio_app.gui_window = self
        self.audio_app.gain = 1.0

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
        self.btn_master.toggled.connect(lambda chk: (setattr(self.audio_app, "upmix_enabled", chk), self.audio_app.save_config()))
        right_panel.addWidget(self.btn_master)

        self.btn_ind = QPushButton("INDEPENDENT PANNING")
        self.btn_ind.setCheckable(True)
        self.btn_ind.setStyleSheet(btn_sm)
        self.btn_ind.toggled.connect(self.toggle_independent)
        right_panel.addWidget(self.btn_ind)

        # ══════════════════════════════════════════════════════════════
        #  GROUPED TOGGLE BUTTONS (2-column grid per section)
        # ══════════════════════════════════════════════════════════════
        self.option_buttons = {}

        button_groups = [
            ("UPMIX ENGINE", [
                ("I/O (MANUAL)", "manual_upmix", "Force upmixing ON even for surround sources."),
                ("STAGE EQUALIZER", "stage_equalizer", "Balance front/surround weight."),
                ("CENTER FOCUS", "center_extraction", "Extract dialogue to center speaker."),
                ("CINEMA STEERING", "cinema_steering", "Phase-detector surround steering."),
                ("GAMING PAN", "gaming_panning", "Extreme pans to side/rear surrounds."),
                ("LFE MUTE", "lfe_muted", "Mute the subwoofer output channel."),
            ]),
            ("AUDIO ENHANCE", [
                ("DYNAMIC EQ", "dynamic_eq", "Fletcher-Munson loudness compensation."),
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
                ("TRANSAURAL XTC", "transaural_widening", "Crosstalk cancellation for wider soundstage."),
                ("SURR DECORR", "surround_decorrelation", "All-pass diffuser for ambient separation from vocals."),
                ("HEAD-SHADOW EQ", "head_shadow_eq", "Angle-dependent HF occlusion model."),
                ("3D TRACK (SPK)", "head_tracking", "Head tracking for speaker distance/delay."),
                ("3D TRACK (HP)", "headphone_tracking", "Head tracking for headphone counter-rotation."),
                ("HP EXTERNALIZE", "hp_externalization", "Crossfeed + reflections for out-of-head imaging."),
                ("3D PREVIEW", "head_tracking_preview", "Show head tracking camera preview."),
                ("VIRTUAL ROOM", "virtual_room", "Simulate early reflections of an acoustic room."),
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
        self.sld_spread.valueChanged.connect(lambda v: (setattr(self.audio_app, "center_spread", v / 100.0), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_spread)

        self.sld_side_gain = LabeledSlider("Side Gain", -6.0, 6.0, audio_app.side_gain_db, suffix=" dB")
        self.sld_side_gain.valueChanged.connect(lambda v: (setattr(self.audio_app, "side_gain_db", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_side_gain)

        self.sld_width = LabeledSlider("Width Factor", 0.5, 2.0, audio_app.width_factor, suffix="x")
        self.sld_width.valueChanged.connect(lambda v: (setattr(self.audio_app, "width_factor", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_width)

        self.sld_sub_gain = LabeledSlider("Sub Gain", 0.0, 5.0, audio_app.sub_gain, suffix="x")
        self.sld_sub_gain.valueChanged.connect(lambda v: (setattr(self.audio_app, "sub_gain", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_sub_gain)

        self.sld_sub_low = LabeledSlider("Sub Low Cutoff", 15.0, 60.0, getattr(audio_app, 'sub_low_cutoff', 20.0), suffix=" Hz")
        self.sld_sub_low.valueChanged.connect(lambda v: (setattr(self.audio_app, "sub_low_cutoff", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_sub_low)

        self.sld_sub_cutoff = LabeledSlider("Sub High Cutoff", 45.0, 200.0, getattr(audio_app, 'sub_drive_cutoff', 80.0), suffix=" Hz")
        self.sld_sub_cutoff.valueChanged.connect(lambda v: (setattr(self.audio_app, "sub_drive_cutoff", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_sub_cutoff)

        self.sld_transient = LabeledSlider("Transient Boost", 1.0, 12.0, getattr(audio_app, 'transient_amount', 4.0), suffix=" dB")
        self.sld_transient.valueChanged.connect(lambda v: (setattr(self.audio_app, "transient_amount", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_transient)

        self.sld_micro_dyn = LabeledSlider("Micro-Dyn Uplift", 1.0, 12.0, getattr(audio_app, 'micro_dyn_amount', 4.0), suffix=" dB")
        self.sld_micro_dyn.valueChanged.connect(lambda v: (setattr(self.audio_app, "micro_dyn_amount", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_micro_dyn)

        self.sld_clarity = LabeledSlider("Clarity Atten.", 1.0, 6.0, getattr(audio_app, 'clarity_amount', 4.0), suffix=" dB")
        self.sld_clarity.valueChanged.connect(lambda v: (setattr(self.audio_app, "clarity_amount", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_clarity)

        self.sld_room_mix = LabeledSlider("Room Mix", 0.0, 100.0, getattr(audio_app, 'virtual_room_mix', 0.25) * 100.0, suffix="%", is_int=True)
        self.sld_room_mix.valueChanged.connect(lambda v: (setattr(self.audio_app, "virtual_room_mix", v / 100.0), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_room_mix)

        self.sld_air = LabeledSlider("Air Gain", 0.0, 100.0, getattr(audio_app, 'air_amount', 0.3) * 100.0, suffix="%", is_int=True)
        self.sld_air.valueChanged.connect(lambda v: (setattr(self.audio_app, "air_amount", v / 100.0), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_air)

        self.sld_trans_delay = LabeledSlider("XTC Delay", 0.0, 30.0, getattr(audio_app, 'transaural_delay', 7), suffix=" smp", is_int=True)
        self.sld_trans_delay.valueChanged.connect(lambda v: (setattr(self.audio_app, "transaural_delay", int(v)), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_trans_delay)

        self.sld_trans_gain = LabeledSlider("XTC Gain", 0.0, 1.25, getattr(audio_app, 'transaural_gain', 0.35), suffix="x")
        self.sld_trans_gain.valueChanged.connect(lambda v: (setattr(self.audio_app, "transaural_gain", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_trans_gain)

        self.sld_trans_feedback = LabeledSlider("XTC Feedback", 0.0, 0.40, getattr(audio_app, 'transaural_feedback', 0.08), suffix="x")
        self.sld_trans_feedback.valueChanged.connect(lambda v: (setattr(self.audio_app, "transaural_feedback", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_trans_feedback)

        self.sld_speaker_width = LabeledSlider("Speaker Width", 20.0, 150.0, getattr(audio_app, 'speaker_width_cm', 60.0), suffix=" cm")
        self.sld_speaker_width.valueChanged.connect(lambda v: (setattr(self.audio_app, "speaker_width_cm", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_speaker_width)

        self.sld_camera_fov = LabeledSlider("Camera FOV", 30.0, 120.0, getattr(audio_app, 'camera_fov', 70.0), suffix="°")
        self.sld_camera_fov.valueChanged.connect(lambda v: (setattr(self.audio_app, "camera_fov", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_camera_fov)

        self.sld_speaker_dist = LabeledSlider("Baseline Distance", 30.0, 200.0, getattr(audio_app, 'speaker_distance_cm', 70.0), suffix=" cm")
        self.sld_speaker_dist.valueChanged.connect(lambda v: (setattr(self.audio_app, "speaker_distance_cm", v), self.audio_app.save_config()))
        right_panel.addWidget(self.sld_speaker_dist)

        self.sld_room_width = LabeledSlider("Room Width", 2.0, 10.0, getattr(audio_app, 'room_width_m', 4.0), suffix=" m")
        self.sld_room_width.valueChanged.connect(lambda v: (setattr(self.audio_app, "room_width_m", v), self.audio_app.save_config()))
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

        self.btn_calibrate = QPushButton("CALIBRATE BINAURAL")
        self.btn_calibrate.setStyleSheet(btn_style + "padding: 8px; background: #003322; color: #00ffaa; border: 1px solid #00ffaa;")
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
        setup_layout.addWidget(lbl_cam)
        setup_layout.addWidget(self.combo_cam)
        setup_layout.addWidget(self.btn_apply_dev)
        setup_layout.addStretch()

        self.tabs.addTab(tab_setup, "DEVICE SETUP")
        self.refresh_devices()

        # Initial state sync
        self.radar.puck_pos = QPointF(self.audio_app.gui_panning_x, self.audio_app.gui_panning_y)
        self.radar.set_independent_panning(self.audio_app.independent_panning)
        self.btn_ind.setChecked(self.audio_app.independent_panning)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(40)

        self.update_slider_states()

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

            self.audio_app.save_config()
            if hasattr(self, 'update_slider_states'):
                self.update_slider_states()
        return handler

    def on_bypass_toggled(self, checked):
        setattr(self.audio_app, "bypass_all", checked)
        self.audio_app.save_config()
        self.update_slider_states()

    def toggle_independent(self, checked):
        self.audio_app.independent_panning = checked
        self.radar.set_independent_panning(checked)
        self.audio_app.save_config()

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

        if hasattr(self, 'option_buttons'):
            for b in self.option_buttons.values():
                b.setDisabled(is_bypass)

        all_sliders = ['sld_spread', 'sld_width', 'sld_side_gain', 'sld_sub_gain', 'sld_sub_cutoff',
                       'sld_sub_low', 'sld_transient',
                       'sld_trans_delay', 'sld_trans_gain', 'sld_trans_feedback',
                       'sld_speaker_width', 'sld_camera_fov', 'sld_speaker_dist', 'sld_room_width']
        for s_name in all_sliders:
            if hasattr(self, s_name):
                getattr(self, s_name).setDisabled(is_bypass)

        if hasattr(self, 'radar'):
            self.radar.setDisabled(is_bypass or is_cinema or is_stereo or is_hp)

        if is_bypass:
            return

        # 2. Cinema disables manual steering sliders
        if is_cinema:
            for s_name in ['sld_spread', 'sld_side_gain', 'sld_width']:
                if hasattr(self, s_name):
                    getattr(self, s_name).setDisabled(True)

        # 3. Stereo output disables surround features
        if is_stereo:
            self.btn_master.setEnabled(False)
            self.btn_master.setText("UPMIX (REQ SURROUND)")
            surround_features = ['center_extraction', 'stage_equalizer', 'cinema_steering', 'gaming_panning', 'vocal_lock', 'lfe_muted']
            for f in surround_features:
                if f in self.option_buttons:
                    self.option_buttons[f].setEnabled(False)
        else:
            self.btn_master.setEnabled(True)
            self.btn_master.setText("UPMIX MASTER")

        if hasattr(self, 'btn_master'):
             self.btn_master.setVisible(is_spk)

        # 4. Mode-exclusive visibility
        speaker_only = ['transaural_widening', 'cinema_steering', 'gaming_panning', 'stage_equalizer',
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

            idx_in = self.combo_in.findData(self.audio_app.input_name)
            if idx_in >= 0: self.combo_in.setCurrentIndex(idx_in)

            idx_out = self.combo_out.findData(self.audio_app.output_name)
            if idx_out >= 0: self.combo_out.setCurrentIndex(idx_out)

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
        if in_name and out_name:
            self.btn_apply_dev.setText("APPLYING...")
            self.btn_apply_dev.setEnabled(False)
            QApplication.processEvents()

            try:
                self.audio_app.change_devices(in_name, out_name)

                cam_name = self.combo_cam.currentData()
                if cam_name and cam_name.startswith("idx:"):
                    cam_idx = int(cam_name.split(":")[1])
                else:
                    cam_idx = self.combo_cam.currentIndex() - 1
                    if cam_idx < 0: cam_idx = 0

                self.audio_app.webcam_name = cam_name
                self.audio_app.webcam_index = cam_idx
                self.audio_app.save_config()

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
        self.audio_app.save_config()

    def import_config(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Import Config", "", "JSON Files (*.json)")
        if path:
            self.audio_app.load_config(path)
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

    def start_calibration(self):
        from PySide6.QtWidgets import QDialog
        dialog = CalibrationDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        selected_device = dialog.combo.currentData()
        if not isinstance(selected_device, int):
            return

        progress = QProgressDialog("Initializing calibration...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Calibrating...")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        worker = CalibrationWorker(self.audio_app, input_device_id=selected_device)

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
            worker.deleteLater()

        worker.progress_signal.connect(_update_progress)
        worker.finished_signal.connect(_finish)
        worker.start()

    def closeEvent(self, event):
        event.ignore()
        self.hide()

def launch_gui(audio_app):
    app = QApplication(sys.argv)
    window = UpmixerWindow(audio_app)
    window.show()
    return app.exec()
