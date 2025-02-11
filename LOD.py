#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ErrorNoName LunarWave Update V2 BETA
"""

import os
import sys
import subprocess
import time
import numpy as np
import threading
import sounddevice as sd
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPainter, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QSlider
)

# Masquer certains messages ALSA
os.environ["ALSA_LOGLEVEL"] = "quiet"
# Sous Wayland, décommentez la ligne suivante si nécessaire :
# os.environ["QT_QPA_PLATFORM"] = "wayland"

def run_cmd(cmd_list):
    """Exécute une commande en masquant sa sortie."""
    try:
        subprocess.run(cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur d'exécution de {' '.join(cmd_list)} : {e}")

def get_pulseaudio_source(description):
    """
    Recherche dans la liste des sources PulseAudio celle dont la description
    contient la chaîne 'description'. Retourne le nom de la source ou None.
    """
    try:
        output = subprocess.check_output(["pactl", "list", "sources"],
                                           stderr=subprocess.DEVNULL).decode()
    except Exception as e:
        print("Erreur lors de la récupération des sources PulseAudio:", e)
        return None
    current_source = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Source #"):
            current_source = {}
        elif line.startswith("Name:"):
            current_source["Name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Description:"):
            current_source["Description"] = line.split(":", 1)[1].strip()
            if description.lower() in current_source["Description"].lower():
                return current_source["Name"]
    return None

def pitch_shift(signal, factor):
    """
    Applique un simple pitch shift au signal mono en rééchantillonnant par interpolation.
    Le paramètre 'factor' > 1 augmente le pitch, < 1 le diminue.
    """
    n = len(signal)
    original_indices = np.arange(n)
    new_indices = original_indices / factor
    shifted = np.interp(original_indices, new_indices, signal, left=0, right=0)
    return shifted

# --- Thread de capture audio pour la prévisualisation (sans traitement) ---
class AudioPreviewThread(QThread):
    volume_signal = pyqtSignal(float)
    
    def __init__(self, device_index, sample_rate=44100, parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.running = True

    def callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        level = np.sqrt(np.mean(indata**2))
        self.volume_signal.emit(level)

    def run(self):
        try:
            with sd.InputStream(device=self.device_index,
                                samplerate=self.sample_rate,
                                channels=1,
                                callback=self.callback):
                while self.running:
                    self.msleep(50)
        except Exception as e:
            print("Erreur dans le stream audio (prévisualisation):", e)

    def stop(self):
        self.running = False
        self.wait()

# --- Thread de traitement audio pour le micro virtuel ---
class ProcessingThread(QThread):
    # Ce thread capte l'audio du micro physique, y applique un effet audio ET un effet de voix,
    # puis envoie le signal vers la sortie (null-sink).
    def __init__(self, in_device, out_device, samplerate, effect="None", gain=1.0, voice_effect="None", parent=None):
        super().__init__(parent)
        self.in_device = in_device
        self.out_device = out_device
        self.samplerate = samplerate
        self.effect = effect
        self.gain = gain
        self.voice_effect = voice_effect
        self.running = True
        self.channels_in = 1
        self.channels_out = 2
        # Pour l'effet Echo
        self.echo_buffer = None
        self.echo_index = 0
        # Pour l'effet Ultra BoostBass
        self.bass_initialized = False

    def callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)
        # Signal de base après gain
        signal = self.gain * indata[:, 0]
        # Appliquer l'effet audio sélectionné
        if self.effect == "None":
            processed = signal
        elif self.effect == "Pan Left":
            processed = signal  # Puis nous gérons la duplication sur canaux plus bas
        elif self.effect == "Stutter":
            t = time.time()
            processed = signal if (t % 0.6) >= 0.1 else np.zeros_like(signal)
        elif self.effect == "Rapid Stutter":
            t = time.time()
            processed = signal if (t % 0.2) >= 0.05 else np.zeros_like(signal)
        elif self.effect == "Ultra Loud":
            ultra_signal = 20 * signal
            processed = np.tanh(ultra_signal)
        elif self.effect == "Ultra Loud Raw":
            processed = 50 * signal
        elif self.effect == "Bit Crusher":
            quant_levels = 16
            processed = np.round(signal * quant_levels) / quant_levels
        elif self.effect == "Echo":
            if self.echo_buffer is None:
                delay_sec = 0.3
                self.delay_samples = int(self.samplerate * delay_sec)
                self.echo_buffer = np.zeros(self.delay_samples)
                self.echo_index = 0
            feedback = 0.5
            processed = np.empty_like(signal)
            for i in range(len(signal)):
                echo_sample = self.echo_buffer[self.echo_index]
                processed[i] = signal[i] + feedback * echo_sample
                self.echo_buffer[self.echo_index] = signal[i]
                self.echo_index = (self.echo_index + 1) % self.delay_samples
        elif self.effect == "Ultra BoostBass":
            if not self.bass_initialized:
                fc = 200.0  # Fréquence de coupure (Hz) pour les basses
                self.bass_alpha = np.exp(-2 * np.pi * fc / self.samplerate)
                self.bass_state = 0.0
                self.boost_factor = 20.0  # Facteur de boost pour les basses
                self.bass_initialized = True
            processed = np.empty_like(signal)
            for i in range(len(signal)):
                self.bass_state = (1 - self.bass_alpha) * signal[i] + self.bass_alpha * self.bass_state
                processed[i] = signal[i] + self.boost_factor * self.bass_state
        else:
            processed = signal

        # Réplication sur deux canaux (pour la sortie stéréo)
        stereo = np.column_stack((processed, processed))
        
        # Appliquer l'effet de voix (pitch shift) si demandé
        if self.voice_effect != "None":
            voice_effect_factors = {
                "Voice Femme": 1.2,
                "Voice Homme": 0.8,
                "Deep Voice": 0.7,
                "Belle Voix": 1.1
            }
            factor = voice_effect_factors.get(self.voice_effect, 1.0)
            # Appliquer le pitch shift sur chaque canal
            stereo[:,0] = pitch_shift(stereo[:,0], factor)
            stereo[:,1] = pitch_shift(stereo[:,1], factor)
        
        # Pour "Pan Left", forcer le canal droit à zéro
        if self.effect == "Pan Left":
            stereo[:,1] = 0

        outdata[:] = stereo

    def run(self):
        try:
            with sd.Stream(device=(self.in_device, self.out_device),
                           samplerate=self.samplerate,
                           channels=(self.channels_in, self.channels_out),
                           callback=self.callback):
                while self.running:
                    self.msleep(50)
        except Exception as e:
            print("Erreur dans le stream de traitement audio:", e)

    def stop(self):
        self.running = False
        self.wait()

# --- Widget de visualisation audio ---
class AudioVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0.0
        self.setMinimumHeight(50)

    def setLevel(self, level):
        self.level = level
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, Qt.black)
        width = int(rect.width() * self.level)
        painter.fillRect(0, 0, width, rect.height(), Qt.green)

# --- Fenêtre principale ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gestionnaire de Micro Virtuel")
        self.virtual_sink_module_id = None
        self.processing_thread = None
        self.preview_thread = None
        self.mic_info = {}
        self.voice_effect = "None"  # Valeur par défaut pour l'effet voix

        # Lock pour éviter les accès concurrents
        self.preview_lock = threading.Lock()

        # Interface graphique
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Sélection du micro physique (via sounddevice)
        mic_layout = QHBoxLayout()
        mic_label = QLabel("Microphone physique:")
        self.mic_combo = QComboBox()
        mic_layout.addWidget(mic_label)
        mic_layout.addWidget(self.mic_combo)
        layout.addLayout(mic_layout)
        self.populate_mic_list()
        self.mic_combo.currentIndexChanged.connect(self.on_mic_changed)

        # Nom du micro virtuel
        virt_layout = QHBoxLayout()
        virt_label = QLabel("Nom du micro virtuel:")
        self.virt_lineedit = QLineEdit("VirtualMic")
        virt_layout.addWidget(virt_label)
        virt_layout.addWidget(self.virt_lineedit)
        layout.addLayout(virt_layout)

        # Contrôle du volume (0 à 1000)
        vol_layout = QHBoxLayout()
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 1000)
        self.vol_slider.setValue(100)
        self.vol_slider.valueChanged.connect(self.on_volume_changed)
        vol_layout.addWidget(QLabel("Volume:"))
        vol_layout.addWidget(self.vol_slider)
        self.loud_button = QPushButton("Loud Mic")
        self.loud_button.clicked.connect(self.loud_mic)
        vol_layout.addWidget(self.loud_button)
        layout.addLayout(vol_layout)

        # Choix de l'effet audio
        effect_layout = QHBoxLayout()
        effect_label = QLabel("Effet audio:")
        self.effect_combo = QComboBox()
        self.effect_combo.addItems(["None", "Pan Left", "Stutter", "Rapid Stutter", "Ultra Loud", "Ultra Loud Raw", "Bit Crusher", "Echo", "Ultra BoostBass"])
        self.effect_combo.currentIndexChanged.connect(self.on_effect_changed)
        effect_layout.addWidget(effect_label)
        effect_layout.addWidget(self.effect_combo)
        layout.addLayout(effect_layout)

        # Choix de l'effet voix
        voice_layout = QHBoxLayout()
        voice_label = QLabel("Effet Voix:")
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["None", "Voice Femme", "Voice Homme", "Deep Voice", "Belle Voix"])
        self.voice_combo.currentIndexChanged.connect(self.on_voice_effect_changed)
        voice_layout.addWidget(voice_label)
        voice_layout.addWidget(self.voice_combo)
        layout.addLayout(voice_layout)

        # Boutons Start/Stop et Fix Audio
        btn_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setCheckable(True)
        self.start_button.toggled.connect(self.toggle_virtual_mic)
        btn_layout.addWidget(self.start_button)
        self.fix_button = QPushButton("Fix Audio")
        self.fix_button.clicked.connect(self.fix_audio)
        btn_layout.addWidget(self.fix_button)
        layout.addLayout(btn_layout)

        # Visualisation audio
        layout.addWidget(QLabel("Visualisation Micro Physique:"))
        self.vis_physical = AudioVisualizer()
        layout.addWidget(self.vis_physical)
        layout.addWidget(QLabel("Visualisation (virtuel):"))
        self.vis_virtual = AudioVisualizer()
        layout.addWidget(self.vis_virtual)

        # Timer pour copier la visualisation (du micro physique vers virtuel)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_virtual_vis)
        self.timer.start(50)

        # Démarrer la prévisualisation dès le lancement
        self.start_preview()

    def populate_mic_list(self):
        self.mic_info = {}
        try:
            devices = sd.query_devices()
        except Exception as e:
            print("Erreur lors de la requête des dispositifs audio:", e)
            return
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                self.mic_info[i] = dev
                self.mic_combo.addItem(dev.get("name", f"Device {i}"), i)

    def on_mic_changed(self):
        if not self.start_button.isChecked():
            self.start_preview()

    def start_preview(self):
        self.stop_preview()
        device_index = self.mic_combo.currentData()
        if device_index is None:
            return
        dev_info = self.mic_info.get(device_index, {})
        sample_rate = int(dev_info.get("default_samplerate", 44100))
        with self.preview_lock:
            self.preview_thread = AudioPreviewThread(device_index, sample_rate)
            self.preview_thread.volume_signal.connect(self.vis_physical.setLevel)
            self.preview_thread.start()

    def stop_preview(self):
        with self.preview_lock:
            if self.preview_thread is not None:
                try:
                    self.preview_thread.stop()
                except Exception as e:
                    print("Erreur lors de l'arrêt de la prévisualisation:", e)
                self.preview_thread = None

    def on_volume_changed(self, value):
        if self.processing_thread is not None:
            self.processing_thread.gain = value / 100.0
        if self.virtual_sink_module_id is not None:
            target = "virt_sink.monitor"
        else:
            desc = self.mic_combo.currentText()
            target = get_pulseaudio_source(desc)
            if target is None:
                target = desc
        run_cmd(["pactl", "set-source-volume", target, f"{value}%"])

    def loud_mic(self):
        self.vol_slider.setValue(1000)
        self.on_volume_changed(1000)

    def on_effect_changed(self, index):
        effect = self.effect_combo.currentText()
        if self.processing_thread is not None:
            self.processing_thread.effect = effect

    def on_voice_effect_changed(self, index):
        self.voice_effect = self.voice_combo.currentText()
        if self.processing_thread is not None:
            self.processing_thread.voice_effect = self.voice_effect

    def toggle_virtual_mic(self, checked):
        if checked:
            self.start_button.setText("Stop")
            self.start_virtual_mic()
        else:
            self.start_button.setText("Start")
            self.stop_virtual_mic()

    def start_virtual_mic(self):
        self.stop_preview()
        virtual_name = self.virt_lineedit.text()
        existing_source = get_pulseaudio_source(virtual_name)
        if existing_source is None:
            try:
                output = subprocess.check_output(
                    ["pactl", "load-module", "module-null-sink",
                     "sink_name=virt_sink",
                     f"sink_properties=device.description={virtual_name}"],
                    stderr=subprocess.DEVNULL
                )
                self.virtual_sink_module_id = output.decode().strip()
                print("Micro virtuel créé.")
            except subprocess.CalledProcessError as e:
                print("Erreur lors de la création du micro virtuel:", e)
                return
        else:
            print("Micro virtuel déjà existant.")
            self.virtual_sink_module_id = "exist"
        time.sleep(1)
        out_device = None
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if (("virt_sink" in dev.get("name", "").lower()) or (virtual_name.lower() in dev.get("name", "").lower())) \
                   and dev.get("max_output_channels", 0) > 0:
                    out_device = i
                    break
        except Exception as e:
            print("Erreur lors de la recherche du dispositif de sortie virtuel:", e)
            return
        if out_device is None:
            print("Dispositif de sortie virtuel introuvable.")
            return
        physical_device_index = self.mic_combo.currentData()
        if physical_device_index is None:
            print("Aucun micro physique sélectionné.")
            return
        dev_info = self.mic_info.get(physical_device_index, {})
        sample_rate = int(dev_info.get("default_samplerate", 44100))
        gain = self.vol_slider.value() / 100.0
        effect = self.effect_combo.currentText()
        voice_effect = self.voice_effect  # Valeur actuellement sélectionnée
        self.processing_thread = ProcessingThread(physical_device_index, out_device, sample_rate,
                                                   effect=effect, gain=gain, voice_effect=voice_effect)
        self.processing_thread.start()

    def stop_virtual_mic(self):
        if self.processing_thread is not None:
            try:
                self.processing_thread.stop()
            except Exception as e:
                print("Erreur lors de l'arrêt du traitement audio:", e)
            self.processing_thread = None
        if self.virtual_sink_module_id:
            run_cmd(["pactl", "unload-module", self.virtual_sink_module_id])
            self.virtual_sink_module_id = None
        self.start_preview()

    def fix_audio(self):
        print("Tentative de réinitialisation des modules audio...")
        try:
            if get_pulseaudio_source("VirtualMic") is None:
                print("VirtualMic non trouvé, création du micro virtuel...")
                run_cmd(["pactl", "load-module", "module-null-sink",
                         "sink_name=virt_sink",
                         "sink_properties=device.description=VirtualMic"])
            output = subprocess.check_output(
                ["pactl", "list", "short", "modules"],
                stderr=subprocess.DEVNULL
            )
            for line in output.decode().splitlines():
                parts = line.split("\t")
                if len(parts) >= 3:
                    module_id = parts[0]
                    module_args = parts[2]
                    if "virt_sink" in module_args:
                        print(f"Déchargement du module {module_id} ({module_args})")
                        run_cmd(["pactl", "unload-module", module_id])
                        time.sleep(0.1)
        except Exception as e:
            print("Erreur lors de la réinitialisation des modules:", e)

    def update_virtual_vis(self):
        self.vis_virtual.setLevel(self.vis_physical.level)

    def closeEvent(self, event):
        self.stop_preview()
        if self.processing_thread is not None:
            self.processing_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    # Appliquer un thème sombre à l'application
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
