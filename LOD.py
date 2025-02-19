#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ErrorNoName LunarWave Update V3 FULL
Version finale avec effets avancés, presets, contrôle dynamique, enregistrement, sound board enrichie,
configuration des raccourcis et nouveaux effets (Strident Talkie, Robotique 2.0) avec gestion des métadonnées.
"""

import os, sys, subprocess, time, threading, json, wave
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPainter, QPalette, QColor, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QSlider, QSplitter, QListWidget, QFileDialog,
    QDialog, QFormLayout, QKeySequenceEdit, QShortcut
)

# Masquer certains messages ALSA
os.environ["ALSA_LOGLEVEL"] = "quiet"
# Sous Wayland, décommentez si nécessaire:
# os.environ["QT_QPA_PLATFORM"] = "wayland"

# --- Gestion de configuration ---
class ConfigManager:
    def __init__(self, filename="config.json"):
        self.filename = filename
        self.config = {"shortcuts": {}, "imported_sounds": {}, "effect_params": {}}
        self.load()
        
    def load(self):
        try:
            with open(self.filename, "r") as f:
                self.config = json.load(f)
        except Exception:
            self.config = {"shortcuts": {}, "imported_sounds": {}, "effect_params": {}}
            
    def save(self):
        try:
            with open(self.filename, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print("Erreur lors de l'enregistrement de la config:", e)
            
    def get_shortcut(self, key):
        return self.config.get("shortcuts", {}).get(key, "")
    
    def set_shortcut(self, key, value):
        self.config.setdefault("shortcuts", {})[key] = value
        self.save()
        
    def add_imported_sound(self, name, path, duration):
        self.config.setdefault("imported_sounds", {})[name] = {"path": path, "duration": duration}
        self.save()
        
    def get_imported_sounds(self):
        return self.config.get("imported_sounds", {})

    def get_effect_params(self):
        return self.config.get("effect_params", {})

    def set_effect_param(self, key, value):
        self.config.setdefault("effect_params", {})[key] = value
        self.save()

config_manager = ConfigManager()

# --- Utilitaires ---
def pitch_shift(signal, factor):
    n = len(signal)
    orig_idx = np.arange(n)
    new_idx = orig_idx / factor
    return np.interp(orig_idx, new_idx, signal, left=0, right=0)

def run_cmd(cmd_list):
    try:
        subprocess.run(cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur d'exécution de {' '.join(cmd_list)} : {e}")

def get_pulseaudio_source(description):
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

def find_monitor_device(virtual_name):
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            name = dev.get("name", "").lower()
            if virtual_name.lower() in name and "monitor" in name and dev.get("max_input_channels", 0) > 0:
                return i
    except Exception as e:
        print("Erreur lors de la recherche du dispositif monitor:", e)
    return None

# --- Classes de l'interface et du traitement ---

# AudioVisualizer doit être défini avant son utilisation dans MainWindow
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
            with sd.InputStream(device=self.device_index, samplerate=self.sample_rate,
                                channels=1, callback=self.callback):
                while self.running:
                    self.msleep(50)
        except Exception as e:
            print("Erreur dans le stream audio (prévisualisation):", e)
            
    def stop(self):
        self.running = False
        self.wait()

class ProcessingThread(QThread):
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
        self.echo_buffer = None
        self.echo_index = 0
        self.bass_initialized = False
        
    def callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)
        signal = self.gain * indata[:, 0]
        if self.effect == "None":
            processed = signal
        elif self.effect == "Pan Left":
            processed = signal
        elif self.effect == "Stutter":
            processed = signal if (time.time() % 0.6) >= 0.1 else np.zeros_like(signal)
        elif self.effect == "Rapid Stutter":
            processed = signal if (time.time() % 0.2) >= 0.05 else np.zeros_like(signal)
        elif self.effect == "Ultra Loud":
            processed = np.tanh(20 * signal)
        elif self.effect == "Ultra Loud Raw":
            processed = 50 * signal
        elif self.effect == "Bit Crusher":
            processed = np.round(signal * 16) / 16
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
                fc = 200.0
                self.bass_alpha = np.exp(-2*np.pi*fc/self.samplerate)
                self.bass_state = 0.0
                self.boost_factor = 20.0
                self.bass_initialized = True
            processed = np.empty_like(signal)
            for i in range(len(signal)):
                self.bass_state = (1 - self.bass_alpha) * signal[i] + self.bass_alpha * self.bass_state
                processed[i] = signal[i] + self.boost_factor * self.bass_state
        else:
            processed = signal
            
        stereo = np.column_stack((processed, processed))
        if self.voice_effect != "None":
            factors = {"Voice Femme": 1.2, "Voice Homme": 0.8, "Deep Voice": 0.7, "Belle Voix": 1.1,
                       "Strident Talkie": 2.0, "Robotique 2.0": 0.5}
            factor = factors.get(self.voice_effect, 1.0)
            stereo[:,0] = pitch_shift(stereo[:,0], factor)
            stereo[:,1] = pitch_shift(stereo[:,1], factor)
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

class SoundBoardThread(QThread):
    def __init__(self, file_path, out_device, samplerate, effect="None", gain=1.0, voice_effect="None", parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.out_device = out_device
        self.samplerate = samplerate
        self.effect = effect
        self.gain = gain
        self.voice_effect = voice_effect
        self.running = True
        self.echo_buffer = None
        self.echo_index = 0
        self.bass_initialized = False
        
    def process_block(self, block):
        signal = self.gain * block
        if self.effect == "None":
            processed = signal
        elif self.effect == "Pan Left":
            processed = signal
        elif self.effect == "Stutter":
            processed = signal if (time.time() % 0.6) >= 0.1 else np.zeros_like(signal)
        elif self.effect == "Rapid Stutter":
            processed = signal if (time.time() % 0.2) >= 0.05 else np.zeros_like(signal)
        elif self.effect == "Ultra Loud":
            processed = np.tanh(20 * signal)
        elif self.effect == "Ultra Loud Raw":
            processed = 50 * signal
        elif self.effect == "Bit Crusher":
            processed = np.round(signal * 16) / 16
        elif self.effect == "Echo":
            if self.echo_buffer is None:
                delay_sec = config_manager.get_effect_params().get("echo_delay", 0.3)
                self.delay_samples = int(self.samplerate * delay_sec)
                self.echo_buffer = np.zeros(self.delay_samples)
                self.echo_index = 0
            feedback = config_manager.get_effect_params().get("echo_feedback", 0.5)
            processed = np.empty_like(signal)
            for i in range(len(signal)):
                echo_sample = self.echo_buffer[self.echo_index]
                processed[i] = signal[i] + feedback * echo_sample
                self.echo_buffer[self.echo_index] = signal[i]
                self.echo_index = (self.echo_index + 1) % self.delay_samples
        elif self.effect == "Ultra BoostBass":
            if not self.bass_initialized:
                fc = config_manager.get_effect_params().get("boostbass_cutoff", 200)
                self.bass_alpha = np.exp(-2*np.pi*fc/self.samplerate)
                self.bass_state = 0.0
                self.boost_factor = config_manager.get_effect_params().get("ultraloudraw_mult", 20)
                self.bass_initialized = True
            processed = np.empty_like(signal)
            for i in range(len(signal)):
                self.bass_state = (1 - self.bass_alpha) * signal[i] + self.bass_alpha * self.bass_state
                processed[i] = signal[i] + self.boost_factor * self.bass_state
        else:
            processed = signal
        stereo = np.column_stack((processed, processed))
        if self.voice_effect != "None":
            factors = {"Voice Femme": 1.2, "Voice Homme": 0.8, "Deep Voice": 0.7, "Belle Voix": 1.1,
                       "Strident Talkie": 2.0, "Robotique 2.0": 0.5}
            factor = factors.get(self.voice_effect, 1.0)
            stereo[:,0] = pitch_shift(stereo[:,0], factor)
            stereo[:,1] = pitch_shift(stereo[:,1], factor)
        if self.effect == "Pan Left":
            stereo[:,1] = 0
        return stereo
        
    def run(self):
        try:
            audio = AudioSegment.from_file(self.file_path)
            audio = audio.set_frame_rate(self.samplerate).set_channels(1)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            samples /= (2**15)
            block_size = 1024
            stream = sd.OutputStream(device=self.out_device, samplerate=self.samplerate, channels=2)
            stream.start()
            for start in range(0, len(samples), block_size):
                if not self.running:
                    break
                block = samples[start:start+block_size]
                if len(block) < block_size:
                    block = np.pad(block, (0, block_size - len(block)))
                processed = self.process_block(block)
                stream.write(processed)
            stream.stop()
            stream.close()
        except Exception as e:
            print("Erreur dans le thread SoundBoard:", e)
            
    def stop(self):
        self.running = False
        self.wait()

class VirtualMonitorThread(QThread):
    def __init__(self, monitor_device, output_device, samplerate, parent=None):
        super().__init__(parent)
        self.monitor_device = monitor_device
        self.output_device = output_device
        self.samplerate = samplerate
        self.running = True
        self.channels = 1

    def callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)
        outdata[:,0] = indata[:,0]
        outdata[:,1] = indata[:,0]

    def run(self):
        try:
            with sd.Stream(device=(self.monitor_device, self.output_device),
                           samplerate=self.samplerate,
                           channels=(1,2),
                           callback=self.callback):
                while self.running:
                    self.msleep(50)
        except Exception as e:
            print("Erreur dans le VirtualMonitorThread:", e)
            
    def stop(self):
        self.running = False
        self.wait()

class RecordingThread(QThread):
    def __init__(self, monitor_device, samplerate, parent=None):
        super().__init__(parent)
        self.monitor_device = monitor_device
        self.samplerate = samplerate
        self.running = True
        self.channels = 1
        
    def run(self):
        filename = f"recorded_{int(time.time())}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.samplerate)
        try:
            with sd.InputStream(device=self.monitor_device, samplerate=self.samplerate,
                                channels=1) as stream:
                while self.running:
                    data, _ = stream.read(1024)
                    data_int16 = np.int16(data * 32767)
                    wf.writeframes(data_int16.tobytes())
        except Exception as e:
            print("Erreur lors de l'enregistrement:", e)
        wf.close()
        print(f"Enregistrement terminé: {filename}")
        
    def stop(self):
        self.running = False
        self.wait()

class SoundBoardPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_file = None
        layout = QVBoxLayout(self)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Rechercher...")
        self.search_edit.textChanged.connect(self.filter_list)
        layout.addWidget(self.search_edit)
        self.list_widget = QListWidget()
        layout.addWidget(QLabel("Fichiers Audio Importés:"))
        layout.addWidget(self.list_widget)
        btn_layout = QHBoxLayout()
        self.add_button = QPushButton("Ajouter Audio")
        self.add_button.clicked.connect(self.add_audio)
        self.play_button = QPushButton("Lancer Audio")
        self.play_button.clicked.connect(self.play_audio)
        self.stop_button = QPushButton("Stop Audio")
        self.stop_button.clicked.connect(self.stop_audio)
        btn_layout.addWidget(self.add_button)
        btn_layout.addWidget(self.play_button)
        btn_layout.addWidget(self.stop_button)
        layout.addLayout(btn_layout)
        
    def filter_list(self, text):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(text.lower() not in item.text().lower())
        
    def add_audio(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Importer Audio", "", "Audio Files (*.mp3 *.ogg *.wav)")
        if files:
            for f in files:
                name = os.path.basename(f)
                self.list_widget.addItem(name)
                try:
                    audio = AudioSegment.from_file(f)
                    duration = round(audio.duration_seconds, 1)
                except Exception:
                    duration = 0
                config_manager.add_imported_sound(name, f, duration)
                
    def play_audio(self):
        item = self.list_widget.currentItem()
        if item:
            name = item.text()
            sounds = config_manager.get_imported_sounds()
            if name in sounds:
                file_path = sounds[name]["path"]
                self.parent().parent().start_sound_board(file_path)
                
    def stop_audio(self):
        self.parent().parent().stop_sound_board()

class ShortcutConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurer les raccourcis")
        self.setModal(True)
        layout = QFormLayout(self)
        
        self.loudMicEdit = QKeySequenceEdit()
        self.toggleVirtEdit = QKeySequenceEdit()
        
        self.loudMicEdit.setKeySequence(config_manager.get_shortcut("loud_mic"))
        self.toggleVirtEdit.setKeySequence(config_manager.get_shortcut("toggle_virtual_mic"))
        
        layout.addRow("Raccourci Loud Mic:", self.loudMicEdit)
        layout.addRow("Raccourci Toggle Virtual Mic:", self.toggleVirtEdit)
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Enregistrer")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Annuler")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(btn_layout)
        
    def get_shortcuts(self):
        return {
            "loud_mic": self.loudMicEdit.keySequence().toString(),
            "toggle_virtual_mic": self.toggleVirtEdit.keySequence().toString()
        }

class EffectParamsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paramètres d'effets")
        layout = QFormLayout(self)
        
        self.echoDelayEdit = QLineEdit()
        self.echoDelayEdit.setText(str(config_manager.get_effect_params().get("echo_delay", 0.3)))
        self.echoFeedbackEdit = QLineEdit()
        self.echoFeedbackEdit.setText(str(config_manager.get_effect_params().get("echo_feedback", 0.5)))
        self.boostBassCutoffEdit = QLineEdit()
        self.boostBassCutoffEdit.setText(str(config_manager.get_effect_params().get("boostbass_cutoff", 200)))
        self.ultraLoudRawMultEdit = QLineEdit()
        self.ultraLoudRawMultEdit.setText(str(config_manager.get_effect_params().get("ultraloudraw_mult", 50)))
        self.voicePitchEdit = QLineEdit()
        self.voicePitchEdit.setText(str(config_manager.get_effect_params().get("voice_pitch", 1.0)))
        
        layout.addRow("Echo Delay (sec):", self.echoDelayEdit)
        layout.addRow("Echo Feedback (0-1):", self.echoFeedbackEdit)
        layout.addRow("BoostBass Cutoff (Hz):", self.boostBassCutoffEdit)
        layout.addRow("Ultra Loud Raw Multiplier:", self.ultraLoudRawMultEdit)
        layout.addRow("Voice Pitch Factor:", self.voicePitchEdit)
        
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Enregistrer")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Annuler")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(btn_layout)
        
    def get_params(self):
        return {
            "echo_delay": float(self.echoDelayEdit.text()),
            "echo_feedback": float(self.echoFeedbackEdit.text()),
            "boostbass_cutoff": float(self.boostBassCutoffEdit.text()),
            "ultraloudraw_mult": float(self.ultraLoudRawMultEdit.text()),
            "voice_pitch": float(self.voicePitchEdit.text())
        }

class RecordingThread(QThread):
    def __init__(self, monitor_device, samplerate, parent=None):
        super().__init__(parent)
        self.monitor_device = monitor_device
        self.samplerate = samplerate
        self.running = True
        self.channels = 1
        
    def run(self):
        filename = f"recorded_{int(time.time())}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.samplerate)
        try:
            with sd.InputStream(device=self.monitor_device, samplerate=self.samplerate,
                                channels=1) as stream:
                while self.running:
                    data, _ = stream.read(1024)
                    data_int16 = np.int16(data * 32767)
                    wf.writeframes(data_int16.tobytes())
        except Exception as e:
            print("Erreur lors de l'enregistrement:", e)
        wf.close()
        print(f"Enregistrement terminé: {filename}")
        
    def stop(self):
        self.running = False
        self.wait()

class VirtualMonitorThread(QThread):
    def __init__(self, monitor_device, output_device, samplerate, parent=None):
        super().__init__(parent)
        self.monitor_device = monitor_device
        self.output_device = output_device
        self.samplerate = samplerate
        self.running = True
        self.channels = 1

    def callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)
        outdata[:,0] = indata[:,0]
        outdata[:,1] = indata[:,0]

    def run(self):
        try:
            with sd.Stream(device=(self.monitor_device, self.output_device),
                           samplerate=self.samplerate,
                           channels=(1,2),
                           callback=self.callback):
                while self.running:
                    self.msleep(50)
        except Exception as e:
            print("Erreur dans le VirtualMonitorThread:", e)
            
    def stop(self):
        self.running = False
        self.wait()

class SoundBoardThread(QThread):
    def __init__(self, file_path, out_device, samplerate, effect="None", gain=1.0, voice_effect="None", parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.out_device = out_device
        self.samplerate = samplerate
        self.effect = effect
        self.gain = gain
        self.voice_effect = voice_effect
        self.running = True
        self.echo_buffer = None
        self.echo_index = 0
        self.bass_initialized = False
        
    def process_block(self, block):
        signal = self.gain * block
        if self.effect == "None":
            processed = signal
        elif self.effect == "Pan Left":
            processed = signal
        elif self.effect == "Stutter":
            processed = signal if (time.time() % 0.6) >= 0.1 else np.zeros_like(signal)
        elif self.effect == "Rapid Stutter":
            processed = signal if (time.time() % 0.2) >= 0.05 else np.zeros_like(signal)
        elif self.effect == "Ultra Loud":
            processed = np.tanh(20 * signal)
        elif self.effect == "Ultra Loud Raw":
            processed = 50 * signal
        elif self.effect == "Bit Crusher":
            processed = np.round(signal * 16) / 16
        elif self.effect == "Echo":
            if self.echo_buffer is None:
                delay_sec = config_manager.get_effect_params().get("echo_delay", 0.3)
                self.delay_samples = int(self.samplerate * delay_sec)
                self.echo_buffer = np.zeros(self.delay_samples)
                self.echo_index = 0
            feedback = config_manager.get_effect_params().get("echo_feedback", 0.5)
            processed = np.empty_like(signal)
            for i in range(len(signal)):
                echo_sample = self.echo_buffer[self.echo_index]
                processed[i] = signal[i] + feedback * echo_sample
                self.echo_buffer[self.echo_index] = signal[i]
                self.echo_index = (self.echo_index + 1) % self.delay_samples
        elif self.effect == "Ultra BoostBass":
            if not self.bass_initialized:
                fc = config_manager.get_effect_params().get("boostbass_cutoff", 200)
                self.bass_alpha = np.exp(-2*np.pi*fc/self.samplerate)
                self.bass_state = 0.0
                self.boost_factor = config_manager.get_effect_params().get("ultraloudraw_mult", 20)
                self.bass_initialized = True
            processed = np.empty_like(signal)
            for i in range(len(signal)):
                self.bass_state = (1 - self.bass_alpha) * signal[i] + self.bass_alpha * self.bass_state
                processed[i] = signal[i] + self.boost_factor * self.bass_state
        else:
            processed = signal
        stereo = np.column_stack((processed, processed))
        if self.voice_effect != "None":
            factors = {"Voice Femme": 1.2, "Voice Homme": 0.8, "Deep Voice": 0.7, "Belle Voix": 1.1,
                       "Strident Talkie": 2.0, "Robotique 2.0": 0.5}
            factor = factors.get(self.voice_effect, 1.0)
            stereo[:,0] = pitch_shift(stereo[:,0], factor)
            stereo[:,1] = pitch_shift(stereo[:,1], factor)
        if self.effect == "Pan Left":
            stereo[:,1] = 0
        return stereo
        
    def run(self):
        try:
            audio = AudioSegment.from_file(self.file_path)
            audio = audio.set_frame_rate(self.samplerate).set_channels(1)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            samples /= (2**15)
            block_size = 1024
            stream = sd.OutputStream(device=self.out_device, samplerate=self.samplerate, channels=2)
            stream.start()
            for start in range(0, len(samples), block_size):
                if not self.running:
                    break
                block = samples[start:start+block_size]
                if len(block) < block_size:
                    block = np.pad(block, (0, block_size - len(block)))
                processed = self.process_block(block)
                stream.write(processed)
            stream.stop()
            stream.close()
        except Exception as e:
            print("Erreur dans le thread SoundBoard:", e)
            
    def stop(self):
        self.running = False
        self.wait()

class VirtualMonitorThread(QThread):
    def __init__(self, monitor_device, output_device, samplerate, parent=None):
        super().__init__(parent)
        self.monitor_device = monitor_device
        self.output_device = output_device
        self.samplerate = samplerate
        self.running = True
        self.channels = 1

    def callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)
        outdata[:,0] = indata[:,0]
        outdata[:,1] = indata[:,0]

    def run(self):
        try:
            with sd.Stream(device=(self.monitor_device, self.output_device),
                           samplerate=self.samplerate,
                           channels=(1,2),
                           callback=self.callback):
                while self.running:
                    self.msleep(50)
        except Exception as e:
            print("Erreur dans le VirtualMonitorThread:", e)
            
    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gestionnaire de Micro Virtuel Avancé")
        self.virtual_sink_module_id = None
        self.processing_thread = None
        self.preview_thread = None
        self.soundboard_thread = None
        self.monitor_thread = None
        self.recording_thread = None
        self.mic_info = {}
        self.voice_effect = "None"
        self.preview_lock = threading.Lock()
        
        self.shortcut_loud = None
        self.shortcut_toggle_virt = None
        
        splitter = QSplitter(Qt.Horizontal)
        # Panel gauche : Contrôles, presets, monitoring, enregistrement
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        mic_layout = QHBoxLayout()
        mic_label = QLabel("Microphone physique:")
        self.mic_combo = QComboBox()
        mic_layout.addWidget(mic_label)
        mic_layout.addWidget(self.mic_combo)
        left_layout.addLayout(mic_layout)
        self.populate_mic_list()
        self.mic_combo.currentIndexChanged.connect(self.on_mic_changed)
        
        virt_layout = QHBoxLayout()
        virt_label = QLabel("Nom du micro virtuel:")
        self.virt_lineedit = QLineEdit("VirtualMic")
        virt_layout.addWidget(virt_label)
        virt_layout.addWidget(self.virt_lineedit)
        left_layout.addLayout(virt_layout)
        
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
        left_layout.addLayout(vol_layout)
        
        effect_layout = QHBoxLayout()
        effect_label = QLabel("Effet audio:")
        self.effect_combo = QComboBox()
        self.effect_combo.addItems(["None", "Pan Left", "Stutter", "Rapid Stutter", "Ultra Loud", "Ultra Loud Raw", "Bit Crusher", "Echo", "Ultra BoostBass"])
        self.effect_combo.currentIndexChanged.connect(self.on_effect_changed)
        effect_layout.addWidget(effect_label)
        effect_layout.addWidget(self.effect_combo)
        left_layout.addLayout(effect_layout)
        
        voice_layout = QHBoxLayout()
        voice_label = QLabel("Effet Voix:")
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["None", "Voice Femme", "Voice Homme", "Deep Voice", "Belle Voix", "Strident Talkie", "Robotique 2.0"])
        self.voice_combo.currentIndexChanged.connect(self.on_voice_effect_changed)
        voice_layout.addWidget(voice_label)
        voice_layout.addWidget(self.voice_combo)
        left_layout.addLayout(voice_layout)
        
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Presets:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["-- Sélectionner --", "Mode Talkie", "Mode Robot", "Mode Concert"])
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        left_layout.addLayout(preset_layout)
        
        btn_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setCheckable(True)
        self.start_button.toggled.connect(self.toggle_virtual_mic)
        btn_layout.addWidget(self.start_button)
        self.fix_button = QPushButton("Fix Audio")
        self.fix_button.clicked.connect(self.fix_audio)
        btn_layout.addWidget(self.fix_button)
        self.shortcut_button = QPushButton("Configurer raccourcis")
        self.shortcut_button.clicked.connect(self.configure_shortcuts)
        btn_layout.addWidget(self.shortcut_button)
        self.param_button = QPushButton("Paramètres Effets")
        self.param_button.clicked.connect(self.configure_effect_params)
        btn_layout.addWidget(self.param_button)
        self.record_button = QPushButton("Record Output")
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self.toggle_record)
        btn_layout.addWidget(self.record_button)
        left_layout.addLayout(btn_layout)
        
        self.monitor_button = QPushButton("Monitor Virtual Output (OFF)")
        self.monitor_button.setCheckable(True)
        self.monitor_button.toggled.connect(self.toggle_monitor)
        left_layout.addWidget(self.monitor_button)
        
        left_layout.addWidget(QLabel("Visualisation Micro Physique:"))
        self.vis_physical = AudioVisualizer()
        left_layout.addWidget(self.vis_physical)
        left_layout.addWidget(QLabel("Visualisation (virtuel):"))
        self.vis_virtual = AudioVisualizer()
        left_layout.addWidget(self.vis_virtual)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_virtual_vis)
        self.timer.start(50)
        self.start_preview()
        splitter.addWidget(left_panel)
        
        # Panel droit : Sound Board
        right_panel = SoundBoardPanel()
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 300])
        self.setCentralWidget(splitter)
        
        self.load_shortcuts()
        
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
                if (("virt_sink" in dev.get("name", "").lower()) or (virtual_name.lower() in dev.get("name", "").lower())) and dev.get("max_output_channels", 0) > 0:
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
        voice_effect = self.voice_combo.currentText()
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
        
    def start_sound_board(self, file_path):
        self.stop_virtual_mic()
        virtual_name = self.virt_lineedit.text()
        out_device = None
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if (("virt_sink" in dev.get("name", "").lower()) or (virtual_name.lower() in dev.get("name", "").lower())) and dev.get("max_output_channels", 0) > 0:
                    out_device = i
                    break
        except Exception as e:
            print("Erreur lors de la recherche du dispositif de sortie virtuel:", e)
            return
        if out_device is None:
            print("Dispositif de sortie virtuel introuvable.")
            return
        physical_device_index = self.mic_combo.currentData()
        dev_info = self.mic_info.get(physical_device_index, {})
        sample_rate = int(dev_info.get("default_samplerate", 44100))
        gain = self.vol_slider.value() / 100.0
        effect = self.effect_combo.currentText()
        voice_effect = self.voice_combo.currentText()
        self.soundboard_thread = SoundBoardThread(file_path, out_device, sample_rate,
                                                   effect=effect, gain=gain, voice_effect=voice_effect)
        self.soundboard_thread.start()
        self.soundboard_thread.finished.connect(self.start_virtual_mic)
        
    def stop_sound_board(self):
        if self.soundboard_thread is not None:
            try:
                self.soundboard_thread.stop()
            except Exception as e:
                print("Erreur lors de l'arrêt de la Sound Board:", e)
            self.soundboard_thread = None
        
    def toggle_monitor(self, checked):
        if checked:
            self.monitor_button.setText("Monitor Virtual Output (ON)")
            self.start_monitor_virtual_output()
        else:
            self.monitor_button.setText("Monitor Virtual Output (OFF)")
            self.stop_monitor_virtual_output()
            
    def start_monitor_virtual_output(self):
        virtual_name = self.virt_lineedit.text()
        monitor_index = find_monitor_device(virtual_name)
        if monitor_index is None:
            print("Dispositif monitor du micro virtuel introuvable.")
            return
        default_output = sd.default.device[1]
        dev_info = sd.query_devices(default_output)
        samplerate = int(dev_info.get("default_samplerate", 44100))
        self.monitor_thread = VirtualMonitorThread(monitor_index, default_output, samplerate)
        self.monitor_thread.start()
        
    def stop_monitor_virtual_output(self):
        if self.monitor_thread is not None:
            try:
                self.monitor_thread.stop()
            except Exception as e:
                print("Erreur lors de l'arrêt du monitor virtual output:", e)
            self.monitor_thread = None
            
    def toggle_record(self, checked):
        if checked:
            self.record_button.setText("Stop Recording")
            virtual_name = self.virt_lineedit.text()
            monitor_index = find_monitor_device(virtual_name)
            if monitor_index is None:
                print("Dispositif monitor du micro virtuel introuvable pour l'enregistrement.")
                self.record_button.setChecked(False)
                return
            self.recording_thread = RecordingThread(monitor_index, 44100)
            self.recording_thread.start()
        else:
            self.record_button.setText("Record Output")
            if self.recording_thread is not None:
                self.recording_thread.stop()
                self.recording_thread = None
            
    def configure_shortcuts(self):
        dlg = ShortcutConfigDialog(self)
        if dlg.exec_():
            shortcuts = dlg.get_shortcuts()
            config_manager.set_shortcut("loud_mic", shortcuts["loud_mic"])
            config_manager.set_shortcut("toggle_virtual_mic", shortcuts["toggle_virtual_mic"])
            self.load_shortcuts()
            
    def load_shortcuts(self):
        if self.shortcut_loud:
            self.shortcut_loud.disconnect()
            self.shortcut_loud = None
        if self.shortcut_toggle_virt:
            self.shortcut_toggle_virt.disconnect()
            self.shortcut_toggle_virt = None
        loud = config_manager.get_shortcut("loud_mic")
        toggle = config_manager.get_shortcut("toggle_virtual_mic")
        if loud:
            self.shortcut_loud = QShortcut(QKeySequence(loud), self)
            self.shortcut_loud.activated.connect(self.loud_mic)
        if toggle:
            self.shortcut_toggle_virt = QShortcut(QKeySequence(toggle), self)
            self.shortcut_toggle_virt.activated.connect(self.on_toggle_virtual_mic_shortcut)
            
    def on_toggle_virtual_mic_shortcut(self):
        self.start_button.toggle()
        
    def configure_effect_params(self):
        dlg = EffectParamsDialog(self)
        if dlg.exec_():
            params = dlg.get_params()
            for key, value in params.items():
                config_manager.set_effect_param(key, value)
            print("Paramètres d'effets mis à jour.")
            
    def apply_preset(self, index):
        preset = self.preset_combo.currentText()
        if preset == "Mode Talkie":
            self.effect_combo.setCurrentText("Ultra Loud Raw")
            self.voice_combo.setCurrentText("Strident Talkie")
        elif preset == "Mode Robot":
            self.effect_combo.setCurrentText("Bit Crusher")
            self.voice_combo.setCurrentText("Robotique 2.0")
        elif preset == "Mode Concert":
            self.effect_combo.setCurrentText("Ultra Loud")
            self.voice_combo.setCurrentText("None")
            
    def update_virtual_vis(self):
        self.vis_virtual.setLevel(self.vis_physical.level)
        
    def fix_audio(self):
        print("Tentative de réinitialisation des modules audio...")
        try:
            if get_pulseaudio_source("VirtualMic") is None:
                print("VirtualMic non trouvé, création du micro virtuel...")
                run_cmd(["pactl", "load-module", "module-null-sink",
                         "sink_name=virt_sink",
                         "sink_properties=device.description=VirtualMic"])
            output = subprocess.check_output(["pactl", "list", "short", "modules"],
                                             stderr=subprocess.DEVNULL)
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
            
    def closeEvent(self, event):
        self.stop_preview()
        if self.processing_thread is not None:
            self.processing_thread.stop()
        if self.soundboard_thread is not None:
            self.soundboard_thread.stop()
        if self.monitor_thread is not None:
            self.monitor_thread.stop()
        if self.recording_thread is not None:
            self.recording_thread.stop()
        event.accept()

class VirtualMonitorThread(QThread):
    def __init__(self, monitor_device, output_device, samplerate, parent=None):
        super().__init__(parent)
        self.monitor_device = monitor_device
        self.output_device = output_device
        self.samplerate = samplerate
        self.running = True
        self.channels = 1

    def callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)
        outdata[:,0] = indata[:,0]
        outdata[:,1] = indata[:,0]

    def run(self):
        try:
            with sd.Stream(device=(self.monitor_device, self.output_device),
                           samplerate=self.samplerate,
                           channels=(1,2),
                           callback=self.callback):
                while self.running:
                    self.msleep(50)
        except Exception as e:
            print("Erreur dans le VirtualMonitorThread:", e)
            
    def stop(self):
        self.running = False
        self.wait()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53,53,53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25,25,25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53,53,53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53,53,53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42,130,218))
    dark_palette.setColor(QPalette.Highlight, QColor(42,130,218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    window = MainWindow()
    # Ajout d'un bouton pour changer de thème
    theme_button = QPushButton("Changer de thème")
    def toggle_theme():
        if app.palette().color(QPalette.Window) == QColor(53,53,53):
            light_palette = QPalette()
            light_palette.setColor(QPalette.Window, QColor(240,240,240))
            light_palette.setColor(QPalette.WindowText, Qt.black)
            light_palette.setColor(QPalette.Base, QColor(255,255,255))
            light_palette.setColor(QPalette.AlternateBase, QColor(233,231,227))
            light_palette.setColor(QPalette.ToolTipBase, Qt.black)
            light_palette.setColor(QPalette.ToolTipText, Qt.black)
            light_palette.setColor(QPalette.Text, Qt.black)
            light_palette.setColor(QPalette.Button, QColor(240,240,240))
            light_palette.setColor(QPalette.ButtonText, Qt.black)
            light_palette.setColor(QPalette.Link, QColor(0, 0, 255))
            app.setPalette(light_palette)
        else:
            app.setPalette(dark_palette)
    theme_button.clicked.connect(toggle_theme)
    # Placer le bouton dans la fenêtre principale
    window.layout().addWidget(theme_button) if window.layout() else None
    
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
