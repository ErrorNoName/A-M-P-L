# 🎤 LoudMic - Micro Virtuel avec Effets 🎵  

LoudMic est une application Qt5 permettant de rediriger l'audio d'un microphone physique vers un microphone virtuel en utilisant PulseAudio.  
Elle offre un contrôle avancé du volume (jusqu'à 700%) et la possibilité d'appliquer des effets audio en temps réel.  

## 🚀 Fonctionnalités  
✅ **Sélection du microphone physique** : Choisissez le micro que vous souhaitez rediriger.  
✅ **Microphone virtuel** : Création automatique d'un micro virtuel pour être utilisé dans d'autres applications.  
✅ **Contrôle du volume avancé** : Slider de volume (0-700%) et bouton **Loud Mic** pour amplification maximale.  
✅ **Effets audio** :  
   - 🎧 *None* : Aucun effet, le son est simplement redirigé.  
   - 🔊 *Pan Left* : Envoie l’audio uniquement sur le canal gauche.  
   - 🎵 *Stutter* : Coupe et remet le son très rapidement pour un effet de bégaiement.  
✅ **Visualisation du son en temps réel** : Affichage du niveau audio du micro physique et du micro virtuel.  
✅ **Bouton Fix Audio** : Réinitialisation des modules PulseAudio pour corriger d'éventuels problèmes.  

## 🖥️ Captures d'écran  
(Si vous souhaitez ajouter des captures d'écran, vous pouvez les héberger sur GitHub et les inclure ici.)

---

## 🛠️ Installation  

### 🔽 1. Prérequis  
- **Python 3.7+**  
- **PulseAudio** installé et actif sur votre système  
- **Qt5 (PyQt5)** pour l’interface graphique  

### 📥 2. Installation des dépendances  

#### Sous Linux (Arch, Debian, Ubuntu)  

```bash
sudo pacman -S pulseaudio pulseaudio-alsa  # Arch Linux
sudo apt install pulseaudio pulseaudio-utils  # Debian/Ubuntu
chmod +x create_venv.sh
./create_venv.sh
pip install -r requirements.txt
sudo pacman -S python-scipy

Pour Fix
pactl load-module module-null-sink sink_name=virt_sink sink_properties=device.description=VirtualMic
yay -S python-pydub
pip install build
sudo pacman -S python-scipy
