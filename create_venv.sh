#!/bin/bash

##############################
# Script: create_venv.sh
# Description: Crée un environnement virtuel Python avec une interface interactive et des fonctionnalités avancées.
# Author: Ezio/ErrorNoName
# Date: 2024-04-27
##############################

# Variables
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
SCRIPT_NAME="VenvManager"

# Fonction pour afficher le logo
display_logo() {
    clear
    figlet "$SCRIPT_NAME"
    echo "============================================"
    echo "   Gestionnaire d'Environnement Virtuel    "
    echo "============================================"
    echo ""
}

# Fonction pour afficher une barre de progression
show_progress() {
    local duration=$1
    (
        for i in $(seq 1 $duration); do
            echo -n "#"
            sleep 0.1
        done
    ) | pv -n -s $duration >/dev/null
}

# Fonction pour afficher les logs
log() {
    echo -e "\e[32m[INFO]\e[0m $1"
}

# Fonction pour afficher les erreurs
error() {
    echo -e "\e[31m[ERROR]\e[0m $1"
}

# Fonction pour afficher une animation de chargement
show_spinner() {
    local pid=$1
    local delay=0.1
    local spinner=( '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏' )
    local i=0

    while kill -0 "$pid" 2>/dev/null; do
        printf "\r%s Création de l'environnement virtuel... %s" "$(date +%H:%M:%S)" "${spinner[i]}"
        i=$(( (i + 1) % 10 ))
        sleep "$delay"
    done
    printf "\r%s Environnement virtuel créé avec succès!     \n" "$(date +%H:%M:%S)"
}

# Vérifier les dépendances
check_dependencies() {
    for cmd in python3 dialog figlet pv; do
        if ! command -v "$cmd" &>/dev/null; then
            error "Le paquet '$cmd' n'est pas installé. Installation en cours..."
            sudo pacman -S --noconfirm "$cmd"
            if [ $? -ne 0 ]; then
                error "Échec de l'installation de '$cmd'. Veuillez l'installer manuellement."
                exit 1
            fi
        fi
    done
}

# Afficher le logo
display_logo

# Vérifier les dépendances
check_dependencies

log "Vérification des prérequis terminée."

# Créer l'environnement virtuel
if [ -d "$VENV_DIR" ]; then
    dialog --msgbox "Un environnement virtuel existe déjà dans ce répertoire." 8 50
    log "Environnement virtuel déjà existant."
else
    log "Création de l'environnement virtuel..."
    python3 -m venv "$VENV_DIR" &
    VENV_PID=$!
    show_spinner "$VENV_PID"
    if [ ! -d "$VENV_DIR" ]; then
        error "La création de l'environnement virtuel a échoué."
        exit 1
    fi
fi

# Activer l'environnement virtuel
source "./$VENV_DIR/bin/activate"
log "Environnement virtuel activé."

# Menu pour installer les dépendances
CHOICE=$(dialog --clear --backtitle "$SCRIPT_NAME" \
    --title "Installer les dépendances" \
    --menu "Voulez-vous installer les dépendances ?" 15 50 3 \
    "1" "Installer depuis requirements.txt" \
    "2" "Installer manuellement" \
    "3" "Ne pas installer" \
    3>&1 1>&2 2>&3)

clear  # Nettoyer l'écran après 'dialog'

case $CHOICE in
    1)
        if [ -f "$REQUIREMENTS_FILE" ]; then
            log "Installation des dépendances depuis $REQUIREMENTS_FILE..."
            pip install --upgrade pip
            pip install -r "$REQUIREMENTS_FILE" &
            PIP_PID=$!
            show_spinner "$PIP_PID"
            log "Dépendances installées avec succès."
        else
            log "$REQUIREMENTS_FILE n'existe pas. Création du fichier..."
            pip freeze > "$REQUIREMENTS_FILE"
            log "$REQUIREMENTS_FILE créé."
            dialog --msgbox "$REQUIREMENTS_FILE a été créé. Vous pouvez ajouter vos dépendances et réexécuter le script." 8 50
        fi
        ;;
    2)
        echo "Entrez les packages à installer, séparés par des espaces :"
        read -r PACKAGES
        if [ -n "$PACKAGES" ]; then
            log "Installation des packages: $PACKAGES"
            pip install --upgrade pip
            pip install $PACKAGES &
            PIP_PID=$!
            show_spinner "$PIP_PID"
            log "Packages installés avec succès."
            # Mettre à jour requirements.txt
            pip freeze > "$REQUIREMENTS_FILE"
            log "$REQUIREMENTS_FILE mis à jour."
        else
            log "Aucun package spécifié."
        fi
        ;;
    3)
        log "Installation des dépendances ignorée."
        ;;
    *)
        log "Aucune action sélectionnée."
        ;;
esac

# Vérifier si requirements.txt existe, sinon le créer
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    log "Création de $REQUIREMENTS_FILE..."
    pip freeze > "$REQUIREMENTS_FILE"
    log "$REQUIREMENTS_FILE créé."
fi

# Message final
log "Processus terminé. Votre environnement virtuel est prêt à être utilisé."

# Garder le shell actif avec le venv activé
echo "L'environnement virtuel est activé. Vous pouvez commencer à travailler."
echo "Pour désactiver, tapez 'deactivate'."
