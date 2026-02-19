#!/bin/bash
set -e

echo "Installing VoiceType..."

# Install dependencies
sudo apt-get update && sudo apt-get install -y xdotool xbindkeys pulseaudio-utils python3-gi gir1.2-gtk-3.0 golang-go

# Build
go build -o voicetype .

# Install to ~/.local/bin and ~/.local/share
mkdir -p ~/.local/bin ~/.local/share/voicetype
cp voicetype ~/.local/bin/
rm -f ~/.local/bin/*.py
cp *.py ~/.local/share/voicetype/

echo "Installed to ~/.local/bin/voicetype"

# Create config dir
mkdir -p ~/.config/voicetype

if [ ! -f ~/.config/voicetype/config.json ]; then
    echo '{"api_key": ""}' > ~/.config/voicetype/config.json
fi

if [ ! -f ~/.config/voicetype/env ]; then
    echo 'TEXT_GENERATOR_API_KEY=' > ~/.config/voicetype/env
    chmod 600 ~/.config/voicetype/env
    echo "Created env file at ~/.config/voicetype/env"
fi

# Create systemd user service
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/voicetype.service << 'EOF'
[Unit]
Description=VoiceType - Voice to Text
After=graphical-session.target

[Service]
Type=simple
ExecStart=%h/.local/bin/voicetype
Restart=on-failure
Environment=DISPLAY=:0
EnvironmentFile=-%h/.config/voicetype/env

[Install]
WantedBy=default.target
EOF

# Model server service (optional - for faster local fallback)
cat > ~/.config/systemd/user/voicetype-model.service << 'EOF'
[Unit]
Description=VoiceType Model Server
After=graphical-session.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 %h/.local/share/voicetype/model_server.py
Restart=on-failure
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=default.target
EOF

# Create desktop entry
mkdir -p ~/.local/share/applications
cat > ~/.local/share/applications/voicetype.desktop << 'EOF'
[Desktop Entry]
Name=VoiceType
Comment=Voice to text with global hotkey
Exec=voicetype
Terminal=false
Type=Application
Categories=Utility;
EOF

# Setup xbindkeys config addition
XBINDKEYS_RC="$HOME/.xbindkeysrc"
TRIGGER_FILE="$HOME/.cache/voicetype-trigger"

if ! grep -q "voicetype-trigger" "$XBINDKEYS_RC" 2>/dev/null; then
    cat >> "$XBINDKEYS_RC" << EOF

# VoiceType hotkey
"touch $TRIGGER_FILE"
    Control+Mod4 + h
EOF
    echo "Added hotkey to ~/.xbindkeysrc"
fi

# Reload xbindkeys
pkill xbindkeys 2>/dev/null || true
xbindkeys &

# Enable and start service
systemctl --user daemon-reload
systemctl --user enable voicetype
systemctl --user start voicetype

echo ""
echo "VoiceType installed!"
echo "Hotkey: Ctrl+Super+H"
echo ""
echo "Set your API key:"
echo "  echo 'TEXT_GENERATOR_API_KEY=your_key' > ~/.config/voicetype/env"
