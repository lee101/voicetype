#!/bin/bash
echo "Uninstalling VoiceType..."

systemctl --user stop voicetype 2>/dev/null || true
systemctl --user disable voicetype 2>/dev/null || true

sudo rm -f /usr/local/bin/voicetype
rm -f ~/.config/systemd/user/voicetype.service
rm -f ~/.local/share/applications/voicetype.desktop

# Remove xbindkeys entry
sed -i '/voicetype-trigger/,+2d' ~/.xbindkeysrc 2>/dev/null || true
sed -i '/VoiceType hotkey/d' ~/.xbindkeysrc 2>/dev/null || true

pkill xbindkeys 2>/dev/null || true
xbindkeys &

echo "VoiceType uninstalled. Config preserved at ~/.config/voicetype/"
