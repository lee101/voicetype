# VoiceType

Global hotkey voice-to-text for Linux. Press `Ctrl+Super+H` to record, press again to transcribe and type into the focused window.

## Install

```bash
git clone https://github.com/lee101/voicetype && cd voicetype && ./install.sh
```

## Setup

Set your [text-generator.io](https://text-generator.io) API key:

```bash
echo 'TEXT_GENERATOR_API_KEY=your_key_here' > ~/.config/voicetype/env
```

## Usage

- `Ctrl+Super+H` - Start/stop recording
- Shows visualizer while recording
- Transcribed text is typed into the previously focused window

## Uninstall

```bash
./uninstall.sh
```
