package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

type Config struct {
	APIKey string `json:"api_key"`
}

type APIResponse struct {
	Text  string `json:"text"`
	Error string `json:"error"`
}

var (
	config     Config
	configPath string
	recording  bool
	audioFile  string
)

func main() {
	home, _ := os.UserHomeDir()
	configPath = filepath.Join(home, ".config", "voicetype", "config.json")
	loadConfig()

	if config.APIKey == "" {
		fmt.Println("No API key configured. Set it in:", configPath)
		fmt.Println("Format: {\"api_key\": \"YOUR_KEY\"}")
		os.MkdirAll(filepath.Dir(configPath), 0755)
		os.WriteFile(configPath, []byte(`{"api_key": ""}`), 0600)
	}

	fmt.Println("VoiceType started. Press Ctrl+Super+H to record.")
	fmt.Println("Press Ctrl+C to exit.")

	go startHotkeyListener()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	fmt.Println("\nExiting...")
}

func loadConfig() {
	// Env var takes priority
	if key := os.Getenv("TEXT_GENERATOR_API_KEY"); key != "" {
		config.APIKey = key
		return
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		return
	}
	json.Unmarshal(data, &config)
}

func startHotkeyListener() {
	listenForHotkey()
}

func listenForHotkey() {
	runXbindkeysListener()
}

func runXbindkeysListener() {
	tmpDir := os.TempDir()
	fifoPath := filepath.Join(tmpDir, "voicetype-fifo")
	os.Remove(fifoPath)
	syscall.Mkfifo(fifoPath, 0600)

	xbindkeysrc := filepath.Join(tmpDir, "voicetype-xbindkeysrc")
	rcContent := fmt.Sprintf(`"echo trigger > %s"
    Control+Mod4 + h
`, fifoPath)
	os.WriteFile(xbindkeysrc, []byte(rcContent), 0600)

	// Kill any existing xbindkeys for this config
	exec.Command("pkill", "-f", "xbindkeys.*voicetype").Run()

	cmd := exec.Command("xbindkeys", "-n", "-f", xbindkeysrc)
	cmd.Start()

	go func() {
		for {
			data, err := os.ReadFile(fifoPath)
			if err != nil {
				time.Sleep(100 * time.Millisecond)
				continue
			}
			if len(data) > 0 {
				os.Truncate(fifoPath, 0)
				handleHotkey()
			}
			time.Sleep(50 * time.Millisecond)
		}
	}()

	// Fallback: also try with direct approach
	go directHotkeyPoll()

	select {}
}

func directHotkeyPoll() {
	// Use dbus or direct X11 - for now use a named pipe with a helper
	home, _ := os.UserHomeDir()
	triggerFile := filepath.Join(home, ".cache", "voicetype-trigger")
	os.MkdirAll(filepath.Dir(triggerFile), 0755)

	for {
		if _, err := os.Stat(triggerFile); err == nil {
			os.Remove(triggerFile)
			handleHotkey()
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func handleHotkey() {
	if recording {
		stopRecording()
	} else {
		startRecording()
	}
}

func getActiveWindow() string {
	out, err := exec.Command("xdotool", "getactivewindow").Output()
	if err != nil {
		return ""
	}
	return string(bytes.TrimSpace(out))
}

func startRecording() {
	recording = true
	audioFile = filepath.Join(os.TempDir(), fmt.Sprintf("voicetype-%d.wav", time.Now().UnixNano()))

	// Store the active window before we show visualizer
	activeWin := getActiveWindow()
	os.WriteFile(filepath.Join(os.TempDir(), "voicetype-window"), []byte(activeWin), 0600)

	// Start recording with visualizer
	go showVisualizer()

	cmd := exec.Command("parecord", "--file-format=wav", "--channels=1", "--rate=16000", audioFile)
	cmd.Start()

	// Watch for visualizer close (Enter/ESC pressed)
	go func() {
		for recording {
			time.Sleep(100 * time.Millisecond)
			if _, err := os.Stat("/tmp/voicetype-visualizer-run"); os.IsNotExist(err) {
				if recording {
					stopRecording()
				}
				return
			}
		}
	}()

	fmt.Println("Recording started... Press hotkey again to stop.")
}

func stopRecording() {
	recording = false

	// Stop parecord
	exec.Command("pkill", "-f", "parecord.*voicetype").Run()

	// Hide visualizer
	hideVisualizer()

	// Check if cancelled with ESC
	if wasCancelled() {
		fmt.Println("Recording cancelled")
		os.Remove(audioFile)
		return
	}

	fmt.Println("Recording stopped. Transcribing...")

	// Show spinner
	showSpinner()

	// Transcribe
	text := transcribeAudio(audioFile)

	// Hide spinner
	hideSpinner()

	if text != "" {
		// Type into original window
		winData, _ := os.ReadFile(filepath.Join(os.TempDir(), "voicetype-window"))
		windowID := string(bytes.TrimSpace(winData))
		typeText(windowID, text)
	}

	os.Remove(audioFile)
}

func showVisualizer() {
	// Find visualizer.py next to binary or in common locations
	locations := []string{
		filepath.Join(filepath.Dir(os.Args[0]), "visualizer.py"),
		"/usr/local/share/voicetype/visualizer.py",
		filepath.Join(os.Getenv("HOME"), ".local/share/voicetype/visualizer.py"),
	}

	var scriptPath string
	for _, loc := range locations {
		if _, err := os.Stat(loc); err == nil {
			scriptPath = loc
			break
		}
	}

	if scriptPath == "" {
		fmt.Println("Warning: visualizer.py not found")
		return
	}

	cmd := exec.Command("/usr/bin/python3", scriptPath)
	cmd.Start()
}

func hideVisualizer() {
	os.Remove("/tmp/voicetype-visualizer-run")
	exec.Command("pkill", "-f", "visualizer.py").Run()
}

func showSpinner() {
	locations := []string{
		filepath.Join(filepath.Dir(os.Args[0]), "spinner.py"),
		"/usr/local/share/voicetype/spinner.py",
		filepath.Join(os.Getenv("HOME"), ".local/share/voicetype/spinner.py"),
	}

	var scriptPath string
	for _, loc := range locations {
		if _, err := os.Stat(loc); err == nil {
			scriptPath = loc
			break
		}
	}

	if scriptPath == "" {
		return
	}

	cmd := exec.Command("/usr/bin/python3", scriptPath)
	cmd.Start()
	time.Sleep(100 * time.Millisecond)
}

func hideSpinner() {
	os.Remove("/tmp/voicetype-spinner-run")
	exec.Command("pkill", "-f", "spinner.py").Run()
}

func wasCancelled() bool {
	if _, err := os.Stat("/tmp/voicetype-cancelled"); err == nil {
		os.Remove("/tmp/voicetype-cancelled")
		return true
	}
	return false
}

func transcribeAudio(audioPath string) string {
	// Try API first, fall back to local
	text := transcribeAPI(audioPath)
	if text != "" {
		return text
	}

	fmt.Println("API failed, trying local Parakeet...")
	return transcribeLocal(audioPath)
}

func transcribeAPI(audioPath string) string {
	if config.APIKey == "" {
		return ""
	}

	// Compress to Opus for minimal upload size
	compressedPath := audioPath + ".ogg"
	cmd := exec.Command("ffmpeg", "-i", audioPath, "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", "-application", "voip", compressedPath, "-y")
	cmd.Stderr = nil
	if err := cmd.Run(); err != nil {
		fmt.Println("ffmpeg opus error, trying mp3")
		compressedPath = audioPath + ".mp3"
		cmd = exec.Command("ffmpeg", "-i", audioPath, "-ac", "1", "-ar", "16000", "-b:a", "32k", compressedPath, "-y")
		if err := cmd.Run(); err != nil {
			compressedPath = audioPath
		}
	}
	if compressedPath != audioPath {
		defer os.Remove(compressedPath)
	}

	file, err := os.Open(compressedPath)
	if err != nil {
		return ""
	}
	defer file.Close()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, _ := writer.CreateFormFile("audio_file", filepath.Base(compressedPath))
	io.Copy(part, file)
	writer.WriteField("translate_to_english", "false")
	writer.Close()

	req, _ := http.NewRequest("POST", "https://api.text-generator.io/api/v1/audio-file-extraction", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("secret", config.APIKey)

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("API request error:", err)
		return ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		fmt.Println("API HTTP error:", resp.StatusCode)
		return ""
	}

	body, _ := io.ReadAll(resp.Body)
	fmt.Println("API response:", string(body))

	var result APIResponse
	json.Unmarshal(body, &result)

	if result.Error != "" || result.Text == "" {
		return ""
	}

	fmt.Println("Transcribed:", result.Text)
	return result.Text
}

func transcribeLocal(audioPath string) string {
	locations := []string{
		filepath.Join(filepath.Dir(os.Args[0]), "fallback_asr.py"),
		"/usr/local/share/voicetype/fallback_asr.py",
		filepath.Join(os.Getenv("HOME"), ".local/share/voicetype/fallback_asr.py"),
	}

	var scriptPath string
	for _, loc := range locations {
		if _, err := os.Stat(loc); err == nil {
			scriptPath = loc
			break
		}
	}

	if scriptPath == "" {
		fmt.Println("Local ASR not available")
		return ""
	}

	cmd := exec.Command("/usr/bin/python3", scriptPath, audioPath)
	out, err := cmd.Output()
	if err != nil {
		fmt.Println("Local ASR error:", err)
		return ""
	}

	text := string(bytes.TrimSpace(out))
	fmt.Println("Local transcribed:", text)
	return text
}

func typeText(windowID, text string) {
	if windowID != "" {
		exec.Command("xdotool", "windowactivate", "--sync", windowID).Run()
		time.Sleep(100 * time.Millisecond)
	}

	// Type the text
	exec.Command("xdotool", "type", "--clearmodifiers", "--", text).Run()
	fmt.Println("Typed text into window")
}
