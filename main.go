package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type Config struct {
	APIKey string `json:"api_key"`
	FalKey string `json:"-"`
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

const chunkDir = "/tmp/voicetype-chunks"
const localModelSocket = "/tmp/voicetype-model.sock"

type ChunkResult struct {
	Index int
	Text  string
}

var (
	chunkResults   map[int]string
	chunkResultsMu sync.Mutex
	chunkWg        sync.WaitGroup
	vadCmd         *exec.Cmd
	localModelMu   sync.Mutex
	localModelReady bool
	localModelWarming bool
	instanceRecordingFlag string
	lastHotkeyTime time.Time
)

var nemoPythonPath = findNemoPython()

func findNemoPython() string {
	home, _ := os.UserHomeDir()
	venv := filepath.Join(home, "code", "20-questions", ".venv", "bin", "python3")
	if _, err := os.Stat(venv); err == nil {
		return venv
	}
	if p, err := exec.LookPath("python3"); err == nil {
		return p
	}
	return "/usr/bin/python3"
}

type localASRResponse struct {
	Status string `json:"status"`
	Text   string `json:"text"`
	Error  string `json:"error"`
}

func main() {
	home, _ := os.UserHomeDir()
	configPath = filepath.Join(home, ".config", "voicetype", "config.json")
	instanceRecordingFlag = filepath.Join(os.TempDir(), fmt.Sprintf("voicetype-recording-%d", os.Getpid()))
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
	if key := os.Getenv("TEXT_GENERATOR_API_KEY"); key != "" {
		config.APIKey = key
	}
	if key := os.Getenv("FAL_KEY"); key != "" {
		config.FalKey = key
	}

	data, err := os.ReadFile(configPath)
	if err == nil {
		json.Unmarshal(data, &config)
	}

	envFile := filepath.Join(filepath.Dir(configPath), "env")
	envData, err := os.ReadFile(envFile)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(envData), "\n") {
		line = strings.TrimSpace(line)
		if config.APIKey == "" && strings.HasPrefix(line, "TEXT_GENERATOR_API_KEY=") {
			config.APIKey = strings.Trim(strings.SplitN(line, "=", 2)[1], "\"' ")
		}
		if config.FalKey == "" && strings.HasPrefix(line, "FAL_KEY=") {
			config.FalKey = strings.Trim(strings.SplitN(line, "=", 2)[1], "\"' ")
		}
	}
}

func startHotkeyListener() {
	home, _ := os.UserHomeDir()
	triggerFile := filepath.Join(home, ".cache", "voicetype-trigger")
	os.MkdirAll(filepath.Dir(triggerFile), 0755)

	for {
		if _, err := os.Stat(triggerFile); err == nil {
			os.Remove(triggerFile)
			handleHotkey()
		}
		time.Sleep(80 * time.Millisecond)
	}
}

func handleHotkey() {
	now := time.Now()
	if now.Sub(lastHotkeyTime) < 500*time.Millisecond {
		return
	}
	lastHotkeyTime = now

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
	os.WriteFile(instanceRecordingFlag, []byte("1"), 0600)
	chunkResults = make(map[int]string)

	activeWin := getActiveWindow()
	os.WriteFile(filepath.Join(os.TempDir(), "voicetype-window"), []byte(activeWin), 0600)

	// Eagerly warm local ASR model in background.
	go preloadLocalModel()

	os.WriteFile("/tmp/voicetype-visualizer-run", []byte("1"), 0600)
	os.Remove("/tmp/voicetype-cancelled")
	go showVisualizer()

	// Start VAD chunker
	locations := []string{
		filepath.Join(filepath.Dir(os.Args[0]), "vad_chunker.py"),
		"/usr/local/share/voicetype/vad_chunker.py",
		filepath.Join(os.Getenv("HOME"), ".local/share/voicetype/vad_chunker.py"),
	}
	var scriptPath string
	for _, loc := range locations {
		if _, err := os.Stat(loc); err == nil {
			scriptPath = loc
			break
		}
	}

	if scriptPath != "" {
		vadCmd = exec.Command("/usr/bin/python3", scriptPath)
		vadCmd.Start()
		go watchChunks()
	} else {
		// Fallback to old recording method
		audioFile = filepath.Join(os.TempDir(), fmt.Sprintf("voicetype-%d.wav", time.Now().UnixNano()))
		cmd := exec.Command("parecord", "--file-format=wav", "--channels=1", "--rate=16000", audioFile)
		cmd.Start()
	}

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

	fmt.Println("Recording started...")
}

func stopRecording() {
	recording = false
	os.Remove(instanceRecordingFlag)

	hideVisualizer()

	if wasCancelled() {
		fmt.Println("Recording cancelled")
		if vadCmd != nil {
			vadCmd.Process.Kill()
			vadCmd = nil
		}
		exec.Command("pkill", "-f", "parecord.*voicetype").Run()
		exec.Command("pkill", "-f", "parec.*16000").Run()
		os.RemoveAll(chunkDir)
		os.Remove(audioFile)
		return
	}

	fmt.Println("Recording stopped. Transcribing...")
	showSpinner()

	var text string

	if vadCmd != nil {
		// Chunked mode - stop VAD chunker and wait for final chunk
		vadCmd.Process.Signal(syscall.SIGINT)
		vadCmd.Wait()
		vadCmd = nil

		// Wait for done signal
		for i := 0; i < 50; i++ {
			if _, err := os.Stat(filepath.Join(chunkDir, "done")); err == nil {
				break
			}
			time.Sleep(100 * time.Millisecond)
		}

		// Process any remaining chunks in ready file
		processReadyChunks()

		// Wait for all uploads
		chunkWg.Wait()

		// Combine results in order
		text = combineChunkResults()

		os.RemoveAll(chunkDir)
	} else {
		// Legacy single-file mode
		exec.Command("pkill", "-f", "parecord.*voicetype").Run()
		text = transcribeAudio(audioFile)
		os.Remove(audioFile)
	}

	hideSpinner()

	if text != "" {
		windowID := string(bytes.TrimSpace([]byte(getActiveWindow())))
		if windowID == "" {
			winData, _ := os.ReadFile(filepath.Join(os.TempDir(), "voicetype-window"))
			windowID = string(bytes.TrimSpace(winData))
		}
		typeText(windowID, text)
	}
}

func watchChunks() {
	readyPath := filepath.Join(chunkDir, "ready")
	var lastPos int64 = 0

	for recording {
		time.Sleep(200 * time.Millisecond)

		file, err := os.Open(readyPath)
		if err != nil {
			continue
		}

		file.Seek(lastPos, 0)
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if idx, err := strconv.Atoi(line); err == nil {
				chunkPath := filepath.Join(chunkDir, fmt.Sprintf("chunk-%d.ogg", idx))
				if _, err := os.Stat(chunkPath); err == nil {
					chunkWg.Add(1)
					go uploadChunk(chunkPath, idx)
				}
			}
		}

		pos, _ := file.Seek(0, 1)
		lastPos = pos
		file.Close()
	}
}

func processReadyChunks() {
	readyPath := filepath.Join(chunkDir, "ready")
	data, err := os.ReadFile(readyPath)
	if err != nil {
		return
	}

	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if idx, err := strconv.Atoi(line); err == nil {
			chunkResultsMu.Lock()
			_, exists := chunkResults[idx]
			chunkResultsMu.Unlock()

			if !exists {
				chunkPath := filepath.Join(chunkDir, fmt.Sprintf("chunk-%d.ogg", idx))
				if _, err := os.Stat(chunkPath); err == nil {
					chunkWg.Add(1)
					go uploadChunk(chunkPath, idx)
				}
			}
		}
	}
}

func uploadChunk(path string, index int) {
	defer chunkWg.Done()

	rawPath := strings.TrimSuffix(path, filepath.Ext(path)) + ".raw.ogg"
	text := transcribeChunk(path, rawPath)

	chunkResultsMu.Lock()
	chunkResults[index] = text
	chunkResultsMu.Unlock()

	fmt.Printf("Chunk %d: %s\n", index, truncate(text, 50))
}

func transcribeChunk(processedPath string, rawPath string) string {
	localText := transcribeLocal(processedPath)
	if localText == "" && rawPath != processedPath {
		if _, err := os.Stat(rawPath); err == nil {
			if t := transcribeLocal(rawPath); betterText(t, localText) {
				localText = t
			}
		}
	}
	if localText != "" {
		return localText
	}

	fmt.Println("Local ASR empty, racing API + fal...")
	result := make(chan string, 2)

	go func() {
		t := transcribeChunkAPI(processedPath)
		if t == "" && rawPath != processedPath {
			if _, err := os.Stat(rawPath); err == nil {
				if rt := transcribeChunkAPI(rawPath); betterText(rt, t) {
					t = rt
				}
			}
		}
		result <- t
	}()

	go func() {
		t := transcribeFal(processedPath)
		if t == "" && rawPath != processedPath {
			if _, err := os.Stat(rawPath); err == nil {
				if rt := transcribeFal(rawPath); betterText(rt, t) {
					t = rt
				}
			}
		}
		result <- t
	}()

	for i := 0; i < 2; i++ {
		if t := <-result; t != "" {
			return t
		}
	}
	return ""
}

func betterText(a string, b string) bool {
	a = strings.TrimSpace(a)
	b = strings.TrimSpace(b)
	if a == "" {
		return false
	}
	if b == "" {
		return true
	}
	return len(a) > len(b)+4
}

func transcribeChunkAPI(audioPath string) string {
	if config.APIKey == "" {
		return ""
	}

	file, err := os.Open(audioPath)
	if err != nil {
		return ""
	}
	defer file.Close()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, _ := writer.CreateFormFile("audio_file", filepath.Base(audioPath))
	io.Copy(part, file)
	writer.WriteField("translate_to_english", "false")
	writer.Close()

	req, _ := http.NewRequest("POST", "https://api.text-generator.io/api/v1/audio-file-extraction", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("secret", config.APIKey)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Chunk API error:", err)
		return ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return ""
	}

	body, _ := io.ReadAll(resp.Body)
	var result APIResponse
	json.Unmarshal(body, &result)

	return result.Text
}

func transcribeFal(audioPath string) string {
	locations := []string{
		filepath.Join(filepath.Dir(os.Args[0]), "fal_whisper.py"),
		"/usr/local/share/voicetype/fal_whisper.py",
		filepath.Join(os.Getenv("HOME"), ".local/share/voicetype/fal_whisper.py"),
	}

	var scriptPath string
	for _, loc := range locations {
		if _, err := os.Stat(loc); err == nil {
			scriptPath = loc
			break
		}
	}

	if scriptPath == "" {
		return ""
	}

	cmd := exec.Command("/usr/bin/python3", scriptPath, audioPath)
	if config.FalKey != "" {
		cmd.Env = append(os.Environ(), "FAL_KEY="+config.FalKey)
	}
	out, err := cmd.Output()
	if err != nil {
		fmt.Println("Fal whisper error:", err)
		return ""
	}

	text := string(bytes.TrimSpace(out))
	if text != "" {
		fmt.Println("Fal transcribed:", truncate(text, 50))
	}
	return text
}

func combineChunkResults() string {
	chunkResultsMu.Lock()
	defer chunkResultsMu.Unlock()

	if len(chunkResults) == 0 {
		return ""
	}

	indices := make([]int, 0, len(chunkResults))
	for idx := range chunkResults {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	var parts []string
	for _, idx := range indices {
		if text := chunkResults[idx]; text != "" {
			parts = append(parts, text)
		}
	}

	return strings.Join(parts, " ")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func preloadLocalModel() {
	localModelMu.Lock()
	if localModelReady || localModelWarming {
		localModelMu.Unlock()
		return
	}
	localModelWarming = true
	localModelMu.Unlock()

	go func() {
		ready := false

		if _, err := localASRRequest("preload", ""); err == nil {
			ready = true
		} else {
			serverScript := findScriptPath("model_server.py")
			if serverScript != "" {
				if isSocketStale(localModelSocket) {
					os.Remove(localModelSocket)
				}

				exec.Command(nemoPythonPath, serverScript).Start()

				deadline := time.Now().Add(18 * time.Second)
				for time.Now().Before(deadline) {
					if _, err := localASRRequest("preload", ""); err == nil {
						ready = true
						break
					}
					time.Sleep(250 * time.Millisecond)
				}
			}
		}

		localModelMu.Lock()
		localModelReady = ready
		localModelWarming = false
		localModelMu.Unlock()
	}()
}

func waitForLocalModelReady(timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		localModelMu.Lock()
		ready := localModelReady
		warming := localModelWarming
		localModelMu.Unlock()

		if ready {
			return true
		}
		if !warming || time.Now().After(deadline) {
			return false
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func showVisualizer() {
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
	text := transcribeLocal(audioPath)
	if text != "" {
		return text
	}

	text = transcribeAPI(audioPath)
	if text != "" {
		return text
	}

	fmt.Println("API failed, trying fal whisper...")
	text = transcribeFal(audioPath)
	if text != "" {
		return text
	}
	return ""
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

	client := &http.Client{Timeout: 30 * time.Second}
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
	preloadLocalModel()
	if !waitForLocalModelReady(1500 * time.Millisecond) {
		return ""
	}

	text, err := localASRRequest("transcribe", audioPath)
	if err != nil {
		localModelMu.Lock()
		localModelReady = false
		localModelMu.Unlock()

		fmt.Println("Local ASR error:", err)
		return ""
	}

	text = strings.TrimSpace(text)
	if text == "" {
		fmt.Println("Local ASR not available")
		return ""
	}

	fmt.Println("Local transcribed:", text)
	return text
}

func findScriptPath(name string) string {
	locations := []string{
		filepath.Join(filepath.Dir(os.Args[0]), name),
		"/usr/local/share/voicetype/" + name,
		filepath.Join(os.Getenv("HOME"), ".local/share/voicetype", name),
	}

	for _, loc := range locations {
		if _, err := os.Stat(loc); err == nil {
			return loc
		}
	}

	return ""
}

func isSocketStale(path string) bool {
	if _, err := os.Stat(path); err != nil {
		return false
	}
	conn, err := net.DialTimeout("unix", path, 150*time.Millisecond)
	if err != nil {
		return true
	}
	conn.Close()
	return false
}

func localASRRequest(action, audioPath string) (string, error) {
	payload := map[string]string{"action": action}
	if audioPath != "" {
		payload["path"] = audioPath
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	conn, err := net.DialTimeout("unix", localModelSocket, 200*time.Millisecond)
	if err != nil {
		return "", err
	}
	defer conn.Close()

	conn.SetDeadline(time.Now().Add(45 * time.Second))
	if _, err := conn.Write(body); err != nil {
		return "", err
	}

	respBytes, err := io.ReadAll(conn)
	if err != nil {
		return "", err
	}

	var resp localASRResponse
	if err := json.Unmarshal(respBytes, &resp); err != nil {
		return "", err
	}

	if resp.Error != "" {
		return "", fmt.Errorf(resp.Error)
	}
	if action == "preload" {
		if resp.Status != "ok" {
			return "", fmt.Errorf("local preload failed")
		}
		return "", nil
	}

	if resp.Text == "" && resp.Status != "ok" {
		return "", nil
	}

	return strings.TrimSpace(resp.Text), nil
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
