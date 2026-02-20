package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"crypto/sha1"
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
const localModelGlobalLockPath = "/tmp/voicetype-model.lock"
const hotkeyGlobalLockPath = "/tmp/voicetype-hotkey.lock"
const (
	defaultAutoLearnBudget         = 90
	defaultAutoLearnTopSamples     = 10
	defaultAutoLearnMinSamples     = 10
	defaultAutoLearnWarmupUses      = 10
	defaultAutoLearnCooldownMins   = 45
)

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
	recordingMu    sync.Mutex
	localModelReady bool
	localModelWarming bool
	instanceRecordingFlag string
	lastHotkeyTime time.Time
	autoLearnMu sync.Mutex
	autoLearnRunning bool
	autoLearnLastSignature string
	autoLearnLastRan time.Time
	autoLearnStatePath string
	instanceLock *os.File
	benchmarkMu sync.Mutex
	benchmarkData BenchmarkData
	benchmarkPath string
 )

const benchmarkWindowSize = 20

type WordBucket string
const (
	BucketShort  WordBucket = "short"
	BucketMedium WordBucket = "medium"
	BucketLong   WordBucket = "long"
)

type TranscriptionRecord struct {
	Provider  string     `json:"provider"`
	ElapsedMs float64    `json:"elapsed_ms"`
	WordCount int        `json:"word_count"`
	Bucket    WordBucket `json:"bucket"`
	Timestamp int64      `json:"ts"`
}

type BenchmarkData struct {
	Stats map[string]map[WordBucket][]TranscriptionRecord `json:"stats"`
}

type providerEntry struct {
	name       string
	available  func() bool
	transcribe func(string) string
}

type AutoLearnState struct {
	UseCount int `json:"use_count"`
}

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
	if !getEnvBool("VOICETYPE_ALLOW_MULTI_INSTANCE", false) && !acquireInstanceLock() {
		fmt.Println("Another voicetype instance is already running. Exiting.")
		return
	}
	defer func() {
		if instanceLock != nil {
			instanceLock.Close()
		}
	}()

	home, _ := os.UserHomeDir()
	configPath = filepath.Join(home, ".config", "voicetype", "config.json")
	autoLearnStatePath = filepath.Join(filepath.Dir(configPath), "auto_learn_state.json")
	instanceRecordingFlag = filepath.Join(os.TempDir(), fmt.Sprintf("voicetype-recording-%d", os.Getpid()))
	loadConfig()
	benchmarkPath = filepath.Join(filepath.Dir(configPath), "provider_stats.json")
	loadBenchmark()

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

func acquireInstanceLock() bool {
	lockPath := filepath.Join(os.TempDir(), "voicetype.lock")
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return true
	}

	// Try to obtain an exclusive lock without blocking so a second process
	// exits instead of racing for audio hardware and trigger files.
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = f.Close()
		return false
	}

	_ = f.Truncate(0)
	_, _ = f.WriteString(fmt.Sprintf("%d\n", os.Getpid()))
	instanceLock = f
	return true
}

func setRecording(active bool) bool {
	recordingMu.Lock()
	defer recordingMu.Unlock()
	if active {
		if recording {
			return false
		}
		recording = true
		return true
	}
	if !recording {
		return false
	}
	recording = false
	return true
}

func isRecording() bool {
	recordingMu.Lock()
	defer recordingMu.Unlock()
	return recording
}

func withNamedLock(lockPath string, timeout time.Duration, fn func() error) error {
	lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer lockFile.Close()

	deadline := time.Now().Add(timeout)
	var lockErr error
	for {
		if err := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err == nil {
			defer syscall.Flock(int(lockFile.Fd()), syscall.LOCK_UN)
			return fn()
		} else {
			lockErr = err
		}
		if time.Now().After(deadline) {
			return lockErr
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func withHotkeyLock(fn func() error) error {
	return withNamedLock(hotkeyGlobalLockPath, 500*time.Millisecond, fn)
}

func withModelLock(fn func() error) error {
	return withNamedLock(localModelGlobalLockPath, 30*time.Second, fn)
}

func setLocalModelReady(ready bool) {
	localModelMu.Lock()
	localModelReady = ready
	localModelWarming = false
	localModelMu.Unlock()
}

func isLocalModelReady() bool {
	localModelMu.Lock()
	defer localModelMu.Unlock()
	return localModelReady
}

func getEnvFileValue(key string) string {
	if configPath == "" {
		return ""
	}
	candidates := []string{
		filepath.Join(filepath.Dir(configPath), "env"),
		filepath.Join(os.Getenv("HOME"), ".secretbashrc"),
	}
	for _, envFile := range candidates {
		envData, err := os.ReadFile(envFile)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(envData), "\n") {
			line = strings.TrimSpace(line)
			prefix := key + "="
			if strings.HasPrefix(line, prefix) {
				return strings.Trim(strings.SplitN(line, "=", 2)[1], "\"' ")
			}
		}
	}
	return ""
}

func ensureAutoLearnStateDir() {
	if autoLearnStatePath == "" {
		return
	}
	_ = os.MkdirAll(filepath.Dir(autoLearnStatePath), 0o755)
}

func incrementAutoLearnUseCount() int {
	if autoLearnStatePath == "" {
		return 1
	}
	ensureAutoLearnStateDir()
	state := AutoLearnState{}
	if data, err := os.ReadFile(autoLearnStatePath); err == nil {
		_ = json.Unmarshal(data, &state)
	}
	state.UseCount++
	data, _ := json.Marshal(state)
	_ = os.WriteFile(autoLearnStatePath, data, 0644)
	return state.UseCount
}

func getGroqKey() string {
	if key := strings.TrimSpace(os.Getenv("GROQ_API_KEY")); key != "" {
		return key
	}
	return getEnvFileValue("GROQ_API_KEY")
}

func getGeminiKey() string {
	if key := strings.TrimSpace(os.Getenv("GEMINI_API_KEY")); key != "" {
		return key
	}
	return getEnvFileValue("GEMINI_API_KEY")
}

func resolveAutoLearnProvider() string {
	provider := strings.TrimSpace(os.Getenv("VOICETYPE_AUTO_LEARN_PROVIDER"))
	if provider != "" {
		return provider
	}
	if getGroqKey() != "" {
		return "groq"
	}
	if getGeminiKey() != "" {
		return "gemini"
	}
	if config.FalKey != "" {
		return "fal"
	}
	return "auto"
}

func resolveAutoLearnProfile(provider string) string {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "groq":
		return "speed"
	case "gemini":
		return "balanced"
	case "fal":
		return "clean"
	default:
		return "default"
	}
}

func getEnvBool(name string, def bool) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
	if v == "" {
		return def
	}
	switch v {
	case "1", "true", "yes", "on", "y":
		return true
	case "0", "false", "no", "off", "n":
		return false
	default:
		return def
	}
}

func getEnvInt(name string, def int) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	if n, err := strconv.Atoi(v); err == nil {
		return n
	}
	return def
}

func setEnvValue(env []string, key, value string) []string {
	if value == "" {
		return env
	}
	prefix := key + "="
	for i, e := range env {
		if strings.HasPrefix(e, prefix) {
			env[i] = prefix + value
			return env
		}
	}
	return append(env, prefix+value)
}

func hashSignature(data string) string {
	sum := sha1.Sum([]byte(data))
	return fmt.Sprintf("%x", sum[:])
}

func isAudioFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".wav", ".ogg", ".mp3", ".m4a", ".flac", ".webm":
		return true
	default:
		return false
	}
}

func listSamples(samplesDir string) ([]string, error) {
	candidates := []string{
		samplesDir,
		filepath.Join(samplesDir, "utterances"),
		filepath.Join(samplesDir, "rolling"),
	}
	seen := make(map[string]struct{})
	var out []string

	for _, root := range candidates {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			full := filepath.Join(root, entry.Name())
			if isAudioFile(full) {
				if _, exists := seen[full]; !exists {
					seen[full] = struct{}{}
					out = append(out, full)
				}
			}
		}
	}

	return out, nil
}

func getSampleStats(samplesDir string) (count int, signature string, err error) {
	indexPath := filepath.Join(samplesDir, "sample_index.json")
	if indexBytes, readErr := os.ReadFile(indexPath); readErr == nil {
		type sampleRec struct {
			Path string `json:"path"`
		}
		var recs []sampleRec
		if err := json.Unmarshal(indexBytes, &recs); err == nil {
			valid := 0
			for _, r := range recs {
				if r.Path != "" {
					if _, err := os.Stat(r.Path); err == nil {
						valid++
					}
				}
			}
			return valid, hashSignature(string(indexBytes)), nil
		}
	}

	paths, listErr := listSamples(samplesDir)
	if listErr != nil {
		return 0, "", listErr
	}
	if len(paths) == 0 {
		return 0, "", nil
	}

	hash := sha1.New()
	for _, path := range paths {
		hash.Write([]byte(path))
		info, infoErr := os.Stat(path)
		if infoErr != nil {
			continue
		}
		hash.Write([]byte(strconv.FormatInt(info.Size(), 10)))
		hash.Write([]byte(strconv.FormatInt(info.ModTime().UnixNano(), 10)))
	}
	return len(paths), fmt.Sprintf("%x", hash.Sum(nil)), nil
}

func startAutoOptimization() {
	if !getEnvBool("VOICETYPE_AUTO_LEARN", true) {
		return
	}
	if config.APIKey == "" && config.FalKey == "" && getGroqKey() == "" && getGeminiKey() == "" {
		fmt.Println("auto-learning skipped: no API key configured")
		return
	}

	warmupUses := getEnvInt("VOICETYPE_AUTO_LEARN_WARMUP_USES", defaultAutoLearnWarmupUses)
	if warmupUses < 1 {
		warmupUses = defaultAutoLearnWarmupUses
	}
	useCount := incrementAutoLearnUseCount()
	if useCount < warmupUses {
		fmt.Printf("auto-learning skipped: warmup %d/%d voicetype uses\n", useCount, warmupUses)
		return
	}

	minSamples := getEnvInt("VOICETYPE_AUTO_LEARN_MIN_SAMPLES", defaultAutoLearnMinSamples)
	if minSamples <= 0 {
		minSamples = defaultAutoLearnMinSamples
	}
	topSamples := getEnvInt("VOICETYPE_AUTO_LEARN_TOP_SAMPLES", defaultAutoLearnTopSamples)
	if topSamples <= 0 {
		topSamples = defaultAutoLearnTopSamples
	}
	budget := getEnvInt("VOICETYPE_AUTO_LEARN_BUDGET", defaultAutoLearnBudget)
	if budget <= 0 {
		budget = defaultAutoLearnBudget
	}
	cooldownMins := getEnvInt("VOICETYPE_AUTO_LEARN_COOLDOWN_MIN", defaultAutoLearnCooldownMins)
	if cooldownMins < 0 {
		cooldownMins = 0
	}

	vadScript := findScriptPath("vad_chunker.py")
	if vadScript == "" {
		fmt.Println("auto-learning skipped: vad_chunker.py not found")
		return
	}
	samplesDir := filepath.Join(filepath.Dir(vadScript), "samples")
	scriptDir := filepath.Dir(vadScript)
	optimizeScript := findScriptPath("optimize.py")
	if optimizeScript == "" {
		optimizeScript = filepath.Join(scriptDir, "optimize.py")
	}
	if _, err := os.Stat(optimizeScript); err != nil {
		fmt.Println("auto-learning skipped: optimize.py not found")
		return
	}

	count, signature, err := getSampleStats(samplesDir)
	if err != nil {
		fmt.Printf("auto-learning skipped: sample scan failed (%v)\n", err)
		return
	}
	if count < minSamples {
		fmt.Printf("auto-learning skipped: %d samples, need %d\n", count, minSamples)
		return
	}
	if topSamples > count {
		topSamples = count
	}

	autoLearnMu.Lock()
	if autoLearnRunning {
		autoLearnMu.Unlock()
		return
	}

	cooldown := time.Duration(cooldownMins) * time.Minute
	if !autoLearnLastRan.IsZero() && !autoLearnLastSignatureEmpty(autoLearnLastSignature) && signature == autoLearnLastSignature && time.Since(autoLearnLastRan) < cooldown {
		autoLearnMu.Unlock()
		return
	}

	autoLearnRunning = true
	autoLearnLastSignature = signature
	autoLearnLastRan = time.Now()
	autoLearnMu.Unlock()

	go func() {
		defer func() {
			autoLearnMu.Lock()
			autoLearnRunning = false
			autoLearnMu.Unlock()
		}()

		args := []string{
			optimizeScript,
			"--use-real-samples",
			"--samples-dir", samplesDir,
			"--top-samples", strconv.Itoa(topSamples),
			"--budget", strconv.Itoa(budget),
		}
		autoProvider := resolveAutoLearnProvider()
		if strings.TrimSpace(autoProvider) != "" {
			args = append(args, "--provider", autoProvider)
		}
		autoProfile := strings.TrimSpace(os.Getenv("VOICETYPE_AUTO_LEARN_ENCODE_PROFILE"))
		if autoProfile == "" {
			autoProfile = resolveAutoLearnProfile(autoProvider)
		}
		if autoProfile != "" {
			args = append(args, "--encode-profile", autoProfile)
		}
		if getEnvBool("VOICETYPE_AUTO_LEARN_REFRESH_SAMPLES", false) {
			args = append(args, "--refresh-samples")
		}
		if getEnvBool("VOICETYPE_AUTO_LEARN_AGGRESSIVE", false) {
			args = append(args, "--aggressive-encode")
		}

		cmd := exec.Command("python3", args...)
		cmd.Env = setEnvValue(os.Environ(), "TEXT_GENERATOR_API_KEY", config.APIKey)
		if config.FalKey != "" {
			cmd.Env = setEnvValue(cmd.Env, "FAL_KEY", config.FalKey)
		}
	if groqKey := getGroqKey(); strings.TrimSpace(groqKey) != "" {
		cmd.Env = setEnvValue(cmd.Env, "GROQ_API_KEY", groqKey)
	}
	if geminiKey := getGeminiKey(); geminiKey != "" {
		cmd.Env = setEnvValue(cmd.Env, "GEMINI_API_KEY", geminiKey)
	}

		fmt.Printf("auto-learning: running %s\n", strings.Join(args, " "))
		out, err := cmd.CombinedOutput()
		if err != nil {
			msg := strings.TrimSpace(string(out))
			if msg == "" {
				msg = err.Error()
			}
			fmt.Printf("auto-learning failed: %s\n", msg)
			return
		}
		fmt.Printf("auto-learning completed (samples=%d, top=%d)\n", count, topSamples)
	}()
}

func autoLearnLastSignatureEmpty(signature string) bool {
	return strings.TrimSpace(signature) == ""
}

func startHotkeyListener() {
	home, _ := os.UserHomeDir()
	triggerFile := filepath.Join(home, ".cache", "voicetype-trigger")
	os.MkdirAll(filepath.Dir(triggerFile), 0755)

	for {
		if _, err := os.Stat(triggerFile); err == nil {
			_ = withHotkeyLock(func() error {
				if _, err2 := os.Stat(triggerFile); err2 == nil {
					os.Remove(triggerFile)
					handleHotkey()
				}
				return nil
			})
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

	if isRecording() {
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
	if !setRecording(true) {
		return
	}

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
	scriptPath := findScriptPath("vad_chunker.py")
	if scriptPath != "" {
		samplesDir := filepath.Join(filepath.Dir(scriptPath), "samples")
		fmt.Printf("using VAD chunker: %s\n", scriptPath)
		fmt.Printf("samples dir: %s\n", samplesDir)
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
		for isRecording() {
			time.Sleep(100 * time.Millisecond)
			if _, err := os.Stat("/tmp/voicetype-visualizer-run"); os.IsNotExist(err) {
				if isRecording() {
					stopRecording()
				}
				return
			}
		}
	}()

	fmt.Println("Recording started...")
}

func stopRecording() {
	if !setRecording(false) {
		return
	}
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

	go startAutoOptimization()
}

func watchChunks() {
	readyPath := filepath.Join(chunkDir, "ready")
	var lastPos int64 = 0

	for isRecording() {
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

func wordBucket(n int) WordBucket {
	if n <= 5 {
		return BucketShort
	}
	if n <= 20 {
		return BucketMedium
	}
	return BucketLong
}

func loadBenchmark() {
	benchmarkMu.Lock()
	defer benchmarkMu.Unlock()
	benchmarkData.Stats = make(map[string]map[WordBucket][]TranscriptionRecord)
	data, err := os.ReadFile(benchmarkPath)
	if err != nil {
		return
	}
	json.Unmarshal(data, &benchmarkData)
	if benchmarkData.Stats == nil {
		benchmarkData.Stats = make(map[string]map[WordBucket][]TranscriptionRecord)
	}
}

func saveBenchmark() {
	data, err := json.MarshalIndent(benchmarkData, "", "  ")
	if err != nil {
		return
	}
	os.MkdirAll(filepath.Dir(benchmarkPath), 0755)
	os.WriteFile(benchmarkPath, data, 0644)
}

func recordTranscription(provider string, elapsedMs float64, text string) {
	wc := len(strings.Fields(text))
	bucket := wordBucket(wc)
	rec := TranscriptionRecord{
		Provider:  provider,
		ElapsedMs: elapsedMs,
		WordCount: wc,
		Bucket:    bucket,
		Timestamp: time.Now().Unix(),
	}
	benchmarkMu.Lock()
	defer benchmarkMu.Unlock()
	if benchmarkData.Stats == nil {
		benchmarkData.Stats = make(map[string]map[WordBucket][]TranscriptionRecord)
	}
	if benchmarkData.Stats[provider] == nil {
		benchmarkData.Stats[provider] = make(map[WordBucket][]TranscriptionRecord)
	}
	recs := append(benchmarkData.Stats[provider][bucket], rec)
	if len(recs) > benchmarkWindowSize {
		recs = recs[len(recs)-benchmarkWindowSize:]
	}
	benchmarkData.Stats[provider][bucket] = recs
	saveBenchmark()
}

func avgLatency(provider string, bucket WordBucket) (float64, bool) {
	benchmarkMu.Lock()
	defer benchmarkMu.Unlock()
	recs, ok := benchmarkData.Stats[provider][bucket]
	if !ok || len(recs) == 0 {
		return 0, false
	}
	var sum float64
	for _, r := range recs {
		sum += r.ElapsedMs
	}
	return sum / float64(len(recs)), true
}

func avgLatencyOverall(provider string) (float64, bool) {
	benchmarkMu.Lock()
	defer benchmarkMu.Unlock()
	buckets, ok := benchmarkData.Stats[provider]
	if !ok {
		return 0, false
	}
	var sum float64
	var count int
	for _, recs := range buckets {
		for _, r := range recs {
			sum += r.ElapsedMs
			count++
		}
	}
	if count == 0 {
		return 0, false
	}
	return sum / float64(count), true
}

func getProviderOrder() []providerEntry {
	return []providerEntry{
		{"groq", func() bool { return getGroqKey() != "" }, transcribeGroq},
		{"gemini", func() bool { return getGeminiKey() != "" }, transcribeGemini},
		{"textgen", func() bool { return config.APIKey != "" }, transcribeChunkAPI},
		{"fal", func() bool { return config.FalKey != "" || findScriptPath("fal_whisper.py") != "" }, transcribeFal},
		{"local", func() bool { return true }, transcribeLocal},
	}
}

func rankedProviders(bucket WordBucket) []providerEntry {
	providers := getProviderOrder()
	sort.SliceStable(providers, func(i, j int) bool {
		li, oki := avgLatency(providers[i].name, bucket)
		if !oki {
			li, oki = avgLatencyOverall(providers[i].name)
		}
		lj, okj := avgLatency(providers[j].name, bucket)
		if !okj {
			lj, okj = avgLatencyOverall(providers[j].name)
		}
		if !oki || !okj {
			return false
		}
		return li < lj
	})
	return providers
}

func timedTranscribe(name string, fn func(string) string, audioPath string) string {
	start := time.Now()
	text := fn(audioPath)
	if text != "" {
		elapsed := float64(time.Since(start).Milliseconds())
		recordTranscription(name, elapsed, text)
	}
	return text
}

func transcribeChunk(processedPath string, rawPath string) string {
	providers := rankedProviders(BucketMedium)
	fmt.Println("fastest:", providers[0].name)
	for _, p := range providers {
		if !p.available() {
			continue
		}
		text := timedTranscribe(p.name, p.transcribe, processedPath)
		if text == "" && rawPath != processedPath {
			if _, err := os.Stat(rawPath); err == nil {
				if rt := timedTranscribe(p.name, p.transcribe, rawPath); betterText(rt, text) {
					text = rt
				}
			}
		}
		if text != "" {
			return text
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

func transcribeGroq(audioPath string) string {
	key := getGroqKey()
	if strings.TrimSpace(key) == "" {
		return ""
	}

	file, err := os.Open(audioPath)
	if err != nil {
		return ""
	}
	defer file.Close()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, _ := writer.CreateFormFile("file", filepath.Base(audioPath))
	if _, err := io.Copy(part, file); err != nil {
		return ""
	}
	writer.WriteField("model", "whisper-large-v3-turbo")
	if err := writer.Close(); err != nil {
		return ""
	}

	req, _ := http.NewRequest("POST", "https://api.groq.com/openai/v1/audio/transcriptions", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+key)

	client := &http.Client{Timeout: 45 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		if len(body) > 0 {
			fmt.Println("Groq API error:", strings.TrimSpace(string(body)))
		}
		return ""
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ""
	}

	var result APIResponse
	if err := json.Unmarshal(body, &result); err != nil {
		type groqResponse struct {
			Text string `json:"text"`
		}
		var r groqResponse
		if err2 := json.Unmarshal(body, &r); err2 != nil {
			return ""
		}
		result.Text = r.Text
	}

	if result.Text == "" {
		return ""
	}
	fmt.Println("Groq transcribed:", truncate(result.Text, 50))
	return result.Text
}

func transcribeGemini(audioPath string) string {
	key := getGeminiKey()
	if key == "" {
		return ""
	}

	audioData, err := os.ReadFile(audioPath)
	if err != nil {
		return ""
	}
	b64 := base64.StdEncoding.EncodeToString(audioData)

	mime := "audio/ogg"
	if strings.HasSuffix(audioPath, ".wav") {
		mime = "audio/wav"
	} else if strings.HasSuffix(audioPath, ".mp3") {
		mime = "audio/mp3"
	}

	reqBody := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"role": "user",
				"parts": []map[string]interface{}{
					{"text": "Transcribe this audio exactly. Output only the transcription text, nothing else."},
					{"inline_data": map[string]interface{}{"mime_type": mime, "data": b64}},
				},
			},
		},
		"generationConfig": map[string]interface{}{
			"thinkingConfig": map[string]interface{}{
				"thinkingLevel": "LOW",
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return ""
	}

	url := "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key=" + key
	req, _ := http.NewRequest("POST", url, bytes.NewReader(jsonData))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Gemini error:", err)
		return ""
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		fmt.Println("Gemini API error:", resp.StatusCode, truncate(string(body), 200))
		return ""
	}

	var gemResp struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text    string `json:"text"`
					Thought bool   `json:"thought"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(body, &gemResp); err != nil {
		fmt.Println("Gemini parse error:", err)
		return ""
	}

	var texts []string
	for _, c := range gemResp.Candidates {
		for _, p := range c.Content.Parts {
			if p.Thought {
				continue
			}
			if t := strings.TrimSpace(p.Text); t != "" {
				texts = append(texts, t)
			}
		}
	}
	text := strings.Join(texts, " ")
	if text == "" {
		return ""
	}
	fmt.Println("Gemini transcribed:", truncate(text, 50))
	return text
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
	go func() {
		_ = withModelLock(func() error {
			if isLocalModelReady() {
				return nil
			}
			return ensureLocalModelReadyLocked()
		})
	}()
}

func ensureLocalModelReadyLocked() error {
	if _, err := localASRRequest("preload", ""); err == nil {
		setLocalModelReady(true)
		return nil
	}

	serverScript := findScriptPath("model_server.py")
	if serverScript == "" {
		setLocalModelReady(false)
		return fmt.Errorf("model_server.py not found")
	}
	if isSocketStale(localModelSocket) {
		os.Remove(localModelSocket)
	}

	cmd := exec.Command(nemoPythonPath, serverScript)
	if err := cmd.Start(); err != nil {
		setLocalModelReady(false)
		return err
	}

	deadline := time.Now().Add(18 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := localASRRequest("preload", ""); err == nil {
			setLocalModelReady(true)
			return nil
		}
		time.Sleep(250 * time.Millisecond)
	}

	setLocalModelReady(false)
	return fmt.Errorf("local model never became ready")
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
	return transcribeChunk(audioPath, audioPath)
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
	var text string
	if err := withModelLock(func() error {
		if !isLocalModelReady() {
			if err := ensureLocalModelReadyLocked(); err != nil {
				return err
			}
		}
		var err error
		text, err = localASRRequest("transcribe", audioPath)
		if err != nil {
			setLocalModelReady(false)
			return err
		}
		text = strings.TrimSpace(text)
		return nil
	}); err != nil {
		return ""
	}

	if text == "" {
		fmt.Println("Local ASR not available")
		return ""
	}

	fmt.Println("Local transcribed:", text)
	return text
}

func findScriptPath(name string) string {
	home, _ := os.UserHomeDir()
	locations := []string{
		filepath.Join(home, "code", "voicetype", name),
		filepath.Join(os.Getenv("HOME"), ".local/share/voicetype", name),
		"/usr/local/share/voicetype/" + name,
		filepath.Join(filepath.Dir(os.Args[0]), name),
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
