#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, os, sys, time, logging, subprocess, threading
import numpy as np, sentencepiece as spm, soundfile as sf
from typing import Optional, Tuple, Dict, Any, List, Union
import signal, audioop, atexit 
from queue import Queue
from pathlib import Path
from enum import Enum
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions, get_available_providers, get_device)
import pyaudio, select, yaml, math, psutil, gc, uuid, cn2an, kaldi_native_fbank as knf, onnxruntime as ort
from rknnlite.api.rknn_lite import RKNNLite

# 内存监控工具类
class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.logger = logging.getLogger("MemoryMonitor")
        self.baseline_memory = self.get_memory_info()
        vm = psutil.virtual_memory()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔧 RK3588 板子内存配置:")
        self.logger.info(f"  总内存: {vm.total / 1024 / 1024:.2f} MB ({vm.total / 1024 / 1024 / 1024:.2f} GB)")
        self.logger.info(f"  初始可用: {vm.available / 1024 / 1024:.2f} MB")
        self.logger.info(f"  初始使用率: {vm.percent:.2f}%")
        self.logger.info(f"{'='*60}\n")
        
    def get_memory_info(self) -> Dict[str, float]:
        mem_info = self.process.memory_info()
        virtual_mem = psutil.virtual_memory()
        
        return {
            'process_rss': mem_info.rss / 1024 / 1024,
            'process_vms': mem_info.vms / 1024 / 1024,
            'process_percent': self.process.memory_percent(),
            'system_total': virtual_mem.total / 1024 / 1024,
            'system_available': virtual_mem.available / 1024 / 1024,
            'system_used': virtual_mem.used / 1024 / 1024,
            'system_percent': virtual_mem.percent,
            'system_free': virtual_mem.free / 1024 / 1024,
            'system_buffers': getattr(virtual_mem, 'buffers', 0) / 1024 / 1024,
            'system_cached': getattr(virtual_mem, 'cached', 0) / 1024 / 1024,
        }
    
    def log_memory(self, stage: str, details: str = ""):
        mem = self.get_memory_info()
        delta_process = mem['process_rss'] - self.baseline_memory['process_rss']
        delta_system = mem['system_used'] - self.baseline_memory['system_used']
        
        log_msg = (
            f"\n{'='*70}\n"
            f"📊 [{stage}] RK3588 板子内存状态 {details}\n"
            f"{'='*70}\n"
            f"🖥️  板子系统内存:\n"
            f"  ├─ 总内存: {mem['system_total']:.2f} MB\n"
            f"  ├─ 已使用: {mem['system_used']:.2f} MB ({mem['system_percent']:.2f}%)\n"
            f"  ├─ 可用内存: {mem['system_available']:.2f} MB\n"
            f"  └─ 系统内存变化: {delta_system:+.2f} MB\n"
            f"\n"
            f"📱 本进程内存:\n"
            f"  ├─ 物理内存: {mem['process_rss']:.2f} MB\n"
            f"  ├─ 占板子总内存: {mem['process_percent']:.2f}%\n"
            f"  └─ 进程内存变化: {delta_process:+.2f} MB\n"
            f"{'='*70}"
        )
        self.logger.info(log_msg)
        return mem
    
    def get_memory_delta(self, start_mem: Dict[str, float]) -> Dict[str, float]:
        current_mem = self.get_memory_info()
        return {
            'process_rss_delta': current_mem['process_rss'] - start_mem['process_rss'],
            'system_used_delta': current_mem['system_used'] - start_mem['system_used'],
            'system_available_delta': current_mem['system_available'] - start_mem['system_available'],
            'system_percent_delta': current_mem['system_percent'] - start_mem['system_percent'],
        }
    
    def format_delta(self, delta: Dict[str, float]) -> str:
        return (
            f"[板子] 已用: {delta['system_used_delta']:+.2f} MB, "
            f"使用率: {delta['system_percent_delta']:+.2f}% | "
            f"[进程] RSS: {delta['process_rss_delta']:+.2f} MB"
        )

memory_monitor = MemoryMonitor()

# --- 全局配置 ---
ASR_DIR = "/home/orangepi/rknn-asr/runtime/RK3588/Linux/librknn_api/include"
LLM_SCRIPT_PATH = "/root/voice_assistant/run_llm.sh"
LLM_DIR = os.path.dirname(LLM_SCRIPT_PATH)
TTS_DIR = "/home/orangepi/rknn-tts/MeloTTS-RKNN2"
WORKDIR = os.path.join(os.path.expanduser("~"), "voice_assistant")
os.makedirs(WORKDIR, exist_ok=True)

sys.path.append(ASR_DIR)
sys.path.append(TTS_DIR)
from sensevoice_rknn import *
from melotts_rknn import *

ASR_RKNN_PATH = os.path.join(ASR_DIR, "sense-voice-encoder.rknn") ##
ASR_EMBED_PATH = os.path.join(ASR_DIR, "embedding.npy")
ASR_BPE_PATH = os.path.join(ASR_DIR, "chn_jpn_yue_eng_ko_spectok.bpe.model")
ASR_VAD_ONNX_PATH = os.path.join(ASR_DIR, "fsmnvad-offline.onnx")
ASR_VAD_CONFIG_YAML = os.path.join(ASR_DIR, "fsmn-config.yaml")
ASR_MVN_PATH = os.path.join(ASR_DIR, "am.mvn")

TTS_ENCODER_PATH = os.path.join(TTS_DIR, "encoder.onnx")
TTS_DECODER_PATH = os.path.join(TTS_DIR, "decoder.rknn") ##
TTS_LEXICON_PATH = os.path.join(TTS_DIR, "lexicon.txt")
TTS_TOKEN_PATH = os.path.join(TTS_DIR, "tokens.txt")
TTS_G_BIN_PATH = os.path.join(TTS_DIR, "g.bin")

RATE = 16000
PLAY_DEVICE = "hw:0,0"
TARGET_PLAY_SR = 16000
TARGET_PLAY_CH = 2

audio_queue = Queue()
play_queue = Queue()

def playback_worker():
    logger = logging.getLogger("PlaybackWorker")
    while True:
        wav_path = play_queue.get()  # 阻塞等待
        if wav_path is None:  # 结束信号（程序退出时可发）
            break
        success = play_audio_file(wav_path, 0, PLAY_DEVICE)
        os.remove(wav_path)

class RecorderState(Enum):
    STOPPED = 0
    LISTENING = 1
    RECORDING = 2

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# --- ASR 服务 ---
class AsrService:
    def __init__(self, mvn_path, embed_path, rknn_path, bpe_path, asr_dir):
        self.logger = logging.getLogger("AsrService")
        self.logger.info("加载 ASR 模型...")
        start_time = time.time() ###
        self.front = WavFrontend(cmvn_file=mvn_path) 
        self.vad = FSMNVad(asr_dir)
        self.model = SenseVoiceInferenceSession(
            embed_path, rknn_path, bpe_path, device_id=-1, intra_op_num_threads=4
        )
        self.languages = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        
        self.logger.info(f"ASR模型加载完毕，耗时 {time.time() - start_time:.2f} 秒。")

    def transcribe(self, waveform_16k_f32, language="zh", use_itn=True) -> Tuple[str, float]:
        self.logger.info("开始 ASR 推理...")
        start_time = time.time()
        
        segments = self.vad.segments_offline(waveform_16k_f32) 
        
        if not segments:
            self.logger.warning("VAD 未检测到语音片段。")
            return "", 0.0
            
        self.logger.info(f"VAD 检测到 {len(segments)} 个片段。")
        full_text = []

        for i, part in enumerate(segments):
            start_ms, end_ms = part[0], part[1]
            start_frame = int(start_ms * 16) 
            end_frame = int(end_ms * 16)
            segment_audio = waveform_16k_f32[start_frame:end_frame]
            
            if len(segment_audio) < 160: 
                continue 

            audio_feats = self.front.get_features(segment_audio)
            asr_result = self.model(
                audio_feats[None, ...], 
                language=self.languages.get(language, 0), 
                use_itn=use_itn
            )
            
            self.logger.info(f"[片段 {i}] [{start_ms/1000:.2f}s - {end_ms/1000:.2f}s] {asr_result}")
            full_text.append(asr_result)
        
        final_text = "".join(full_text)
        cleaned_text = re.sub(r'<\|[^>]*\|>', '', final_text)
        cleaned_text = cleaned_text.strip(' \n\r\t,。!?:;"\'。')
        
        if cleaned_text:
            cleaned_text += "，回答简短一些，保持50字以内！"
            final_text = cleaned_text
            
        elapsed = time.time() - start_time
        return final_text, elapsed

    def close(self):
        self.model.release()

# --- LLM 服务（支持流式输出） ---
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
class LlmService:
    def __init__(self, script_path: str, cwd_dir: Optional[str] = None, idle_timeout: float = 1.2, init_timeout: float = 120.0):
        self.logger = logging.getLogger("LlmService")
        self.script_path = script_path
        self.idle_timeout = float(idle_timeout)
        self.init_timeout = float(init_timeout)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._stdout_fd: Optional[int] = None
        self._start_and_wait_ready()

    def _start_and_wait_ready(self):
        self.logger.info(f"启动 LLM 守护进程: {self.script_path}")
        
        self._proc = subprocess.Popen(
            [self.script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            close_fds=True
        )

        self._stdout_fd = self._proc.stdout.fileno()

        ready_buf = ""
        start = time.time()
        while True:
            elapsed = time.time() - start
            remaining = max(0.1, self.init_timeout - elapsed)
            rlist, _, _ = select.select([self._stdout_fd], [], [], remaining)
            if rlist:
                chunk = os.read(self._stdout_fd, 4096)
                if not chunk:
                    break
                s = chunk.decode("utf-8", errors="ignore")
                ready_buf += s
                if "rkllm init success" in ready_buf:
                    self.logger.info("LLM 已初始化完成。")
                    return
            else:
                if time.time() - start >= self.init_timeout:
                    self.logger.error("等待 'rkllm init success' 超时。")
                    raise TimeoutError("LLM init timeout")

    def chat_stream(self, prompt_text: str, sentence_callback):
        if not prompt_text:
            return "", 0.0, 0.0

        with self._lock:
            start_time = time.time()
            first_sentence_time = None  # 首句生成时间
            
            self._proc.stdin.write((prompt_text + "\n").encode("utf-8"))
            self._proc.stdin.flush()

            collected = ""
            fd = self._stdout_fd
            
            # 句子分隔符（中英文标点）
            sentence_delimiters = ['。', '！', '？', '.', '!', '?', '，', '；']
            buffer = ""  # 当前积累的文本
            sentence_count = 0

            while True:
                timeout = self.idle_timeout
                rlist, _, _ = select.select([fd], [], [], timeout)

                if rlist:
                    chunk = os.read(fd, 4096)
                    if not chunk:
                        break
                    s = chunk.decode("utf-8", errors="ignore")
                    collected += s
                    
                    # 清理 ANSI 和过滤日志行
                    s_clean = ANSI_RE.sub("", s)
                    for line in s_clean.split('\n'):
                        line = line.rstrip()
                        if not line:
                            continue
                        # 跳过日志和提示符
                        if line.startswith("I rkllm:") or line.startswith("rkllm init") or \
                           line.startswith("Input:") or line.startswith("user:") or \
                           "time_used=" in line or line == prompt_text:
                            continue
                        
                        # 提取 robot: 后的内容
                        if line.lower().startswith("robot:"):
                            line = line[len("robot:"):].strip()
                        
                        buffer += line
                    
                    # 检查是否有完整句子
                    for delimiter in sentence_delimiters:
                        if delimiter in buffer:
                            # 按分隔符分割
                            parts = buffer.split(delimiter)
                            # 处理除最后一个之外的所有部分（它们是完整句子）
                            for i in range(len(parts) - 1):
                                sentence = parts[i].strip() + delimiter
                                if sentence.strip(delimiter).strip():  # 确保不是空句子
                                    sentence_count += 1
                                    current_time = time.time()
                                    
                                    # 计算首句时间（从LLM开始到第一个标点符号）
                                    is_first = (sentence_count == 1)
                                    if is_first:
                                        first_sentence_time = current_time - start_time
                                        self.logger.info(f"⚡ LLM 首句生成时间: {first_sentence_time:.2f}s")
                                    
                                    # 计算句子间隔时间
                                    sentence_time = current_time - start_time
                                    
                                    self.logger.info(f"📝 LLM 句子 [{sentence_count}] (累计 {sentence_time:.2f}s): {sentence}")
                                    # 调用回调函数
                                    sentence_callback(sentence, sentence_time, is_first)
                            
                            # 保留最后一个部分（可能是未完成的句子）
                            buffer = parts[-1]
                            break
                    
                    continue
                else:
                    break
            
            # 处理剩余的文本
            if buffer:
                sentence_count += 1
                current_time = time.time()
                sentence_time = current_time - start_time
                is_first = (sentence_count == 1)
                if is_first:
                    first_sentence_time = current_time - start_time
                    self.logger.info(f"⚡ LLM 首句生成时间: {first_sentence_time:.2f}s")
                self.logger.info(f"📝 LLM 最后片段 [{sentence_count}] (累计 {sentence_time:.2f}s): {buffer}")
                sentence_callback(buffer.strip(), sentence_time, is_first)

            # 解析完整输出
            raw_output = ANSI_RE.sub("", collected)
            lines = [ln.strip() for ln in raw_output.splitlines() if ln.strip()]

            # 提取总耗时
            llm_report_sec = 0.0
            for ln in reversed(lines):
                if "time_used=" in ln:
                    m = re.search(r"time_used\s*=\s*(\d+)\s*ms", ln)
                    if m:
                        llm_report_sec = float(m.group(1)) / 1000.0
                    break

            # 提取完整回答
            robot_idx = None
            for i, ln in enumerate(lines):
                if ln.lower().startswith("robot:"):
                    robot_idx = i

            answer = ""
            if robot_idx is not None:
                captured = []
                for ln in lines[robot_idx:]:
                    if "time_used=" in ln:
                        break
                    if ln.lower().startswith("robot:"):
                        captured.append(ln[len("robot:"):].strip())
                    else:
                        captured.append(ln)
                answer = " ".join([c for c in captured if c]).strip()
            else:
                filtered = []
                for ln in lines:
                    if ln.startswith("I rkllm:") or ln.startswith("rkllm init") or ln.startswith("Input:"):
                        continue
                    if "time_used=" in ln:
                        continue
                    if ln.lower().startswith("user:"):
                        continue
                    if ln == prompt_text or ln.startswith(prompt_text):
                        continue
                    filtered.append(ln)
                answer = " ".join(filtered).strip()

            if prompt_text and answer.lower().startswith(prompt_text.lower()):
                answer = answer[len(prompt_text):].strip()
            answer = re.sub(r'(?i)^user:\s*', '', answer).strip()
            answer = re.sub(r'\s+', ' ', answer).strip()

            elapsed = time.time() - start_time
            report_time = llm_report_sec if llm_report_sec > 0 else elapsed

            self.logger.info(f"💬 LLM 完整回答: {answer!r}，总耗时: {report_time:.3f}s")
            return answer, report_time, first_sentence_time if first_sentence_time else 0.0

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            self._proc.wait(timeout=3)
        self.logger.info("LLM 守护进程已关闭。")

# --- TTS 服务 ---
class TtsService:
    def __init__(self, encoder_path, decoder_path, lexicon_path, token_path, g_bin_path, sample_rate=44100, speed=1.0):
        self.logger = logging.getLogger("TtsService")
        self.sample_rate = sample_rate
        self.speed = speed
        self.dec_len = 65536 // 512  # 128
        self.logger.info("正在加载 TTS 模型...")
        start_time = time.time() ###

        self.lexicon = Lexicon(lexicon_path, token_path)
        sess_opt = SessionOptions()
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_enc = InferenceSession(encoder_path, sess_opt, providers=["CPUExecutionProvider"])
        self.decoder = RKNNLite()
        ret = self.decoder.load_rknn(decoder_path)
        if ret != 0:
            raise RuntimeError("Load decoder.rknn failed")
        self.decoder.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        self.g = np.fromfile(g_bin_path, dtype=np.float32).reshape(1, 256, 1)
        self.logger.info(f"TTS 模型加载完成，耗时 {time.time() - start_time:.2f}s")

    def synthesize_sentence(self, text: str, output_path: str) -> Tuple[bool, float, float, float]:
        start_time = time.time()
        enc_time = dec_time = 0.0

        text = text.strip()
        if not text:
            return False, 0, 0, 0

        # 直接使用原句（不再切分），因为句子很短
        audio_segments = []

        phone_str, yinjie_num, phones, tones = self.lexicon.convert(text)

        # 加 blank
        phone_str = intersperse(phone_str, 0)
        phones_np = np.array(intersperse(phones, 0), dtype=np.int32)
        tones_np = np.array(intersperse(tones, 0), dtype=np.int32)
        yinjie_num = np.array(yinjie_num, dtype=np.int32) * 2
        if yinjie_num.size > 0:
            yinjie_num[0] += 1

        pron_slices = generate_pronounce_slice(yinjie_num)
        phone_len = phones_np.shape[0]
        language = np.array([3] * phone_len, dtype=np.int32)

        # Encoder
        enc_start = time.time()
        z_p, pronoun_lens, audio_len_scalar = self.sess_enc.run(None, {
            'phone': phones_np,
            'g': self.g,
            'tone': tones_np,
            'language': language,
            'noise_scale': np.array([0.0], dtype=np.float32),
            'length_scale': np.array([1.0 / self.speed], dtype=np.float32),
            'noise_scale_w': np.array([0.0], dtype=np.float32),
            'sdp_ratio': np.array([0.0], dtype=np.float32),
        })
        enc_time += time.time() - enc_start

        audio_len = int(audio_len_scalar)
        pronoun_lens = np.array(pronoun_lens).flatten()
        pron_num = generate_word_pron_num(pronoun_lens, pron_slices)

        # z_p padding 到 decoder 能整除的长度
        actual_size = z_p.shape[-1]
        need_pad = self.dec_len * ((actual_size + self.dec_len - 1) // self.dec_len) - actual_size
        if need_pad > 0:
            z_p = np.pad(z_p, ((0,0),(0,0),(0, need_pad)), 'constant')

        # 分片解码（带 overlap + strip）
        pron_num_slices, zp_slices, strip_flags, _, is_long_list = generate_decode_slices(pron_num, self.dec_len)

        sub_audio_list = []
        for i in range(len(pron_num_slices)):
            p_start, p_end = pron_num_slices[i]
            z_start, z_end = zp_slices[i]
            strip_head, strip_tail = strip_flags[i]

            if is_long_list[i]:
                # 超长词单独处理
                sub_audio_list.extend(decode_long_word(self.decoder, z_p[..., z_start:z_end], self.g, self.dec_len))
            else:
                zp_slice = z_p[..., z_start:z_end]
                if zp_slice.shape[-1] < self.dec_len:
                    zp_slice = np.pad(zp_slice, ((0,0),(0,0),(0, self.dec_len - zp_slice.shape[-1])), 'constant')

                dec_start = time.time()
                audio_raw = self.decoder.inference(inputs=[zp_slice, self.g])[0].flatten()
                dec_time += time.time() - dec_start

                audio_raw = audio_raw[:512 * (z_end - z_start)]

                if strip_head and p_start > 0:
                    audio_raw = audio_raw[512 * pron_num[p_start]:]
                if strip_tail and p_end < len(pron_num):
                    audio_raw = audio_raw[:-512 * pron_num[p_end - 1]]

                sub_audio_list.append(audio_raw)

        merged_audio = merge_sub_audio(sub_audio_list, pad_size=0, audio_len=audio_len)
        audio_segments.append(merged_audio)

        final_audio = audio_numpy_concat(audio_segments, sr=self.sample_rate, speed=self.speed)
        sf.write(output_path, final_audio, self.sample_rate)
        total_time = time.time() - start_time
        return True, total_time, enc_time, dec_time

    def close(self):
        self.decoder.release()

# --- 录音器 ---
class AudioRecorder:
    logger = logging.getLogger("AudioRecorder")
    p = None
    stream = None
    state = RecorderState.STOPPED
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    CHUNK = 1024
    RMS_THRESHOLD = 300
    SILENCE_TIMEOUT_SEC = 1.8
    SILENCE_MAX_SEC = 5.0
    MAX_RECORD_SEC = 10.0

    @classmethod
    def start_stream(cls):
        if cls.p is None:
            cls.p = pyaudio.PyAudio()
        if cls.stream is None:
            cls.stream = cls.p.open(
                format=cls.FORMAT, channels=cls.CHANNELS, rate=RATE, input=True,
                frames_per_buffer=cls.CHUNK, start=False
            )
        atexit.register(cls.stop_stream)
        cls.logger.info("麦克风流已初始化。")

    @classmethod
    def stop_stream(cls):
        if cls.stream:
            cls.stream.stop_stream()
            cls.stream.close()
            cls.stream = None
        if cls.p:
            cls.p.terminate()
            cls.p = None
        cls.logger.info("麦克风流已关闭。")

    @classmethod
    def record_loop(cls):
        cls.start_stream()
        cls.stream.start_stream() 

        cls.state = RecorderState.LISTENING
        cls.logger.info("录音线程启动，进入监听模式。")

        chunks_per_sec = RATE / cls.CHUNK
        silence_limit_chunks = int(chunks_per_sec * cls.SILENCE_TIMEOUT_SEC)
        max_record_chunks = int(chunks_per_sec * cls.MAX_RECORD_SEC)

        while cls.state != RecorderState.STOPPED:
            cls.logger.info("\n--- 请开始说话 (正在监听麦克风) ---")
            
            frames = []
            silent_chunks = 0
            is_recording = False
            LISTENING_timeout_start = time.time()

            while cls.state != RecorderState.STOPPED:
                if not is_recording and (time.time() - LISTENING_timeout_start > cls.SILENCE_MAX_SEC):
                    cls.logger.debug(f"🕓 {cls.SILENCE_MAX_SEC}秒未检测到语音，继续监听...")
                    LISTENING_timeout_start = time.time()
                
                data = cls.stream.read(cls.CHUNK, exception_on_overflow=False)
                rms = audioop.rms(data, 2)

                if not is_recording:
                    if rms > cls.RMS_THRESHOLD:
                        cls.logger.info("🎯 检测到语音，开始录制...")
                        is_recording = True
                        frames.append(data)
                        silent_chunks = 0
                
                elif is_recording:
                    frames.append(data)
                    if rms < cls.RMS_THRESHOLD:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0
                    
                    current_chunks = len(frames)
                    
                    if silent_chunks > silence_limit_chunks:
                        cls.logger.info(f"🔇 检测到 {cls.SILENCE_TIMEOUT_SEC}s 静音，停止录制。")
                        break
                    
                    if current_chunks > max_record_chunks:
                        cls.logger.info(f"🎤 达到最大录制时长 ({cls.MAX_RECORD_SEC}秒)，停止录制。")
                        break

            if is_recording and frames:
                audio_data_bytes = b"".join(frames)
                audio_data_int16 = np.frombuffer(audio_data_bytes,dtype = np.int16)
                audio_data_f32 = audio_data_int16.astype(np.float32) / 32768.0
                
                duration = len(audio_data_f32) / RATE
                if duration < 0.5:
                    cls.logger.info(f"录音太短 ({duration:.2f}s)，忽略。")
                else:
                    cls.logger.info(f"录音完成，总时长 {duration:.2f} 秒。将数据放入队列。")
                    audio_queue.put(audio_data_f32)
            
        cls.stop_stream()

# --- 音频播放（增强错误处理） ---
def convert_to_target_format(src_file, dst_file, target_sr=TARGET_PLAY_SR, target_ch=TARGET_PLAY_CH):
    logger = logging.getLogger("AudioConverter")
    cmd = [
        "ffmpeg", "-y", "-i", src_file,
        "-ar", str(target_sr),
        "-ac", str(target_ch),
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        dst_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.warning(f"ffmpeg 转换警告: {result.stderr.decode()}")
    return result.returncode == 0

def play_audio_file(play_src_file, tts_gen_time, play_device=PLAY_DEVICE):
    logger = logging.getLogger("AudioPlayer")
    
    # 目标临时文件
    play_file = os.path.join("/dev/shm", f"tts_out_play_{int(time.time()*1000)}.wav")
    
    # 转换音频格式（保证 16k / 2ch / s16）
    if not convert_to_target_format(play_src_file, play_file):
        logger.warning("⚠️ 音频转换失败，将直接播放原文件")
        play_file = play_src_file

    # 读取时长，为 aplay 设置 timeout
    duration = sf.info(play_file).duration
    start_play = time.time()
    played = False

    # 只使用你已经确认能播放的命令
    cmd = ["aplay", "-D", play_device, play_file]
    logger.debug(f"播放命令: {' '.join(cmd)}")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f"🔉 TTS 播放耗时: {time.time() - start_play:.2f}s")
    
    # 播放短复位音（单声道，16000Hz，持续 0.1~0.2s）
    #------- 此RK3576板子特有步骤 ----------
    reset_file = "/dev/shm/audio_reset.wav"
    subprocess.run(["aplay", "-D", play_device, reset_file],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    # 删除临时文件
    if os.path.exists(play_file) and play_file != play_src_file:
        os.remove(play_file)

    return played
    
# --- 主函数（支持 LLM 流式 + TTS 实时合成，增强错误处理） ---
def main():
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start() 
    logger = logging.getLogger("MainPipeline")
    memory_monitor.log_memory("程序启动")
    logger.info("=== 智能助手启动 ===")
    
    # 初始化所有服务
    logger.info("--- 正在加载 ASR 服务 ---")
    asr_service = AsrService(
        mvn_path=ASR_MVN_PATH, embed_path=ASR_EMBED_PATH, rknn_path=ASR_RKNN_PATH,
        bpe_path=ASR_BPE_PATH, asr_dir=ASR_DIR
    )
    
    logger.info("--- 正在加载 LLM 服务 ---")
    llm_service = LlmService(script_path="/root/voice_assistant/run_llm.sh", idle_timeout=5)#1.2

    logger.info("--- 正在加载 TTS 服务 ---")
    tts_service = TtsService(
        encoder_path=TTS_ENCODER_PATH, decoder_path=TTS_DECODER_PATH, lexicon_path=TTS_LEXICON_PATH,
        token_path=TTS_TOKEN_PATH, g_bin_path=TTS_G_BIN_PATH
    )
 
    # 启动录音线程
    recorder_thread = threading.Thread(target=AudioRecorder.record_loop, daemon=True)
    recorder_thread.start()
    
    # 主循环
    while True:
        try:
            mem_pipeline_start = memory_monitor.get_memory_info()
            audio_data_f32 = audio_queue.get() 
            
            logger.info(f"\n--- 从队列获取到新的语音 (时长 {len(audio_data_f32) / RATE:.2f}s) ---")
            pipeline_start_time = time.time()
            memory_monitor.log_memory("新一轮推理开始")
            
            # ASR
            user_text, asr_time = asr_service.transcribe(audio_data_f32, language="zh", use_itn=True)
            if not user_text:
                logger.warning("⚠️ ASR 未返回有效结果，跳过本轮")
                continue
            logger.info(f"📝 听写结果: {user_text}")
            
            # 统计信息
            tts_total_time = 0.0
            sentence_count = 0
            first_sentence_time = None
            
            # 定义句子回调函数：每生成一个句子就进行 TTS 合成并立即播放
            def on_sentence_generated(sentence: str, sentence_time: float, is_first: bool):
                nonlocal tts_total_time, sentence_count, first_sentence_time

                sentence_clean = sentence.strip()
                for prefix in ["[ASR错误]", "[CMD]", "robot:", "assistant:"]:
                    if sentence_clean.lower().startswith(prefix.lower()):
                        sentence_clean = sentence_clean[len(prefix):].strip()

                if not sentence_clean:
                    return

                sentence_count += 1
                if is_first:
                    first_sentence_time = sentence_time

                #logger.info(f"🎵 开始合成第 {sentence_count} 句: {sentence_clean}")
                wav_path = f"/dev/shm/tts_stream_{sentence_count}_{int(time.time()*1000)}.wav"
                
                # 正确调用：两个参数
                success, tts_time, enc_time, dec_time = tts_service.synthesize_sentence(sentence_clean, wav_path)

                if success:
                    play_queue.put(wav_path)  # 放入播放队列
                    logger.info(f"✅ 第 {sentence_count} 句合成完成 ({tts_time:.3f}s)")
                    tts_total_time += tts_time
            
            # LLM 流式生成（每个句子会触发 on_sentence_generated）
            full_reply, llm_time, llm_first_sentence_time = llm_service.chat_stream(user_text, on_sentence_generated)
            
            logger.info(f"💬 LLM 完整回复: {full_reply}")
            logger.info(f"🧠 LLM 总耗时: {llm_time:.2f}s")
            logger.info(f"🗣️ TTS 总合成耗时: {tts_total_time:.2f}s (共 {sentence_count} 个句子)")
            
            pipeline_end_time = time.time()
            total_pipeline_time = pipeline_end_time - pipeline_start_time
            
            # 计算延迟优化效果
            logger.info("\n" + "~"*50)
            logger.info("--- 计时结果（流式优化） ---")
            logger.info(f"🎤 ASR 耗时: {asr_time:.3f}s")
            logger.info(f"⚡ 首句生成时间: {first_sentence_time if first_sentence_time else 0:.3f}s")
            logger.info(f"💡 首次响应延迟: {asr_time + (first_sentence_time if first_sentence_time else 0):.3f}s")
            logger.info(f"🧠 LLM 总耗时: {llm_time:.3f}s")
            logger.info(f"🗣️ TTS 总耗时: {tts_total_time:.3f}s")
            logger.info(f"🔥 整体推理总用时: {total_pipeline_time:.3f}s")
            logger.info("~"*50)
            
            mem_pipeline_end = memory_monitor.get_memory_info()
            pipeline_delta = memory_monitor.get_memory_delta(mem_pipeline_start)
            logger.info(f"--- 本轮推理内存变化: {memory_monitor.format_delta(pipeline_delta)}")
            memory_monitor.log_memory("本轮推理完成")
            
            gc.collect()
            logger.info("--- 流程结束,返回待命状态 ---\n")
            
        except KeyboardInterrupt:
            logger.info("用户中断程序")
            break
        except Exception as e:
            logger.error(f"❌ 未能识别有效语音: {e}", exc_info=True)
            logger.info("⚠️ 跳过本轮，继续下一轮...")
            gc.collect()
            continue

if __name__ == "__main__":
    main()
