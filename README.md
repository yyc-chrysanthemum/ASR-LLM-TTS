# RK3588/3576
🍊 Orange Pi 5 Plus

测试设备: Orange Pi 5 Plus (RK3588)

系统总内存: ~8 GB (7934.67 MB)

测试时间: 2025-11-18 - 2025-11-19 

核心模块: ASR (SenseVoiceSmal), LLM (Qwen2.5-1.5B), TTS (MeloTTS)

Purple Pi OH2

测试设备: Purple Pi OH2 (RK3576)

系统总内存: ~4 GB (3895.01 MB)

核心模块: ASR (SenseVoiceSmal), LLM (Qwen2.5-0.5B), TTS (MeloTTS)

## 快速开始
需将模型转换成rknn格式
参考链接

https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main

https://huggingface.co/happyme531/SenseVoiceSmall-RKNN2

https://huggingface.co/lovemefan/SenseVoice-onnx/tree/main

https://huggingface.co/ThomasTheMaker/SenseVoiceSmall-RKNN2

https://huggingface.co/happyme531/MeloTTS-RKNN2

https://huggingface.co/3ib0n/Qwen2.5-14B-Instruct-rkllm

## 测试(板子上的路径)
#### 1.整体运行
cd /root/voice_assistant/

python q.py

#### 2.LLM模型转换
cd /root/rknn-llm/rknn-llm-release-v1.2.2/examples/multimodal_model_demo/export/

python export_rkllm.py

#### 3.ASR模型
cd /home/orangepi/rknn-asr/runtime/RK3588/Linux/librknn_api/include/

python ./sensevoice_rknn.py --audio_file output.wav

如果使用自己的音频文件测试发现识别不正常，你可能需要提前将它转换为16kHz, 16bit, 单声道的wav格式。

ffmpeg -i input.mp3 -f wav -acodec pcm_s16le -ac 1 -ar 16000 output.wav

RKNN模型转换
你需要提前安装rknn-toolkit2, 测试可用的版本为2.3.3a25，可从https://console.zbox.filez.com/l/I00fc3 下载(密码为"rknn")

下载或转换onnx模型,可以从 https://huggingface.co/lovemefan/SenseVoice-onnx 下载到onnx模型.

模型文件应该命名为'sense-voice-encoder.onnx', 放在转换脚本所在目录.

python convert_rknn.py ./sense-voice-encoder.onnx

#### 4.TTS模型
cd /home/orangepi/rknn-tts/MeloTTS-RKNN2/

python melotts_rknn.py -s "The text you want to generate."

RKNN模型转换

python convert_rknn.py
