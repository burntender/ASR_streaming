# VibeVoice-ASR Container

ローカルに配置した `microsoft/VibeVoice-ASR-HF` をコンテナから使うための最小構成です。

## 前提

- Docker と `docker compose` が使える
- GPU コンテナが使えること
- モデル本体がホスト側にあること
  - 既定: `./models/vibeVoice`

## ファイル

- `Dockerfile`: 実行イメージ
- `compose.yaml`: GPU とボリュームマウント設定
- `api.py`: OpenAI 互換の `/v1/audio/transcriptions` API
- `asr_core.py`: CLI と API が共有する VibeVoice-ASR 推論処理
- `transcribe.py`: 単発文字起こし CLI
- `.env.example`: ホスト側パスの設定例

## OpenAI 互換 API

1. 必要なら設定ファイルを作る

```bash
cp .env.example .env
```

2. イメージをビルド

既定は ARM / CPU でも動く `python:3.12-slim-bookworm` を使います。

```bash
docker compose build
```

3. API サーバを起動

```bash
docker compose up
```

`.env` の `PORT` に公開されます。`CONTAINER_PORT` は通常 `8080` のままでOKです。以下は `PORT=8083` の例です。

アップロードされた音声は API 側で `ffmpeg` を使って 16kHz / mono の wav に変換してから ASR に渡します。主な対象は `wav`, `m4a`, `mp3` です。

```bash
curl http://127.0.0.1:8083/v1/audio/transcriptions \
  -F file=@./data/audio/sample.wav \
  -F model=vibevoice-asr \
  -F language=ja-JP
```

レスポンス:

```json
{
  "text": "文字起こし結果"
}
```

syaberukun 側の設定例:

```json
{
    "asr": {
      "provider": "openai_compatible",
      "base_url": "http://127.0.0.1:8083/v1",
      "model": "vibevoice-asr",
      "api_key_env": null,
      "request_timeout_ms": 120000
  }
}
```

## CLI

音声ファイルを `./data/audio` に置いて実行します。

```bash
docker compose run --rm vibevoice-asr \
  --audio /data/audio/sample.wav \
  --format parsed
```

プロンプトを渡す例:

```bash
docker compose run --rm vibevoice-asr \
  --audio /data/audio/sample.wav \
  --prompt "About VibeVoice and OpenAI" \
  --format transcription_only
```

## メモ

- モデルはイメージに含めず、ホストから読み取り専用マウントします
- `TRANSFORMERS_OFFLINE=1` が既定なので、ローカルモデルだけを使います
- ARM / CPU ホストでは既定の `.env.example` のまま使えます。PyTorch は CPU wheel index から入れます
- ARM / CPU ホストでは `ASR_DISABLE_MKLDNN=1` が既定です。CPU推論中に `Xbyak::Error` が出る環境向けの回避設定です
- x86_64 + NVIDIA GPU で CUDA 版を使う場合は、`.env` で `BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` と `INSTALL_TORCH=0` を指定し、`docker compose -f compose.yaml -f compose.gpu.yaml build` / `docker compose -f compose.yaml -f compose.gpu.yaml up` を使ってください
- API は初回リクエスト時にモデルを読み込み、以後は同じプロセス内で再利用します
