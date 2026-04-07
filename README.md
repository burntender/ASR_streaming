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
- `transcribe.py`: 単発文字起こし CLI
- `.env.example`: ホスト側パスの設定例

## 使い方

1. 必要なら設定ファイルを作る

```bash
cp .env.example .env
```

2. イメージをビルド

```bash
docker compose build
```

3. 音声ファイルを `./audio` に置いて実行

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
- `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` が合わない場合は `.env` の `BASE_IMAGE` を変えてください
- この構成はまず CLI 用です。必要なら次に FastAPI や OpenAI 互換 API を足せます
