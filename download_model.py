from huggingface_hub import snapshot_download

snapshot_download(
    "SparkAudio/Spark-TTS-0.5B",
    local_dir="pretrained_models/Spark-TTS-0.5B"
)
print("Model downloaded to pretrained_models/Spark-TTS-0.5B")
