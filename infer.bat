@echo off
REM Activate your venv or conda env here if needed
REM call .\venv\Scripts\activate.bat
REM or conda activate sparktts

python -m cli.inference --text "Hello world from SparkTTS!" --device 0 --save_dir "output_audio" --model_dir pretrained_models/Spark-TTS-0.5B
pause
