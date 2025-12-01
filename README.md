# NeuroCT-Bench
Reproducible implementation of CNN and Transformer models for CT-based stroke detection.

Place the dataset folder as `dataset/Normal` and `dataset/Stroke` or change `--data_dir` / `DATA_DIR`.

TensorFlow scripts (CNNs): run `python vgg16_train.py --data_dir dataset --epochs 100 --out_prefix vgg16`
PyTorch scripts (Transformers): run `DATA_DIR=dataset NUM_EPOCHS=100 BATCH_SIZE=32 python swin_train.py`

Each script outputs: `<prefix>_best.h5` / `.pth`, `<prefix>_val_predictions.csv`, and `<prefix>_metrics.csv` (or CSV of predictions for PT).
