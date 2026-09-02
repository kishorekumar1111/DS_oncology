# Stage 02 — Deep Learning (Mini Demo)

**Mission:** Go from numbers to instincts — visually spot malignant cell
patterns in pathology tiles, and project ctDNA level 3 months ahead from a
sequence of readings.

**Simplified for classroom pace:** Real CNNs/LSTMs need big libraries (TensorFlow/
PyTorch) and minutes of training. Here we use a small neural network (`MLPClassifier`
from scikit-learn) on tiny 8x8 synthetic "pathology tiles" to teach the exact SAME
workflow (tile in -> malignant/benign out) in seconds. The real large-scale build
(Day 3) will use an actual CNN + LSTM/Transformer.

## Run order

| Script | Role | What it does |
|---|---|---|
| `01_data_engineer.py` | Data Engineer | Generates + labels mini synthetic pathology tiles, does simple augmentation |
| `02_eda_engineer.py` | EDA Engineer | Checks what pixel regions the brightness pattern comes from (saliency stand-in) |
| `03_dl_engineer.py` | DL Engineer | Trains a small neural net (CNN stand-in) + a trend predictor (LSTM stand-in) |
| `04_evaluation_engineer.py` | Evaluation Engineer | Confusion matrix — checks benign-vs-malignant confusion |
| `05_integration_engineer.py` | Integration Engineer | Combines vision + trend score into one alert API |
