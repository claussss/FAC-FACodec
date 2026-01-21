# FAC-FACodec: Controllable Zero-Shot Foreign Accent Conversion with Factorized Speech Codec

[![Paper](https://img.shields.io/badge/arXiv-2510.10785-b31b1b.svg)](https://arxiv.org/abs/2510.10785)
[![Demo](https://img.shields.io/badge/Demo-Page-blue)](https://claussss.github.io/accent_control_demo/)


This repository provides the code for the FAC-FACodec project. We train a non-autoregressive denoising transformer conditioned on phoneme features to perform accent conversion using FACodec representations.

## 🎧 Demo

Listen to samples on our [**Demo Page**](https://claussss.github.io/accent_control_demo/)

## 📄 Paper

**[FAC-FACodec](https://arxiv.org/abs/2510.10785)**  
*Accepted at ICASSP 2026*

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Controlled_AC
```

### 2. Clone Amphion (FACodec implementation)

Clone Amphion at the same directory level as this repository:
```bash
cd ..
git clone https://github.com/open-mmlab/Amphion.git
cd Controlled_AC
```

### 3. Create Environment

```bash
conda create -n facodec python=3.11.11
conda activate facodec
pip install -r requirements.txt
```

### 4. Download Checkpoints and Stats

Download the pretrained model and normalization statistics from:

📥 **[Google Drive](https://drive.google.com/drive/folders/1Pnq_XV5VA_hcIpoOYfbSnLZFA3GKGk1C?usp=sharing)**

The download contains two directories:
- `stats/` - Normalization statistics (mean/std for zc1 and zc2)
- `weights/` - Model checkpoint

Place them in your project:
```
Controlled_AC/
├── checkpoints/
│   └── <your_checkpoint>.pt    # from weights/
├── stats/
│   ├── mean_zc1_indx.pt
│   ├── std_zc1_indx.pt
│   ├── mean_zc2_indx.pt
│   └── std_zc2_indx.pt
└── ...
```

### 5. Configuration

Copy the example config and update paths:
```bash
cp FACodec_AC/config.py.example FACodec_AC/config.py
```

Edit `FACodec_AC/config.py` with your local paths.

## Inference

See **`inference_demo.ipynb`** for a step-by-step guide on running inference with the pretrained model.

The notebook demonstrates:
- Loading FACodec encoder/decoder
- Loading the denoising transformer model
- Performing accent conversion on audio samples
- Configurable diffusion sampling parameters

## Training

### Dataset Preparation

1. **Download LJSpeech**
   ```bash
   wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
   tar xfj LJSpeech-1.1.tar.bz2
   ```

2. **Generate FACodec Dataset**
   ```bash
   python create_facodec_dataset.py
   ```
   This creates `.pt` files with FACodec indices (`prosody_indx`, `zc1_indx`, `zc2_indx`, etc.)

3. **Generate Phone Forced Alignment Data**
   ```bash
   python create_phone_dataset.py
   ```

### Train the Model

```bash
python train.py
```

Training progress and checkpoints are saved to `tensorboard/` and `checkpoints/` respectively.

## Project Structure

```
Controlled_AC/
├── FACodec_AC/
│   ├── config.py.example    # Configuration template (copy to config.py)
│   ├── models.py            # Denoising transformer model
│   ├── dataset.py           # Dataset and dataloader utilities
│   └── utils.py             # Utility functions
├── create_facodec_dataset.py # FACodec feature extraction
├── create_phone_dataset.py   # Phoneme forced alignment
├── train.py                  # Training script
├── inference_demo.ipynb      # Inference demonstration
└── requirements.txt
```

## Citation

*Citation information will be available upon publication.*

<!-- 
@inproceedings{controlled_ac_2026,
  title={FAC-FACodec: Controllable Zero-Shot Foreign Accent Conversion with Factorized Speech Codec},
  booktitle={ICASSP 2026},
  year={2026}
}
-->

## License

This project is released under the MIT License.
