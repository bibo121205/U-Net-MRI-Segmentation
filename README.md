# U-Net Brain Tumor Segmentation

A deep learning project implementing U-Net architecture for brain tumor segmentation using the LGG Segmentation Dataset.

## Dataset

The project uses the LGG Segmentation Dataset containing brain MR images with manual FLAIR abnormality segmentation masks. The dataset includes:

- **110 patients** from The Cancer Genome Atlas (TCGA) lower-grade glioma collection
- **3-channel MR images** in TIFF format of the same size (pre-contrast, FLAIR, post-contrast sequences)
- **True masks** for FLAIR abnormality
- **Patient metadata** including genomic clusters and clinical information (which were not used much in this project)

## Project Structure

```
U-Net_Segmentation/
├── kaggle_3m/                 # Dataset directory
│   ├── data.csv              # Patient metadata and genomic information
│   ├── README.md             # Dataset documentation
│   └── TCGA_*/               # Patient-specific folders containing images and masks
├── main.ipynb                # Main Jupyter notebook with U-Net implementation
├── requirements.txt          # Python dependencies
├── unet.h5                   # Trained U-Net model weights
└── README.md                 # This file
```

## Methodology

### 1. Data Loading
- File paths for MR images and masks are extracted and organized using a pandas DataFrame.
- Dataset is split into training, validation and test sets.

### 2. Exploratory Data Analysis (EDA) and Preprocessing
- Pie charts and histograms are used to analyze class distribution.
- Example MRI slices and their masks are visualized to assess the quality and variability of data.
- Images and masks are resized (256,256), augmented then normalized.
- Data augmentation is performed using `ImageDataGenerator` to increase robustness.

### 3. Model Architecture
- A U-Net model is implemented using TensorFlow and Keras.
- The architecture includes downsampling and upsampling paths with skip connections, following the original paper's architechture.
- The model returns a segmentation mask as its prediction/output.

### 4. Training and Evaluation
- Training was conducted on kaggle using GPU P100.
- Training is monitored using metrics such as accuracy, IoU coefficient, and Dice coefficient.
- The best model is saved using `ModelCheckpoint`.

## Requirements

- Python 3.x
- TensorFlow 2.15.0
- NumPy >= 1.23.0
- Pandas >= 1.5.0
- Matplotlib >= 3.6.0
- OpenCV >= 4.6.0
- Scikit-learn >= 1.1.0
- Seaborn >= 0.12.0
- TQDM >= 4.64.0

## Installation

1. Clone the repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/bibo121205/U-Net-MRI-Segmentation.git
cd U-Net-MRI-Segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure the dataset is properly organized in the `kaggle_3m/` directory
2. Open `main.ipynb` in Jupyter Notebook or JupyterLab
3. Run the cells sequentially to:
   - Load and preprocess the dataset
   - Train the U-Net model
   - Evaluate segmentation performance
   - Generate predictions on test images
4. You can access the pre-trained U-Net model weights [here](https://drive.google.com/file/d/17Qed2G9bMsAAIxlhEc1lD6nyseH3DKXb/view?usp=drive_link).

To download programmatically:

```bash
pip install gdown
gdown --id 17Qed2G9bMsAAIxlhEc1lD6nyseH3DKXb
```