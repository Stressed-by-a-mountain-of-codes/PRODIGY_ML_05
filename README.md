# 🍔🍜 Food Image Classifier and Calorie Intake Calculator 

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Culinary_fusion_pizza.jpg/320px-Culinary_fusion_pizza.jpg" width="200" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Sushi_platter.jpg/320px-Sushi_platter.jpg" width="200" />
</p>

<p align="center">
  <b>A Deep Learning project to classify 101 types of food images using Transfer Learning on the Food-101 dataset.</b><br>
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Keras-High%20Level%20API-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square" />
</p>

---

## 🧾 Project Summary

| 📌 Attribute           | 📊 Details                                 |
|------------------------|---------------------------------------------|
| 📅 Completed On        | 2025-06-02 14:27:37                          |
| 🧠 Model               | CNN with Transfer Learning (e.g., MobileNetV2) |
| 📁 Dataset             | Food-101 (101 food categories)              |
| 🖼️ Total Images        | 101,000                                      |
| 🎯 Final Accuracy      | 51.27% (on 1024 test images)                |
| 🏆 Baseline Accuracy   | ~1% (random guessing)                       |
| 🚀 Improvement         | ~51x better than random                     |

---

## 📊 Performance Metrics

| Class         | Precision | Recall |
|---------------|-----------|--------|
| 🍕 Pizza      | 52%       | 58%    |
| 🍣 Sushi      | 49%       | 50%    |
| 🍔 Burger     | 54%       | 53%    |

> Achieved **51.27% accuracy** on a very challenging 101-class dataset!

---

## 🛠 Features & Workflow

### 🔄 Data Pipeline
- Preprocessing, normalization, and augmentation
- Train/validation/test splits

### 🤖 Model Architecture
- Used **MobileNetV2** as the base
- Custom classification head with Dense layers
- Dropout added to prevent overfitting

### 📚 Training
- 10 epochs using Adam optimizer
- EarlyStopping and ModelCheckpoint callbacks
- Augmented training data for better generalization

---

## 💾 Files & Folders

| File                          | Description                            |
|-------------------------------|----------------------------------------|
| `food_classifier_model.h5`    | Trained model saved for reuse          |
| `food_class_names.pkl`        | Python pickle of class label names     |
| `test_model.py`               | Script to test on new images           |
| `train_model.py`              | Script used for training the model     |
| `data/food-101/`              | Root directory of the Food-101 dataset |

---

## 🖼️ Example Predictions

<p align="center">
  <img src="https://i.imgur.com/WIWJWVr.jpeg" width="250" />
  <img src="https://i.imgur.com/BvoCfLC.jpeg" width="250" />
  <img src="https://i.imgur.com/7iY5JDn.jpeg" width="250" />
</p>

> Predictions: `Pizza`, `Ice Cream`, `Sushi`

---

## ▶️ How to Run

### 🔧 Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/food101-classifier.git
cd food101-classifier

Here's your provided content **properly formatted as Markdown** for your `README.md`:

---

````markdown
## 📦 Step 2: Install dependencies

```bash
pip install -r requirements.txt
````

---

## 🗃️ Step 3: Prepare the dataset

Download **Food-101 dataset** and extract it to:

```
data/food-101/
```

---

## 🚀 Step 4: Train the model (Optional)

```bash
python train_model.py
```

---

## 🧪 Step 5: Run predictions

```bash
python test_model.py --image path/to/your/image.jpg
```

---

## 📁 Project Structure

```
food101-classifier/
├── data/
│   └── food-101/
├── models/
│   └── food_classifier_model.h5
├── src/
│   ├── train_model.py
│   └── test_model.py
├── utils/
│   └── food_class_names.pkl
├── README.md
└── requirements.txt
```

---

## 🌟 Highlights

✅ Trained on **101 food categories**
✅ Used **Transfer Learning (MobileNetV2)**
✅ Saved model and class index for inference
✅ Handled **data augmentation and preprocessing**
✅ Performance \~**51x better than random guess**

---

## 💡 Future Improvements

* 🔧 Hyperparameter tuning (learning rate, batch size)
* 🧠 Try ResNet, EfficientNet, or Inception models
* 🧹 Remove mislabeled images
* 🌐 Deploy using Streamlit/Flask
* 🔁 Increase training epochs

---

## 🙌 Acknowledgements

* 📚 Dataset: [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
* 🧠 Frameworks: **TensorFlow**, **Keras**
* 🛠️ Tools: **NumPy**, **Matplotlib**, **Scikit-learn**, **OpenCV**

---


---

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Love-red?style=for-the-badge" />
</p>
```

---
