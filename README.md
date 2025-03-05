# **ChurnPredictANN - Customer Churn Prediction using ANN**  

## 📌 Overview  
**ChurnPredictANN** is an Artificial Neural Network (ANN) model designed to predict customer churn in a banking dataset. It analyzes customer demographics, account details, and transaction history to determine whether a customer is likely to leave the bank.  

## 📊 Dataset  
The model is trained on the **Churn_Modelling.csv** dataset, which includes:  
- **Features**: Customer ID, Credit Score, Age, Geography, Gender, Balance, Tenure, etc.  
- **Target Variable**: `Exited` (1 if the customer left, 0 otherwise).  

## 🚀 Model Architecture  
The neural network consists of:  
- **Input Layer**: 12 input features (after preprocessing).  
- **Two Hidden Layers**:  
  - **6 neurons each**, with **ReLU activation**.  
- **Output Layer**:  
  - **1 neuron**, with **Sigmoid activation** (binary classification).  

## 🔧 Preprocessing & Feature Engineering  
- **Dropped unnecessary columns** (`RowNumber`, `CustomerId`, `Surname`).  
- **Encoded categorical features**:  
  - `Gender` → Label Encoding.  
  - `Geography` → One-Hot Encoding.  
- **Feature Scaling**: Standardized numerical values using `StandardScaler`.  
- **Train-Test Split**: 80% training, 20% testing.  

## ⚡ Model Training  
- **Loss Function**: `binary_crossentropy` (for binary classification).  
- **Optimizer**: `adam` (adaptive learning rate).  
- **Metrics**: `accuracy`.  
- **Batch Size**: 64  
- **Epochs**: 130  

## 📈 Performance Evaluation  
- **Confusion Matrix**: Measures true positives, true negatives, false positives, and false negatives.  
- **Accuracy Score**: Evaluates model performance on the test set.  

## 🛠 Installation & Usage  
### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/ChurnPredictANN.git
cd ChurnPredictANN
