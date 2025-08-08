# 🍽️ Restaurant Review Sentiment Analyzer

A machine learning-powered web app that analyzes the sentiment of restaurant reviews using Natural Language Processing (NLP) and displays whether the feedback is **Positive**, **Negative**, or **Neutral** — with confidence scores and helpful visualizations.

---

## 🔍 About the Project

This project uses a **Support Vector Machine (SVM)** classifier trained on real-world restaurant reviews to determine the sentiment behind user-submitted feedback. It also integrates **TextBlob** to refine neutral sentiments and deliver more human-aligned predictions.

The model is deployed in a clean and responsive **Streamlit** interface, where users can:
- Input restaurant reviews
- Get real-time sentiment predictions
- View confidence scores and suggestions
- See dynamic pie chart summaries

---

## 🧠 How It Works

1. **Text Preprocessing**
   - Raw user input is cleaned using regex (removing non-alphabetic characters)
   - Converted to lowercase
   - Transformed using **TF-IDF vectorizer** with `max_features=1500`

2. **Model Prediction**
   - A trained **SVM classifier** predicts the review as:
     - `1` → Positive
     - `0` → Negative

3. **TextBlob-Based Sentiment Adjustment**
   After the model's prediction, a secondary check is done using **TextBlob polarity** (ranging from `-1` to `+1`) to improve accuracy:
   ```python
   if -0.2 <= blob_score <= 0.2:
       pred = 2  # Neutral
   elif pred == 1 and blob_score < -0.2:
       pred = 0  # Adjust to Negative
   elif pred == 0 and blob_score > 0.3:
       pred = 1  # Adjust to Positive
   ```
   #### ✅ Interpretation:
   - **Neutral**: If polarity is between -0.2 and +0.2, the review is considered neutral.
   - **Adjustment**:
     - If model says *Positive* but TextBlob detects strong *Negative* → switch to Negative.
     - If model says *Negative* but TextBlob detects strong *Positive* → switch to Positive.

4. **Streamlit Web App**
   - Allows users to:
     - Enter reviews
     - Analyze sentiment instantly
     - Get customized feedback
     - View sentiment pie charts and confidence progress

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sakshamsharma-code/IBM-CSRBOX-Internship.git
cd restaurant-review-sentiment-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional — already included)
```bash
python train_model.py
```

> This will create two files:
> - `sentiment_model.pkl`
> - `tfidf_vectorizer.pkl`

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 📦 Requirements

Main dependencies:

```
streamlit
scikit-learn
pandas
numpy
matplotlib
textblob
```

Install them using:
```bash
pip install -r requirements.txt
```

---

## 📁 Files Description

| File/Folder                          | Description |
|-------------------------------------|-------------|
| `train_model.py`                    | Trains and saves SVM model and TF-IDF vectorizer |
| `app.py`                            | Streamlit app for review input and sentiment analysis |
| `sentiment_model.pkl`               | Trained SVM sentiment classification model |
| `tfidf_vectorizer.pkl`              | Fitted TF-IDF vectorizer for input transformation |
| `Restaurant_Reviews.tsv`            | Dataset used for training the model (tab-separated format) |
| `review_log.csv`                    | (Optional) Log of user-submitted reviews with sentiment |
| `Sentiment_Analysis_of_Restaurants.ipynb` | Jupyter notebook used for exploration and prototyping |
| `requirements.txt`                  | Python dependencies needed to run the app |

---

## ✨ Features

- ✅ Real-time ML sentiment classification
- 📊 Confidence visualization
- 🧠 TextBlob-based sentiment adjustment
- 📈 Pie chart distribution of result
- 💬 Smart suggestions for negative reviews
- 🧪 Optional retraining of model

---

## 🙋‍♂️ Author

Developed By **Saksham Sharma** and members of Team **Syntax Squad**
> 👨‍💻 Project submitted for the **IBM SkillsBuild Internship Program**

---

## 📜 License

This project is open-source and free to use for learning, educational, and non-commercial purposes.

