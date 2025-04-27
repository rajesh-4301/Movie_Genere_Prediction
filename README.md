# 🎬 Movie Genre Prediction using Machine Learning

This project focuses on predicting the genre(s) of a movie based on its metadata using machine learning techniques. Given features like the movie title, plot, keywords, or other descriptive attributes, the system predicts the most probable genre(s) — such as Action, Comedy, Drama, etc.

The goal is to build a model that can automatically classify unseen movies into genres, helping in tasks like movie recommendation, categorization, and content analysis.

## ✨ Project Highlights

- Exploratory Data Analysis (EDA) on movie metadata
- Text preprocessing and feature engineering (e.g., TF-IDF vectorization)
- Building and training machine learning models for multi-label classification
- Evaluation using appropriate metrics (Accuracy, F1-Score, etc.)
- Visualization of results and insights

## 📂 Dataset

The project uses a movie dataset containing information such as:

- Movie Title
- Plot or Description
- Keywords
- Genres (labels)

📥 **Dataset Access**:  
The dataset used for this project is available at the following Google Drive link:  
**[Dataset]:https://drive.google.com/drive/folders/14cxWC6RsR3eW2NwGQW5zB2AVrJSCvhDY?usp=drive_link**

## 🛠️ Tech Stack

- Python 🐍
- Jupyter Notebook 📓
- Scikit-learn (for machine learning models)
- Pandas (for data manipulation)
- Numpy (for numerical operations)
- Matplotlib & Seaborn (for visualization)
- Natural Language Processing (NLP) tools for text preprocessing

## 🔥 Models Used

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)


## 📈 Workflow

1. **Data Loading & Exploration**
   - Load the dataset and understand the structure
   - Visualize genre distributions

2. **Preprocessing**
   - Clean text fields (remove punctuation, lowercase, stopwords removal)
   - Feature extraction using TF-IDF Vectorizer
   - Encode target genres for multi-label classification

3. **Model Building**
   - Train multiple machine learning models
   - Tune hyperparameters for better performance

4. **Evaluation**
   - Assess models using classification metrics
   - Visualize confusion matrices

5. **Prediction**
   - Predict genres for new or unseen movie data

## 📊 Results

- Achieved an accuracy: Test Accuracy: 0.09357933579335793 on the test set.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Movie_genere_Prediction.git
   cd Movie_Genre_Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `Movie_Genere_Prediction.ipynb` and execute the cells.




