#StayUnbiased

**INTRODUCTION**

This project is a web application that uses machine learning and keyword-based detection to identify fake news and biased content. It provides users with a simple interface to paste news headlines or paragraphs and receive predictions on their authenticity (real or fake) and bias detection. The model is trained using datasets of Malaysian news articles, categorized as "real" or "fake," and it detects biases related to race, religion, gender, and politics.


**Our key features include:**

- Bias detection: detects bias based on keywords associated with race, religion, gender, and politics.
- Fake news detection: Predict if the news is real or fake using a machine learning model.
- Multi-language support: this web app supports both english and bahasa melayu


**Files in this project:**

finalapp.py  -> # Main Streamlit app file for the web interface
scrape_fake_news.py -> # Script to scrape and create fake news dataset
scrape_real_news.py -> # Script to scrape and create real news dataset
combining_datasets.py -> # Script to combine real and fake news datasets
malaysia_fake_news_dataset.csv -> # Fake news dataset
real_news_malaysia.csv -> # Real news dataset (manually created for this model)
fake_news_malaysia.csc -> # Fake news datset (also manually created)
malaysia_fake_news_dataset.csv # Combined dataset to be used for machine learning
model_fake_news.pkl -> # Saved machine learning model for fake news detection
vectorizor_fake_news.pkl -> # Saved vectorizer used for transforming text for prediction
ModelCreation.ipynb -> # Jupyter notebook file for machine learning model creation
style.css -> # Custom styles for the Streamlit app
requirements.txt -> # python dependencies
README.md -> # project documentation


**SETUP INSTRUCTIONS**

1. Clone the repository: 
  git clone <your-repository-url>
  cd <repository-directory>

2. Create a virtual environment using venv:
  python -m venv venv
  On Windows use `venv\Scripts\activate`

3.Install dependencies:
  pip install -r requirements.txt

4. Run the application by starting the streamlit web application:
  streamlit run app/finalapp.py

Ensure that the style.css file is present in the app/ directory to customize the look and feel of the Streamlit app.


**FUTURE IMPLEMENTATION**

-Implementing an API to ensure our results are accurate and fact-checked
-Implementing a chrome extension feature where users can detect bias on websites they visit
-Adding more languages for multilingual support


**More in-depth explanation of this project's development**
We created a custome dataset by combinging real headlines scraped from trusted malaysian news sources with fake or misleading headlines, ensuring local relevance and diversity in language and topics.

We manually collected over 200 news headlines from reputable sources:
- [The Star](https://www.thestar.com.my/)
- [Malay Mail](https://www.malaymail.com/)
- [New Straits Times](https://www.nst.com.my/)

Using web scraping, we extracted headlines from these articles.

As for the fake news collection, we collected them from:
- Archived news sources
- [Sebenarnya.my] (https://sebenarnya.my/)
- headlines adapted to resemble real ones but manually modified to introduce bias or false claims

Then we merged the two datasets:
both the real and fake headline datasets were cleaned and merged, combined into a single dataset.

Machine learning model (Jupyter Notebook)
We build and trained a fake news classifier using jupyter notebook with the following steps:
 1.  Text Preprocessing
- Converted all headlines to lowercase
- Removed stopwords and punctuation
- Used `TF-IDF Vectorizer` to transform text into numerical features

 2.  Model Training
- Model: `Linear Support Vector Machine (LinearSVC)`
- Split data: `80%` training, `20%` testing
- Achieved **~87% accuracy** on the test set
- Exported the trained model using `joblib`

python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import dump

 Example pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

X_train, X_test, y_train, y_test = train_test_split(headlines, labels, test_size=0.2)
pipeline.fit(X_train, y_train)

dump(pipeline, 'fake_news_model.joblib')



**Built during ImagineHack 2025 by team Orchid
For questions, reach out at this github or matoukyara@gmail.com**
