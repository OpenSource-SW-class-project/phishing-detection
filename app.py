from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from lightgbm import LGBMClassifier
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import socket
import whois
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
from scipy.sparse import hstack
import pickle


app = Flask(__name__)

# 모델 및 전처리기 로드
lgbm_model = joblib.load('lgbm_model.pkl')
meta_model = joblib.load('stacking_meta_model.pkl')
lstm_model = load_model('lstm_model.keras')
vectorizer = joblib.load('vectorizer.pkl')  # TF-IDF 벡터
# 저장된 tokenizer 로드
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 560
TIMEOUT = 3

# 정규 표현식
ip_pattern = r"(?:\d{1,3}\.){3}\d{1,3}"
shorteningServices = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl\.com|..."

# 피처 추출 함수
def feature_extraction(url):
    features = {}
    features['IP_LIKE'] = int(1 if re.search(ip_pattern, url) else 0)
    features['AT'] = int(1 if "@" in url else 0)
    features['URL_Depth'] = int(len([segment for segment in urlparse(url).path.split('/') if segment]))
    features['Redirection'] = int(1 if url.rfind('//') > 7 else 0)
    features['Is_Https'] = int(1 if urlparse(url).scheme == 'https' else 0)
    features['TINY_URL'] = int(1 if re.search(shorteningServices, url) else 0)
    features['Check_Hyphen'] = int(1 if '-' in urlparse(url).netloc else 0)
    query_count = len(parse_qs(urlparse(url).query))
    features['Query'] = int(-1 if query_count == 0 else 0 if query_count == 1 else 1)

    try:
        domain_name = urlparse(url).netloc
        socket.setdefaulttimeout(TIMEOUT)
        domain_info = whois.whois(domain_name)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date is None or not isinstance(creation_date, datetime):
            features['Domain_Age'] = int(1)
        else:
            today = datetime.today()
            if creation_date <= today - timedelta(days=730):
                features['Domain_Age'] = int(-1)
            elif creation_date <= today - timedelta(days=365):
                features['Domain_Age'] = int(0)
            else:
                features['Domain_Age'] = int(1)
    except Exception:
        features['Domain_Age'] = int(1)

    try:
        expiration_date = domain_info.expiration_date
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        if isinstance(expiration_date, str):
            expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        if expiration_date is None or expiration_date - datetime.now() < timedelta(days=180):
            features['Domain_end'] = int(0)
        else:
            features['Domain_end'] = int(-1)
    except Exception:
        features['Domain_end'] = int(0)

    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        features['Mouseover'] = int(1 if soup.find(attrs={"onmouseover": True}) else 0)
    except Exception:
        features['Mouseover'] = int(-1)

    try:
        response = requests.get(url, allow_redirects=True, timeout=TIMEOUT)
        features['Web_forwards'] = int(1 if len(response.history) > 2 else 0)
    except Exception:
        features['Web_forwards'] = int(1)

    try:
        response = requests.get(url, timeout=TIMEOUT)
        soup = BeautifulSoup(response.content, 'html.parser')
        features['Hyperlinks'] = int(len(soup.find_all('a')))
    except Exception:
        features['Hyperlinks'] = int(-1)

    try:
        response = requests.get(url, timeout=TIMEOUT)
        original_domain = urlparse(url).netloc
        final_domain = urlparse(response.url).netloc
        features['Domain_Cons'] = int(1 if original_domain == final_domain else 0)
    except Exception:
        features['Domain_Cons'] = int(-1)


    # Tokenized_url은 문자열이므로 LightGBM 입력에서는 제외
    tokens = []
    parsed_url = urlparse(url)
    tokens.extend(parsed_url.netloc.split('.'))
    tokens.extend([segment for segment in parsed_url.path.split('/') if segment])
    query_tokens = parse_qs(parsed_url.query)
    for key, values in query_tokens.items():
        tokens.append(key)
        tokens.extend(values)
    features['Tokenized_url'] = ' '.join(tokens)  # TF-IDF 벡터화에만 사용

    return features


# URL을 구성 요소로 분리하는 함수
def tokenize_url(url):
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    netloc = parsed_url.netloc
    path = parsed_url.path
    
    # 구성 요소를 리스트로 결합
    url_parts = [scheme] + netloc.split('.') + path.split('/')
    
    # 빈 문자열 제거
    url_parts = [part for part in url_parts if part]
    
    return url_parts

def preprocess_url(url, tokenizer, max_len=560):
    # Tokenize and convert to string
    tokenized_url = tokenize_url(url)
    url_string = ' '.join(tokenized_url)

    # Encode to sequence for LSTM
    encoded_url = tokenizer.texts_to_sequences([url_string])
    padded_url = pad_sequences(encoded_url, maxlen=max_len)

    # Extract features for LGBM
    feature_dict = feature_extraction(url)
    tokenized_text = feature_dict.pop('Tokenized_url')

    # TF-IDF 벡터화
    tfidf_vector = vectorizer.transform([tokenized_text])

    # 숫자 피처
    feature_vector = np.array([float(v) for v in feature_dict.values()]).reshape(1, -1)

    # 결합된 입력
    combined_features = hstack([tfidf_vector, feature_vector])

    return padded_url, feature_vector, tfidf_vector  # 반환값 수정


# 루트 경로 (홈 페이지)
@app.route('/')
def home():
    return '''
        <h1>URL Prediction</h1>
        <form action="/predict" method="post">
            <label for="url">Enter URL:</label><br>
            <input type="text" id="url" name="url" placeholder="https://example.com" required><br><br>
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 사용자가 제출한 URL 가져오기
        url = request.form['url']

        # URL 전처리
        padded_url, feature_vector, tfidf_vector = preprocess_url(url, tokenizer, max_len)

        # Combine TF-IDF and numerical features
        combined_features = hstack([tfidf_vector, feature_vector])

        # Ensure shape matches LightGBM training data
        assert combined_features.shape[1] == lgbm_model.n_features_, "Feature size mismatch!"
                # LSTM 예측
        lstm_proba = lstm_model.predict(padded_url).flatten()[0]

        # LightGBM 예측
        lgbm_proba = lgbm_model.predict_proba(combined_features)[:, 1][0]

        # 스태킹 모델 예측
        stacked_input = np.array([[lgbm_proba, lstm_proba]])
        final_proba = meta_model.predict_proba(stacked_input)[:, 1][0]
        final_prediction = 1 if final_proba >= 0.5 else 0

        # 결과 반환
        return f"""
            <h1>Prediction Result</h1>
            <p><strong>URL:</strong> {url}</p>
            <p><strong>LSTM Probability:</strong> {lstm_proba:.2f}</p>
            <p><strong>LGBM Probability:</strong> {lgbm_proba:.2f}</p>
            <p><strong>Final Probability:</strong> {final_proba:.2f}</p>
            <p><strong>Is Legitimate:</strong> {'Yes' if final_prediction == 1 else 'No'}</p>
            <a href="/">Back to Home</a>
        """
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p><a href='/'>Back to Home</a>"
    
if __name__ == '__main__':
    app.run(debug=True)