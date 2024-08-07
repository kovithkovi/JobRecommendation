import os
import pickle
from PyPDF2 import PdfReader
import docx2txt
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage


def clean_resume(resume_text):
    resume_text = re.sub('http\\S+\\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#\\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub('@\\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
    resume_text = re.sub('\\s+', ' ', resume_text)  # remove extra whitespace
    return resume_text


def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def index(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['resume']
        # Save the uploaded file to the media directory
        fs = FileSystemStorage(location='media')
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        print(f"Uploaded file saved to: {file_path}")

        if uploaded_file.name.endswith('.docx'):
            resume_text = docx2txt.process(file_path)
        elif uploaded_file.name.endswith('.pdf'):
            resume_text = extract_text_from_pdf(file_path)
        else:
            fs.delete(filename)
            return HttpResponse("Unsupported file format. Please upload a .docx or .pdf file.")
        cleaned_resume = clean_resume(resume_text)
        print(cleaned_resume[:500])
        df = pd.read_csv(
            'C:/records/web development/Intervita/djangoEnv/jobReco/jobReco/UpdatedResumeDataSet111.csv')
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(df['Resume'])
        resume_vector = vectorizer.transform([cleaned_resume])
        with open('C:/records/web development/Intervita/djangoEnv/jobReco/jobReco/job_recommendation_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        predicted_probabilities = model.predict_proba(resume_vector)
        categories = model.classes_
        top_categories = np.argsort(predicted_probabilities[0])[
            ::-1][:5]
        print("Top Predicted Categories:")
        for idx in top_categories:
            print(f"{categories[idx]}")
        fs.delete(filename)
        return render(request, 'upload.html', {'top_categories': [categories[idx] for idx in top_categories]})

    return render(request, 'index.html')
