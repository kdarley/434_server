FROM python:3.9-slim

WORKDIR /server

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
