FROM python:3.10.4

WORKDIR /usr/src/app

COPY DataFormatting.py .
COPY HeartDiseaseAnalysis.py .
COPY LogisticRegression.py .
COPY hd_data.csv .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "./HeartDiseaseAnalysis.py"]
