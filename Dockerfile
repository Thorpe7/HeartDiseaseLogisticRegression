FROM python:3.10.4

WORKDIR /usr/src/app

COPY hd_data.csv .
COPY requirements.txt .
COPY DataFormatting.py .
COPY LogisticRegression.py .
COPY HeartDiseaseAnalysis.py .


RUN pip install -r requirements.txt


CMD ["python", "HeartDiseaseAnalysis.py"]