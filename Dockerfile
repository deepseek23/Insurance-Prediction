# base image
FROM python:3.11
# work dir
WORKDIR /app

# copy

COPY . /app

# run 
RUN pip install -r requirements.txt

# port 
EXPOSE 8501

# command

CMD ["streamlit", "run", "insurance_app.py", "--server.port=8501", "--server.address=0.0.0.0"]