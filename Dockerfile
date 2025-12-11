FROM python:3.13

WORKDIR /app

COPY app.py cli.py pagerank_calculation.py README.md requirments.txt ./

COPY hobbies_test.ini likes_test.ini people_test.txt ./

COPY data ./data
COPY assets ./assets
COPY lib ./lib
COPY output ./output
COPY __pycache__ ./__pycache__

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirments.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
