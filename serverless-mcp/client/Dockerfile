FROM python:3.10-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV REPO_URL=client

WORKDIR $REPO_URL

COPY . .
COPY env.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run","streamlit_app.py"]
