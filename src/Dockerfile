FROM python:3.9

WORKDIR /src

COPY ./api/requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]

