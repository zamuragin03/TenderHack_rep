FROM python:3.9

WORKDIR /src

COPY ./requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

COPY ./main.py ./
COPY ./schemas.py ./
COPY ["./short_train (2).xlsx", "./"]
COPY ./Tender_model.py ./

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

