FROM python:3.9

WORKDIR /src

COPY ./api/requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8000

CMD ["python", "main.py"]

