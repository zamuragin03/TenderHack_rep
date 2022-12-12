# Решение команды Mister MISISter на Tender Hack MSK
### Предсказание количества участников и изменения цены услуги в ходе котировочной сессии на Портале поставщиков

В качестве основных сервисов сейчас предоставленны:

- Модель для предсказания ключевых метрик (количества участников котировочной сессии и отклонение от начальной цены). Модель состоит из 3 моделей CatBoost, результатом работы которых являются метапризнаки для модели линейной регрессии.
- Rest api сервер, для комуникации модели и фронтэнда.
- Фронтэнд реализованный на TS с ипользованием React для взаимодейтсвия с 
моделью.

Реализация Rest api приложения расположена в директории `src`

Запускать приложение рекомендуется с помошью docker compose.

## Run the app

    $ docker compose up --build -d

# REST API

The REST API to the example app is described below.

## Tender Predict Endpoints

### Get predict by tender description

`POST /calculate`

### Request Body
```json
{
  "session_name": "string",
  "OKPD": "string",
  "KPGZ": "string",
  "Region": "string",
  "start_price": 0,
  "date": "string",
  "INN": "string"
}
```

### Response

```json
{
  "percent": 0,
  "participants": 0
}
```

## Get predict by .xlsx file

### Request

`POST /calculate_csv`

### Response
Ссылка на файл
```json
"string"
```
