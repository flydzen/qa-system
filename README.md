# LLM QA System

## Архитектура:

![architecture](./imgs/miro.png)

Логические узлы выделены в докер контейнеры, вот их описание

### fastapi_app

Основное приложение, является точкой входа. 

- Ручка `GET /` отдает этот документ в html
- Ручка `POST /ask` отвечает на заданный вопрос, дополняя материалами.

Приложение асинхронное и не имеет cpu-нагруженных частей, все ожидание является io-bound, их 3:
- Запрос `POST /encode`
- Поиск в базе
- Запрос `POST /llm_ask`

При работе с базой данных используется ThreadPoolExecutor на несколько потоков.
На данный момент при каждом вызове ручки открывается новое синхронное подключение к базе, но в перспективе вместо открытия будет браться подключение из пулла коннекшенов.

### fastapi_app_llm 

Вспомогательное приложение. Является синхронным и cpu-нагруженным.
Выполняет вычисления на моделях (настоящей и моке). Модель работает на cpu, а не gpu для упрощения

- Ручка `GET /llm_ask` отдает стрим с ответами на prompt от мока llm
- Ручка `POST /encode` использует SentenceTransformer ([модель](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) для кодирования входящего сообщения и отдает вектор. 

Обе ручки принимают батч запросов.

### setup_milvus

Одноразовый контейнер, который загрузит данные из [датасета](https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles/data) в базу данных

Логически имеет 3 части:
1) Прочитать датасет из файла
2) Собрать энкодинги для строк из датасета
3) Загрузить в БД данные с их энкодингами

### milvus-standalone

Векторная база данных, специализирующаяся на быстром поиске при больших объемах данных.

В этом приложении использован standalone способ деплоя, так как это является достаточно простым способом и подходит под требования, 
так как, согласно документации, standalone применяется для случаев хранения до 100 миллионов векторов.

В дальнейшем, при масштабировании можно будет перейти на распределенную систему, не меняя при этом логику приложения.

Также стоит отметить, что в секции [setup_milvus](#setup_milvus) создается партиция по ключу `topic`, что позволит в разы ускорить поиск.

### prometheus, loki, grafana

- prometheus служит для сбора метрик. Сами метрики создаются пакетом `prometheus-fastapi-instrumentator` поверх fastapi приложения,
но при необходимости можно добавить свои
- loki служит единой точкой сбора логов из всех приложений
- grafana служит для отображения метрик и логов.

Пример экрана мониторинга

![grafana](imgs/screenshot_grafana.png)

## Инструкция по запуску

### Приложение

1. 
   - Option 1: run `docker compose up setup --build` to create and fill database
   - Option 2: [download](https://drive.google.com/file/d/1zPxLk0wFRi03VD5L0TNZUzJ0XlWHR4cM/view?usp=sharing) db volume and extract it to [./volumes/](https://github.com/flydzen/qa-system/tree/main/volumes) directory
2. run `docker compose up grafana --build -d` to run grafana 
3. run `docker compose up app --build` to run main application

Приложение будет доступно по адресу http://127.0.0.1:8000. 
Swagger по http://127.0.0.1:8000/docs

Мониторинги доступны по http://127.0.0.1:3000

### Тестирование

Есть тесты для `app_llm`, они независимы и их можно запустить в любой момент

Для `app` написаны только интеграционные тесты, их можно запустить только после выполнения команды
`docker compose up app_llm --build` 

тесты запускаются командой `pytest` из директории модуля.

### Нагрузочное тестирование

Также можно запустить нагрузочное тестирование. Для этого запустите приложение и выполните команду

```commandline
locust -f ./locust_testing/locustfile.py
```

После чего укажите host: `http://127.0.0.1:8000` и другие параметры в интерфейсе