# Практическое задание №3

- Определить какой-нибудь внешний источник получения данных и способ получения этих данных (http, curl, wget, API, SQL, SparQL, ...) 
- Поставить задачу для алгоритма машинного обучения, выбрать модель и метрику 
- Создать инфраструктуру, например, виртуальные машины virtualbox, установить и настроить для работы необходимое программное обеспечение, в том исле airflow и mlflow, а также venv для организации работы виртуального окружения 
- Создать python скрипты для 
    - Получения данных из внешнего источника 
    - Преобразования данных 
    - Формирования рабочего набора данных для обучения (train) и тестирования (test) модели 
    - Обучения модели на тренировочных (train) данных и ее сохранения 
    - Загрузки модели и проверки качества ее работы на тестовых (test) данных 
- Добавить код airflow, позволяющий создавать и запускать на регулярной основе описанные операции проекта. 
- Добавить код mlflow, позволяющий мониторить ход выполнения конвейера, сохранять и анализировать полученных артифакты. 


Задача заключается в сборе данных и их обработке, а также последующем обучении модели ML. Airflow используется для автоматизации операций, а MLFlow для мониторинга процессов.

В данной работе мы собираем информацию о рейтинге канала по колличеству лайков на видео из доступных открытых источников, а именно youtube.ru

1. Создание файловой структуры и виртуального окружения

Также было создано виртуальное откружение с версией питона 3.8
![Снимок экрана от 2023-11-18 11-16-52](https://github.com/Marakya/xflow/assets/113238801/ad97989b-3146-482c-964f-ae1c0c10166f)

2. Airflow

   Установка и выполнение необходимых команд:
   ```
   pip install "apache-airflow[celery]==2.7.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.3/constraints-3.8.txt"
   export AIRFLOW_HOME=/home/xflow/project/airflow
   airflow db init
   ```
   Запуск
   
   ![Снимок экрана от 2023-11-18 11-21-49](https://github.com/Marakya/xflow/assets/113238801/d0c3b120-28ea-4155-ab15-e24bd3b7c8b7)

   Запускаем в новом окне Scheduler для выполнения команд по расписанию и отслеживания новых скрпитов в папке dags. Также перед запуском необходимо установить значение переменной окружения AIRFLOW_HOME
   
   ![Снимок экрана от 2023-11-18 11-30-14](https://github.com/Marakya/xflow/assets/113238801/8bfce6b9-09c5-44a2-b320-746a791fee5e)

   В папке dags создаем скрипт (youtube_comments_score.py), который с использованием BashOperator вызывает выполнение python скриптов (в папке scripts) и объединяет эти операторы в конвейер оператором >>.
   
   ![image](https://github.com/Marakya/mlops_xflow/assets/113238801/53697202-2f12-403a-acae-ecf7077c0427)


   В итоге, запустив конвейер операций на выполнение, мы увидим успешно выполненную задачу.

   ![2023-11-17 (4)](https://github.com/Marakya/xflow/assets/113238801/61c5cbe0-de53-4098-af8e-541bb6d8f7c6)

   ![2023-11-17 (5)](https://github.com/Marakya/xflow/assets/113238801/75f62ea6-9f0f-49cc-b977-1ab070ea9744)

4. Mlflow

   Организуем мониторинг выполнения всех операций проекта с использованием MLFlow.

   Необходимый код для установки:
   ```
   pip install mlflow
   export MLFLOW_REGISTRY_URI=mlflow
   ```
   
   ![Снимок экрана от 2023-11-18 11-32-28](https://github.com/Marakya/xflow/assets/113238801/e1ab2b4b-02de-4c92-8715-e9dca6754e17)

   Также в скриптах необходимо установить нужные нам объекты под наблюдение mlflow
   
   Поскольку мы сохранили код python скрипта как артифакт функцией mlflow.log_artifact, мы можем посмотреть этот код в графическом интерфейсе mlflow.
   
   Например, в скрипте train_model.py, мы сохранили следующее
   
   ![image](https://github.com/Marakya/mlops_xflow/assets/113238801/c92ff840-48fe-45e2-a1e8-2094dd98d2dd)

   ![2023-11-17 (2)](https://github.com/Marakya/xflow/assets/113238801/827a6fda-2ebd-4718-ac76-628db313d8f6)
   
   ![2023-11-17 (3)](https://github.com/Marakya/xflow/assets/113238801/9e1178bc-28c7-47d9-9b39-f2002beb539e)
   
   ![2023-11-17 (6)](https://github.com/Marakya/xflow/assets/113238801/e4ffa1bd-d945-4628-bd9e-d5892074a3dd)

   Здесь также можно найти артефакты 
   ![2023-11-17 (1)](https://github.com/Marakya/xflow/assets/113238801/d83f5bfc-f21c-4ea9-a04e-2e89a11c9287)

   
