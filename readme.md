# Lab: Text Reconstruction and Analysis

## Описание

В работе проводится автоматический анализ художественного текста (5.txt).

Основные этапы:
- предобработка текста;
- частотный анализ;
- морфологический анализ;
- коллокации;
- дополнительные метрики;
- визуализация.

Интерпретация вынесена в report.md.

---

## Структура проекта

lab_text_reconstruction/

5.txt

lab_analysis.py

README.md

report.md

тапание по llm.md

для воспроизводимости.txt

results/

---

## Запуск

python lab_analysis.py

После запуска появится папка results.

---

## Анализ

- частоты слов
- части речи
- биграммы
- лексическое разнообразие
- доля прямой речи

---

## Визуализация

- облако слов
- частотный график
- графики по главам
- коллокации
- части речи

---

## Results

### Изображения
- 01_wordcloud.png
- 02_top_units.png
- 03_lexical_diversity.png
- 04_dialogue_share.png
- 05_collocation_network.png
- 06_pos_distribution.png

### Таблицы
- top_units.csv
- pos_distribution.csv
- noun_cases.csv
- dialogue_share.csv
- lexical_diversity_by_chapter.csv
- collocations.csv

### Прочее
- summary.txt
- run_info.txt

---

## Примечания

- pymorphy3 нужен для морфологии
- networkx нужен для сети
- все файлы генерируются автоматически

---

## Интерпретация

См. report.md

---

## LLM

Описание в файле "тапание по llm.md"

---

## Воспроизводимость

См. "для воспроизводимости.txt"
