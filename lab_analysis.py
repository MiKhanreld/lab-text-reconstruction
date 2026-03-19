from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

try:
    import networkx as nx
except ImportError:
    nx = None


BASE_DIR = Path(__file__).resolve().parent
TEXT_PATH = BASE_DIR / "5.txt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BAR_COLOR = "darkorange"
MORPH_COLOR = "cadetblue"
DIALOGUE_COLOR = "firebrick"
DIVERSITY_COLOR = "slateblue"
BIGRAM_COLOR = "goldenrod"
NETWORK_NODE_COLOR = "lightsalmon"
NETWORK_EDGE_COLOR = "dimgray"

WORDCLOUD_WIDTH = 1600
WORDCLOUD_HEIGHT = 900
WORDCLOUD_BG = "white"

TOP_FREQUENCY_LIMIT = 30
BIGRAM_MIN_FREQ = 8

STOPWORDS = {'и', 'в', 'во', 'не', 'что', 'он', 'она', 'оно', 'они', 'на', 'я', 'ты', 'мы', 'вы',
             'с', 'со', 'как', 'а', 'то', 'все', 'его', 'ее', 'их', 'но', 'да', 'к', 'у', 'же',
             'бы', 'по', 'за', 'от', 'до', 'из', 'или', 'ли', 'уж', 'ну', 'так', 'это', 'этот',
             'эта', 'эти', 'того', 'тем', 'при', 'для', 'под', 'над', 'тут', 'там', 'где',
             'когда', 'потом', 'снова', 'теперь', 'очень', 'только', 'уже', 'нет', 'был', 'была',
             'были', 'быть', 'есть', 'если', 'чем', 'чего', 'чтоб', 'чтобы', 'мне', 'тебе', 'ему',
             'ей', 'нас', 'вас', 'себя', 'свой', 'свои', 'мой', 'моя', 'мое', 'мои', 'твой',
             'твоя', 'твое', 'твои', 'наш', 'ваш', 'сам', 'сама', 'само', 'сами', 'какой',
             'какая', 'какие', 'который', 'которая', 'которые', 'тоже', 'почти', 'после',
             'перед', 'через', 'между', 'один', 'одна', 'одно'}


# Предобработка текста

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\ufeff", "")


def clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_chapters(text: str) -> list[str]:
    parts = re.split(r"^\s*[IVXLCDM]+\s*$", text, flags=re.M)
    return [part.strip() for part in parts if part.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яА-Я]+", text.lower())


def get_morph_tools():
    try:
        from pymorphy3 import MorphAnalyzer
        morph = MorphAnalyzer()
        return "pymorphy3", morph
    except Exception:
        try:
            from pymorphy2 import MorphAnalyzer
            morph = MorphAnalyzer()
            return "pymorphy2", morph
        except Exception:
            return None, None


def get_normalizer():
    morph_name, morph = get_morph_tools()

    if morph is not None:
        def normalize(token: str) -> str:
            return morph.parse(token)[0].normal_form
        return morph_name, morph, normalize

    def normalize(token: str) -> str:
        return token.lower()

    return "lowercase_only", None, normalize


def preprocess_for_counts(text: str, normalize) -> tuple[list[str], list[str]]:
    tokens = tokenize(text)
    normalized = [normalize(token) for token in tokens]
    filtered = [
        token for token in normalized
        if len(token) > 2 and token not in STOPWORDS
    ]
    return tokens, filtered


# Анализ

def save_frequency_tables(filtered_units: list[str]) -> pd.DataFrame:
    freq_df = pd.DataFrame(
        Counter(filtered_units).most_common(TOP_FREQUENCY_LIMIT),
        columns=["unit", "freq"]
    )
    freq_df.to_csv(RESULTS_DIR / "top_units.csv", index=False)
    return freq_df


def analyze_morphology(tokens: list[str], morph) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if morph is None:
        return None, None

    pos_counter = Counter()
    noun_case_counter = Counter()

    for token in tokens:
        parsed = morph.parse(token)[0]
        pos = parsed.tag.POS

        if pos:
            pos_counter[pos] += 1

        if pos == "NOUN" and parsed.tag.case:
            noun_case_counter[str(parsed.tag.case)] += 1

    pos_df = pd.DataFrame(pos_counter.most_common(), columns=["pos", "freq"])
    pos_df.to_csv(RESULTS_DIR / "pos_distribution.csv", index=False)

    case_df = pd.DataFrame(
        noun_case_counter.most_common(),
        columns=["case", "freq"]
    )
    case_df.to_csv(RESULTS_DIR / "noun_cases.csv", index=False)
    return pos_df, case_df


def make_dialogue_table(chapters: list[str]) -> pd.DataFrame:
    rows = []

    for i, chapter in enumerate(chapters, start=1):
        lines = [line.strip() for line in chapter.splitlines() if line.strip()]
        dialogue_lines = [
            line for line in lines
            if line.startswith("--") or line.startswith("-") or line.startswith("—")
        ]

        rows.append({
            "chapter": i,
            "all_lines": len(lines),
            "dialogue_lines": len(dialogue_lines),
            "dialogue_share": len(dialogue_lines) / len(lines) if lines else 0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "dialogue_share.csv", index=False)
    return df


def make_lexical_diversity_by_chapter(chapters: list[str], normalize) -> pd.DataFrame:
    rows = []

    for i, chapter in enumerate(chapters, start=1):
        tokens = tokenize(chapter)
        normalized = [normalize(token) for token in tokens]
        content_units = [
            token for token in normalized
            if len(token) > 2 and token not in STOPWORDS
        ]

        unique_units = len(set(content_units))
        all_units = len(content_units)
        diversity = unique_units / all_units if all_units else 0

        rows.append({
            "chapter": i,
            "content_units": all_units,
            "unique_units": unique_units,
            "lexical_diversity": diversity,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "lexical_diversity_by_chapter.csv", index=False)
    return df


def extract_bigrams(units: list[str], min_freq: int = BIGRAM_MIN_FREQ) -> list[tuple[tuple[str, str], int]]:
    bigrams = list(zip(units, units[1:]))

    counts = Counter(
        pair for pair in bigrams
        if pair[0] != pair[1] and pair[0] not in STOPWORDS and pair[1] not in STOPWORDS
    )

    return [(pair, freq) for pair, freq in counts.most_common() if freq >= min_freq][:30]


def make_collocation_table(filtered_units: list[str]) -> pd.DataFrame:
    bigrams = extract_bigrams(filtered_units)
    edge_rows = [
        {"source": left, "target": right, "weight": weight}
        for (left, right), weight in bigrams
    ]
    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(RESULTS_DIR / "collocations.csv", index=False)
    return edge_df


# Визуализация

def make_wordcloud(freq_df: pd.DataFrame) -> None:
    frequencies = dict(zip(freq_df["unit"], freq_df["freq"]))
    wc = WordCloud(
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        background_color=WORDCLOUD_BG
    )
    wc.generate_from_frequencies(frequencies)

    plt.figure(figsize=(14, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_wordcloud.png", dpi=200, bbox_inches="tight")
    plt.close()


def make_top_barplot(freq_df: pd.DataFrame) -> None:
    plot_df = freq_df.head(20).iloc[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(plot_df["unit"], plot_df["freq"], color=BAR_COLOR)
    plt.xlabel("Частота")
    plt.ylabel("Лемма или словоформа")
    plt.title("Топ-20 значимых единиц", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_top_units.png", dpi=200, bbox_inches="tight")
    plt.close()


def make_lexical_diversity_plot(diversity_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(
        diversity_df["chapter"],
        diversity_df["lexical_diversity"],
        marker="o",
        color=DIVERSITY_COLOR
    )
    plt.xlabel("Глава")
    plt.ylabel("Лексическое разнообразие")
    plt.title("Лексическое разнообразие по главам", fontsize=14)
    plt.xticks(diversity_df["chapter"])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_lexical_diversity.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def make_dialogue_plot(dialogue_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(
        dialogue_df["chapter"],
        dialogue_df["dialogue_share"],
        marker="o",
        color=DIALOGUE_COLOR
    )
    plt.xlabel("Глава")
    plt.ylabel("Доля строк с прямой речью")
    plt.title("Распределение прямой речи по главам", fontsize=14)
    plt.xticks(dialogue_df["chapter"])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_dialogue_share.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def make_collocation_visual(edge_df: pd.DataFrame) -> None:
    if edge_df.empty:
        return

    if nx is None:
        plot_df = edge_df.copy()
        plot_df["pair"] = plot_df["source"] + " " + plot_df["target"]
        plot_df = plot_df.sort_values("weight").tail(15)

        plt.figure(figsize=(13, 8))
        plt.barh(plot_df["pair"], plot_df["weight"], color=BIGRAM_COLOR)
        plt.xlabel("Частота")
        plt.ylabel("Биграмма")
        plt.title("Частотные биграммы", fontsize=14)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "05_collocation_network.png",
                    dpi=200, bbox_inches="tight")
        plt.close()
        return

    graph = nx.Graph()
    for _, row in edge_df.iterrows():
        graph.add_edge(row["source"], row["target"], weight=row["weight"])

    plt.figure(figsize=(13, 8))
    pos = nx.spring_layout(graph, seed=42, k=1.1)
    weights = [graph[u][v]["weight"] for u, v in graph.edges()]

    nx.draw_networkx_nodes(graph, pos, node_size=1000,
                           node_color=NETWORK_NODE_COLOR)
    nx.draw_networkx_labels(graph, pos, font_size=9)
    nx.draw_networkx_edges(
        graph,
        pos,
        width=[w / 3 for w in weights],
        edge_color=NETWORK_EDGE_COLOR
    )

    plt.title("Сеть частотных биграмм", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "05_collocation_network.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def make_pos_barplot(pos_df: pd.DataFrame | None) -> None:
    if pos_df is None or pos_df.empty:
        return

    plot_df = pos_df.head(10).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["pos"], plot_df["freq"], color=MORPH_COLOR)
    plt.xlabel("Частота")
    plt.ylabel("Часть речи")
    plt.title("Распределение частей речи", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "06_pos_distribution.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def save_summary(
    chapters: list[str],
    tokens: list[str],
    filtered_units: list[str],
    normalizer_name: str,
    morph_name
) -> None:
    summary_lines = [
        f"Глав: {len(chapters)}",
        f"Словоформ: {len(tokens)}",
        f"Значимых единиц после фильтрации: {len(filtered_units)}",
        f"Тип нормализации: {normalizer_name}",
        f"Морфологический анализ доступен: {'да' if morph_name is not None else 'нет'}",
        f"networkx установлен: {'да' if nx is not None else 'нет'}",
    ]

    (RESULTS_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def save_run_info(normalizer_name: str, morph_name) -> None:
    lines = [f"Нормализация выполнена через: {normalizer_name}"]

    if morph_name is None:
        lines.append(
            "Лемматизация и морфологический анализ не выполнены, потому что pymorphy2/pymorphy3 не найдены.")
        lines.append("Вместо лемм использованы словоформы в нижнем регистре.")
        lines.append("Для полной версии лучше установить pymorphy3.")
    else:
        lines.append(f"Морфологический разбор выполнен через: {morph_name}")

    if nx is None:
        lines.append(
            "Библиотека networkx не найдена, поэтому вместо сети коллокаций построена диаграмма биграмм.")

    (RESULTS_DIR / "run_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not TEXT_PATH.exists():
        print(f"Ошибка: файл не найден -> {TEXT_PATH}")
        print("Положите 5.txt в ту же папку, где находится lab_analysis.py")
        return

    raw_text = read_text(TEXT_PATH)
    clean = clean_text(raw_text)
    chapters = split_chapters(clean)

    normalizer_name, morph, normalize = get_normalizer()
    morph_name = None if morph is None else normalizer_name

    tokens, filtered_units = preprocess_for_counts(clean, normalize)

    freq_df = save_frequency_tables(filtered_units)
    pos_df, case_df = analyze_morphology(tokens, morph)
    dialogue_df = make_dialogue_table(chapters)
    diversity_df = make_lexical_diversity_by_chapter(chapters, normalize)
    collocation_df = make_collocation_table(filtered_units)

    make_wordcloud(freq_df)
    make_top_barplot(freq_df)
    make_lexical_diversity_plot(diversity_df)
    make_dialogue_plot(dialogue_df)
    make_collocation_visual(collocation_df)
    make_pos_barplot(pos_df)

    save_summary(chapters, tokens, filtered_units, normalizer_name, morph_name)
    save_run_info(normalizer_name, morph_name)

    if case_df is not None and not case_df.empty:
        case_df.to_csv(RESULTS_DIR / "noun_cases.csv", index=False)

    print("Готово. Результаты сохранены вот туть ->> results.")


if __name__ == "__main__":
    main()
