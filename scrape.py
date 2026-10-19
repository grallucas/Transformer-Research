"""Build a small text corpus by crawling English Wikipedia breadth-first."""

from collections import deque
from pathlib import Path
import re
import time
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from markdownify import markdownify
import requests


STARTING_URL = "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"
N_BREADTH = 10
N_ARTICLES = 10
SAVE_TO_FILE = "wikipedia_corpus.txt"

REQUEST_DELAY_SECONDS = 0.25
REQUEST_TIMEOUT_SECONDS = 20
USER_AGENT = "TransformerResearchEducationalCorpus/0.1 (local educational project)"

REFERENCE_SECTION_NAMES = {
    "bibliography",
    "citations",
    "external links",
    "further reading",
    "notes",
    "references",
    "sources",
}


def normalize_article_url(url: str, base_url: str = STARTING_URL) -> str | None:
    """Return a canonical-looking Wikipedia article URL, or None if invalid."""
    absolute_url = urljoin(base_url, url)
    parsed = urlsplit(absolute_url)

    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() != "en.wikipedia.org":
        return None
    if not parsed.path.startswith("/wiki/"):
        return None

    title = unquote(parsed.path.removeprefix("/wiki/")).replace(" ", "_")
    if not title or ":" in title:
        return None

    normalized_path = "/wiki/" + quote(title, safe="/()_,-.'")
    return urlunsplit(("https", "en.wikipedia.org", normalized_path, "", ""))


def article_links(soup: BeautifulSoup, page_url: str) -> list[str]:
    """Collect unique, valid article links in their document order."""
    content = soup.select_one("#mw-content-text")
    if content is None:
        return []

    links: list[str] = []
    seen: set[str] = set()
    for anchor in content.find_all("a", href=True):
        if "new" in anchor.get("class", []):
            continue
        normalized = normalize_article_url(anchor["href"], page_url)
        if normalized is not None and normalized not in seen:
            seen.add(normalized)
            links.append(normalized)
    return links


def clean_formula(latex: str) -> str:
    """Turn Wikipedia's verbose TeX representation into one compact line."""
    latex = re.sub(r"\s+", " ", latex).strip()
    wrapper = re.fullmatch(r"\{\\(?:display|text)style\s+(.+)\}", latex)
    if wrapper is not None:
        latex = wrapper.group(1).strip()
    return latex


def replace_math(content: BeautifulSoup) -> None:
    """Replace MathML, TeX, and image fallbacks with one inline TeX formula."""
    for math_element in content.select(".mwe-math-element"):
        math = math_element.find("math")
        image = math_element.find("img")
        annotation = math_element.find("annotation", attrs={"encoding": "application/x-tex"})

        latex = None
        if annotation is not None:
            latex = annotation.get_text()
        elif math is not None:
            latex = math.get("alttext")
        elif image is not None:
            latex = image.get("alt")

        if latex:
            math_element.replace_with(f" ${clean_formula(latex)}$ ")
        else:
            math_element.decompose()


def remove_reference_sections(content: BeautifulSoup) -> None:
    """Remove reference-oriented sections through the next peer heading."""
    headings = list(content.find_all(re.compile(r"^h[1-6]$")))
    for heading in headings:
        if heading.parent is None:
            continue
        section_name = heading.get_text(" ", strip=True).casefold()
        if section_name not in REFERENCE_SECTION_NAMES:
            continue

        heading_level = int(heading.name[1])
        section_start = heading.parent if "mw-heading" in heading.parent.get("class", []) else heading
        sibling = section_start.find_next_sibling()
        while sibling is not None:
            next_sibling = sibling.find_next_sibling()
            next_heading = sibling if re.fullmatch(r"h[1-6]", sibling.name or "") else sibling.find(re.compile(r"^h[1-6]$"))
            if next_heading is not None and int(next_heading.name[1]) <= heading_level:
                break
            sibling.decompose()
            sibling = next_sibling
        section_start.decompose()


def article_markdown(soup: BeautifulSoup) -> tuple[str, str]:
    """Extract an article title and cleaned Markdown body."""
    title_element = soup.select_one("#firstHeading")
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if title_element is None or content is None:
        raise ValueError("page does not contain a recognizable Wikipedia article")

    title = title_element.get_text(" ", strip=True)
    replace_math(content)
    remove_reference_sections(content)
    for element in content.select(
        "script, style, table, figure, sup.reference, .mw-ref, "
        "[typeof~='mw:Extension/ref'], .mw-editsection, .navbox, "
        ".vertical-navbox, .reflist, .mw-references-wrap, ol.references, "
        ".shortdescription, .hatnote, .Inline-Template, .citation-needed"
    ):
        element.decompose()

    body = markdownify(str(content), heading_style="ATX", bullets="-").strip()
    body = re.sub(r"[ \t]+([,.;:!?])", r"\1", body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    if not body:
        raise ValueError("article body is empty")
    return title, body


def scrape(
    starting_url: str = STARTING_URL,
    n_breadth: int = N_BREADTH,
    n_articles: int = N_ARTICLES,
    save_to_file: str | Path = SAVE_TO_FILE,
) -> int:
    """Crawl Wikipedia breadth-first and return the number of articles saved."""
    if n_breadth < 1:
        raise ValueError("n_breadth must be at least 1")
    if n_articles < 1:
        raise ValueError("n_articles must be at least 1")

    first_url = normalize_article_url(starting_url)
    if first_url is None:
        raise ValueError("starting_url must be an English Wikipedia article URL")

    queue = deque([first_url])
    queued = {first_url}
    saved_urls: set[str] = set()
    saved_count = 0
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    output_path = Path(save_to_file)
    with output_path.open("w", encoding="utf-8") as output:
        while queue and saved_count < n_articles:
            requested_url = queue.popleft()
            try:
                response = session.get(requested_url, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                final_url = normalize_article_url(response.url, requested_url)
                soup = BeautifulSoup(response.text, "html.parser")
                canonical = soup.select_one('link[rel="canonical"][href]')
                if canonical is not None:
                    final_url = normalize_article_url(canonical["href"], requested_url)
                if final_url is None or final_url in saved_urls:
                    continue

                links = article_links(soup, final_url)
                title, body = article_markdown(soup)
            except (requests.RequestException, ValueError) as error:
                print(f"Skipping {requested_url}: {error}")
                continue

            saved_urls.add(final_url)
            saved_count += 1
            output.write(f"# {title}\n\nSource: {final_url}\n\n{body}\n\n")
            output.write("\n---\n\n")
            print(f"[{saved_count}/{n_articles}] {title}")

            added = 0
            for link in links:
                if link in queued or link in saved_urls:
                    continue
                queue.append(link)
                queued.add(link)
                added += 1
                if added == n_breadth:
                    break

            if saved_count < n_articles:
                time.sleep(REQUEST_DELAY_SECONDS)

    print(f"Saved {saved_count} articles to {output_path}")
    return saved_count


if __name__ == "__main__":
    scrape()
