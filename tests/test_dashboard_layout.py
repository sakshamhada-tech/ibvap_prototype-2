from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}


class _DashboardParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack: list[tuple[str, set[str]]] = []
        self.alert_ancestors: list[set[str]] | None = None
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if "id" in attributes:
            self.ids.append(attributes["id"])
        if "panel--log" in classes:
            self.alert_ancestors = [ancestor_classes for _tag, ancestor_classes in self.stack]
        if tag not in VOID_ELEMENTS:
            self.stack.append((tag, classes))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                return


def test_alert_panel_is_not_nested_in_the_scrollable_evidence_rail():
    parser = _DashboardParser()
    parser.feed((ROOT / "dashboard/static/index.html").read_text(encoding="utf-8"))
    parser.close()

    assert parser.alert_ancestors is not None
    assert any("console__body" in classes for classes in parser.alert_ancestors)
    assert not any("console__rail" in classes for classes in parser.alert_ancestors)
    assert len(parser.ids) == len(set(parser.ids))


def test_desktop_dashboard_reserves_a_separate_alert_column():
    stylesheet = (ROOT / "dashboard/static/style.css").read_text(encoding="utf-8")

    assert "minmax(280px, 0.9fr) minmax(280px, 0.9fr)" in stylesheet
    assert ".panel--log" in stylesheet
    assert "grid-column: 1 / -1" in stylesheet
