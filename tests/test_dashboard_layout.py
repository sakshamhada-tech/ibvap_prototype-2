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


def test_visual_surfaces_use_astra_i_without_renaming_internal_configuration():
    index = (ROOT / "dashboard/static/index.html").read_text(encoding="utf-8")
    login = (ROOT / "dashboard/static/login.html").read_text(encoding="utf-8")
    pipeline = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    server = (ROOT / "server.py").read_text(encoding="utf-8")

    assert "<title>Astra I — Border Surveillance Console</title>" in index
    assert '<span class="title-mark">Astra I</span>' in index
    assert "<title>Sign in — Astra I</title>" in login
    assert '<div class="login-mark">Astra I</div>' in login
    assert 'f"Astra I | {mode_text}' in pipeline
    assert 'cv2.imshow("Astra I - Border Surveillance"' in main
    assert 'title="Astra I Dashboard"' in server

    assert "<title>IBVAP" not in index
    assert ">IBVAP<" not in index
    assert "Sign in — IBVAP" not in login
    assert ">IBVAP<" not in login
    assert 'f"IBVAP | {mode_text}' not in pipeline
    assert 'cv2.imshow("IBVAP - Border Surveillance"' not in main
    assert 'title="IBVAP Dashboard"' not in server
