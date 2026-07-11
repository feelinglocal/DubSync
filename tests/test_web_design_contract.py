from pathlib import Path


ROOT = Path(__file__).parents[1]
WEB_SOURCE = ROOT / "web" / "src"


def test_web_uses_approved_palette_and_inter_without_decorative_gradients():
    css = (WEB_SOURCE / "styles.css").read_text(encoding="utf-8").lower()

    assert "--color-primary: #006cff" in css
    assert "--color-ink: #091717" in css
    assert "--color-accent: #fff04b" in css
    assert "--color-canvas: #ffffff" in css
    assert "font-family: 'inter variable'" in css
    assert "gradient(" not in css
    assert "transition: all" not in css
    for old_color in ("#042f34", "#16232b", "#b5f2db", "#48b89a", "#ffc933"):
        assert old_color not in css


def test_visible_web_copy_has_no_dash_flourishes():
    files = [WEB_SOURCE / "App.tsx", *(WEB_SOURCE / "components").glob("*.tsx")]
    visible_source = "\n".join(path.read_text(encoding="utf-8") for path in files)

    assert "\N{EN DASH}" not in visible_source
    assert "\N{EM DASH}" not in visible_source
