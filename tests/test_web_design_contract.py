from pathlib import Path


ROOT = Path(__file__).parents[1]
WEB_SOURCE = ROOT / "web" / "src"
WEB_PUBLIC = ROOT / "web" / "public"


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


def test_brand_and_crawler_assets_are_declared_and_shippable():
    index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

    for fragment in (
        'rel="canonical" href="https://dubsync.onrender.com/"',
        'rel="icon" href="/favicon.svg"',
        'rel="manifest" href="/site.webmanifest"',
        'property="og:title"',
        'property="og:image" content="https://dubsync.onrender.com/brand/dubsync-social.png"',
        'name="twitter:card" content="summary_large_image"',
        'type="application/ld+json"',
        'src="/theme-init.js"',
    ):
        assert fragment in index

    for relative_path in (
        "favicon.svg",
        "site.webmanifest",
        "robots.txt",
        "sitemap.xml",
        "theme-init.js",
        "brand/dubsync-mark.svg",
        "brand/dubsync-icon-192.png",
        "brand/dubsync-icon-512.png",
        "brand/dubsync-apple-touch.png",
        "brand/dubsync-social.png",
    ):
        asset = WEB_PUBLIC / relative_path
        assert asset.is_file(), f"Missing public brand or SEO asset: {relative_path}"
        assert asset.stat().st_size > 0
