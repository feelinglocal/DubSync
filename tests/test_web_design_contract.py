import json
from pathlib import Path
from struct import unpack
from xml.etree import ElementTree


ROOT = Path(__file__).parents[1]
WEB_SOURCE = ROOT / "web" / "src"
WEB_PUBLIC = ROOT / "web" / "public"


def _cue_geometry(svg_path: Path):
    root = ElementTree.fromstring(svg_path.read_text(encoding="utf-8"))
    mask = next(element for element in root if element.tag.endswith("mask"))
    cue_rects = [element for element in mask if element.tag.endswith("rect")][1:]
    marker = next(
        element
        for element in root
        if element.tag.endswith("rect") and element.attrib.get("fill") == "#FFF04B"
    )
    return (
        [
            tuple(float(rect.attrib[key]) for key in ("x", "y", "width", "height", "rx"))
            for rect in cue_rects
        ],
        tuple(float(marker.attrib[key]) for key in ("x", "width")),
    )


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
        'name="twitter:image:alt" content="DubSync. Timing follows the performance."',
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
        "brand/dubsync-maskable-192.png",
        "brand/dubsync-maskable-512.png",
        "brand/dubsync-apple-touch.png",
        "brand/dubsync-social.png",
    ):
        asset = WEB_PUBLIC / relative_path
        assert asset.is_file(), f"Missing public brand or SEO asset: {relative_path}"
        assert asset.stat().st_size > 0

    manifest = json.loads((WEB_PUBLIC / "site.webmanifest").read_text(encoding="utf-8"))
    declared_icons = {(icon["sizes"], icon["purpose"], icon["src"]) for icon in manifest["icons"]}
    assert declared_icons == {
        ("192x192", "any", "/brand/dubsync-icon-192.png"),
        ("512x512", "any", "/brand/dubsync-icon-512.png"),
        ("192x192", "maskable", "/brand/dubsync-maskable-192.png"),
        ("512x512", "maskable", "/brand/dubsync-maskable-512.png"),
    }

    for name, expected_size, expected_color_type in (
        ("dubsync-icon-192.png", 192, 6),
        ("dubsync-icon-512.png", 512, 6),
        ("dubsync-apple-touch.png", 180, 2),
        ("dubsync-maskable-192.png", 192, 2),
        ("dubsync-maskable-512.png", 512, 2),
    ):
        payload = (WEB_PUBLIC / "brand" / name).read_bytes()
        width, height = unpack(">II", payload[16:24])
        assert (width, height) == (expected_size, expected_size)
        assert payload[25] == expected_color_type

    generator = (ROOT / "web" / "scripts" / "build-brand-assets.mjs").read_text(encoding="utf-8")
    assert "width:64%;height:64%" in generator
    assert "document.fonts.ready" in generator


def test_logo_cues_cross_the_yellow_timing_marker_like_the_reference():
    expected_cues = [
        (14.0, 19.0, 34.0, 7.0, 3.5),
        (14.0, 38.0, 26.0, 7.0, 3.5),
    ]

    for svg_path in (WEB_PUBLIC / "brand" / "dubsync-mark.svg", WEB_PUBLIC / "favicon.svg"):
        cues, (marker_x, marker_width) = _cue_geometry(svg_path)

        assert cues == expected_cues
        assert all(cue_x < marker_x for cue_x, _, _, _, _ in cues)
        assert all(cue_x + cue_width > marker_x + marker_width for cue_x, _, cue_width, _, _ in cues)
