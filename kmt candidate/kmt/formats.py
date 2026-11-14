"""Render any response dict as JSON, plain text, Markdown, XML or CSV."""
import csv, io, json
import xml.etree.ElementTree as ET

FORMATS = {"json": "application/json", "text": "text/plain", "markdown": "text/markdown",
           "xml": "application/xml", "csv": "text/csv"}
_MAIN = ("reply", "text", "answer")


def _main(obj):
    for k in _MAIN:
        if obj.get(k):
            return str(obj[k])
    return ""


_MD_SPECIAL = set("\\`*_{}[]()<>#+-!|~&")


def _md(text):
    """Escape Markdown/HTML-significant characters so replies cannot inject formatting or markup."""
    return "".join("\\" + c if c in _MD_SPECIAL else c for c in str(text))


def _scalar(v):
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else ("" if v is None else str(v))


def _xml_add(parent, key, value):
    if isinstance(value, dict):
        node = ET.SubElement(parent, str(key))
        for k, v in value.items():
            _xml_add(node, k, v)
    elif isinstance(value, list):
        node = ET.SubElement(parent, str(key))
        for v in value:
            _xml_add(node, "item", v)
    else:
        ET.SubElement(parent, str(key)).text = _scalar(value)


def render(obj, fmt="json"):
    """Returns (content_type, body_string). Raises ValueError for an unknown format."""
    fmt = (fmt or "json").lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown format '{fmt}'; choose one of {', '.join(FORMATS)}")
    ct = FORMATS[fmt] + "; charset=utf-8"
    if fmt == "json":
        return ct, json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    if fmt == "text":
        lines = [_main(obj) or json.dumps(obj, ensure_ascii=False, default=str)]
        if obj.get("warning"):
            lines.append(f"(warning: {obj['warning']})")
        return ct, "\n".join(lines)
    if fmt == "markdown":
        out = [f"**{_md(_main(obj))}**" if _main(obj) else "_(no reply)_", ""]
        for k, v in obj.items():
            if k not in _MAIN and v not in (None, "", [], {}):
                out.append(f"- **{_md(k)}**: {_md(_scalar(v))}")
        return ct, "\n".join(out)
    if fmt == "xml":
        root = ET.Element("response")
        for k, v in obj.items():
            _xml_add(root, k, v)
        return ct, ET.tostring(root, encoding="unicode")
    buf = io.StringIO(); w = csv.writer(buf)                      # csv
    w.writerow(["field", "value"])
    for k, v in obj.items():
        w.writerow([k, _scalar(v)])
    return ct, buf.getvalue()
