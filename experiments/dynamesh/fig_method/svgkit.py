import base64, io, math

def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))

def rbox(x, y, w, h, fill, stroke, rx=9, sw=1.4, extra=''):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')

SANS  = "Helvetica, Arial, sans-serif"
SERIF = "Georgia, 'Times New Roman', Times, serif"
MONO  = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"

def txt(x, y, s, size=15, fill='#12203A', anchor='middle', weight='400',
        style='normal', family=SANS, ls='0', op=1.0):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}" '
            f'text-anchor="{anchor}" letter-spacing="{ls}" opacity="{op}">{esc(s)}</text>')

def rich(x, y, parts, size=15, fill='#12203A', anchor='start', weight='400',
         family=SANS, style='normal'):
    """parts: list of (text, kind) with kind in '', 'sub', 'sup', 'it', 'itsub'."""
    out = []
    for t, k in parts:
        a = ''
        if 'sub' in k:  a += f' baseline-shift="sub" font-size="{size*0.66:.1f}"'
        if 'sup' in k:  a += f' baseline-shift="super" font-size="{size*0.66:.1f}"'
        if 'it' in k:   a += ' font-style="italic"'
        if 'sm' in k:   a += f' font-size="{size*0.78:.1f}"'
        if 'sf' in k:   a += f' font-family="{SERIF}"'
        out.append(f'<tspan{a}>{esc(t)}</tspan>')
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}" '
            f'text-anchor="{anchor}" xml:space="preserve">' + ''.join(out) + '</text>')

def arrow(x1, y1, x2, y2, stroke='#4A5A76', sw=1.9, marker='ah', dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" marker-end="url(#{marker})"{d}/>')

def curve(x1, y1, cx1, cy1, cx2, cy2, x2, y2, stroke='#4A5A76', sw=1.9, marker='ah', dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<path d="M {x1:.1f},{y1:.1f} C {cx1:.1f},{cy1:.1f} {cx2:.1f},{cy2:.1f} '
            f'{x2:.1f},{y2:.1f}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
            f'marker-end="url(#{marker})"{d}/>')

def lock(x, y, s=1.0, col='#2F4C8C'):
    return (f'<g transform="translate({x:.1f},{y:.1f}) scale({s})" stroke="{col}" fill="{col}">'
            f'<path d="M -3.6,-1.2 v -2.6 a 3.6,3.6 0 0 1 7.2,0 v 2.6" fill="none" '
            f'stroke-width="1.7" stroke-linecap="round"/>'
            f'<rect x="-5.6" y="-1.4" width="11.2" height="8.4" rx="1.7" stroke="none"/></g>')

def pill(x, y, w, h, label, fill, edge, tcol, size=12.5, weight='600'):
    return (rbox(x, y, w, h, fill, edge, rx=h / 2, sw=1.2)
            + txt(x + w / 2, y + h / 2 + size * 0.36, label, size=size, fill=tcol, weight=weight))

def png_b64(path, max_px=520):
    from PIL import Image
    im = Image.open(path).convert('RGBA')
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((int(im.size[0] * r), int(im.size[1] * r)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, 'PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), im.size

def image(x, y, w, h, b64, extra=''):
    return (f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'preserveAspectRatio="xMidYMid meet" {extra} '
            f'xlink:href="data:image/png;base64,{b64}"/>')

def brace_h(x1, x2, y, depth=9, col='#4A5A76', sw=1.5, gap=6):
    xm = (x1 + x2) / 2
    return (f'<path d="M {x1},{y} v {-depth} H {xm-gap} M {xm+gap},{y-depth} H {x2} v {depth}" '
            f'fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"/>')

DEFS = '''<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"
        orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#4A5A76"/></marker>
<marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"
        orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#2E9E5B"/></marker>
<marker id="ahw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5"
        orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#8FA2C0"/></marker>
</defs>'''
