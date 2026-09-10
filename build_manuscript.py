#!/usr/bin/env python3
"""Build the editable C3FAR-Sense manuscript as a polished Word document."""

from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_upgrade_10seed"
FIGURES = ROOT / "manuscript_figures_upgrade"
LATENCY = ROOT / "results_adaptive" / "latency.csv"
OUTDIR = ROOT.parent / "deliverables"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTDIR / "C3FAR-Sense_Upgraded_Physical_Communication_Manuscript.docx"

NAVY = "17365D"
BLUE = "2F75B5"
LIGHT_BLUE = "D9EAF7"
LIGHT_ORANGE = "FBE5D6"
LIGHT_GRAY = "F2F4F7"
MID_GRAY = "D0D5DD"
TEXT_GRAY = "475467"
WHITE = "FFFFFF"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, **kwargs) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_borders = tc_pr.first_child_found_in("w:tcBorders")
    if tc_borders is None:
        tc_borders = OxmlElement("w:tcBorders")
        tc_pr.append(tc_borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge not in kwargs:
            continue
        edge_data = kwargs[edge]
        tag = "w:" + edge
        element = tc_borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            tc_borders.append(element)
        for key, value in edge_data.items():
            element.set(qn("w:" + key), str(value))


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_cell_margins(cell, top=70, start=80, bottom=70, end=80) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn("w:" + m))
        if node is None:
            node = OxmlElement("w:" + m)
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_repeat_keep(paragraph, keep_with_next=False, keep_together=False, page_break_before=False) -> None:
    pf = paragraph.paragraph_format
    pf.keep_with_next = keep_with_next
    pf.keep_together = keep_together
    pf.page_break_before = page_break_before


def add_field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = instruction
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)


def add_hyperlink(paragraph, text: str, url: str) -> None:
    part = paragraph.part
    rid = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLUE)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.append(color)
    rpr.append(underline)
    run.append(rpr)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def configure_document(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(1.85)
    sec.bottom_margin = Cm(1.75)
    sec.left_margin = Cm(1.9)
    sec.right_margin = Cm(1.9)
    sec.header_distance = Cm(0.75)
    sec.footer_distance = Cm(0.75)
    sec.different_first_page_header_footer = True

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor.from_string("202124")
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = 1.08
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.widow_control = True

    for name, size, color, before, after in (
        ("Title", 20, NAVY, 0, 10),
        ("Heading 1", 14, NAVY, 12, 5),
        ("Heading 2", 12, BLUE, 9, 3),
        ("Heading 3", 11.5, TEXT_GRAY, 7, 2),
    ):
        style = styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.widow_control = True

    if "Figure Caption" not in [s.name for s in styles]:
        cap = styles.add_style("Figure Caption", WD_STYLE_TYPE.PARAGRAPH)
    else:
        cap = styles["Figure Caption"]
    cap.font.name = "Times New Roman"
    cap._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    cap.font.size = Pt(9)
    cap.font.color.rgb = RGBColor.from_string(TEXT_GRAY)
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cap.paragraph_format.space_before = Pt(2)
    cap.paragraph_format.space_after = Pt(7)
    cap.paragraph_format.keep_together = True
    cap.paragraph_format.widow_control = True

    if "Reference" not in [s.name for s in styles]:
        ref = styles.add_style("Reference", WD_STYLE_TYPE.PARAGRAPH)
    else:
        ref = styles["Reference"]
    ref.font.name = "Times New Roman"
    ref._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    ref.font.size = Pt(9.5)
    ref.paragraph_format.left_indent = Cm(0.7)
    ref.paragraph_format.first_line_indent = Cm(-0.7)
    ref.paragraph_format.space_after = Pt(3)
    ref.paragraph_format.line_spacing = 1.0
    ref.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    header = sec.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = hp.add_run("C³FAR-Sense | Manuscript for Physical Communication")
    hr.font.name = "Times New Roman"
    hr.font.size = Pt(8.5)
    hr.font.color.rgb = RGBColor.from_string(TEXT_GRAY)
    ppr = hp._p.get_or_add_pPr()
    bottom = OxmlElement("w:pBdr")
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), "4")
    b.set(qn("w:space"), "1")
    b.set(qn("w:color"), MID_GRAY)
    bottom.append(b)
    ppr.append(bottom)

    footer = sec.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("Page ")
    fr.font.name = "Times New Roman"
    fr.font.size = Pt(9)
    add_field(fp, "PAGE")
    fp.add_run(" of ")
    add_field(fp, "NUMPAGES")

    props = doc.core_properties
    props.title = "C³FAR-Sense: Context-conditioned conformal false-alarm control for short-window spectrum sensing under receiver mismatch"
    props.subject = "Full-length research manuscript prepared for Physical Communication"
    props.keywords = "spectrum sensing; conformal calibration; false-alarm control; cognitive radio; receiver mismatch"
    props.comments = "Editable upgraded manuscript with reproducible ten-seed simulation results. Replace author metadata before submission."


def add_body(doc: Document, text: str, *, first_line=True, italic=False, bold=False, align=None, space_after=5) -> None:
    p = doc.add_paragraph()
    if first_line:
        p.paragraph_format.first_line_indent = Cm(0.55)
    p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.italic = italic
    r.bold = bold
    return p


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.3)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_together = True
        p.add_run(item)


def add_equation(doc: Document, expression: str, number: int) -> None:
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Cm(14.9)
    table.columns[1].width = Cm(1.4)
    for cell in table.rows[0].cells:
        set_cell_border(cell,
                        top={"val": "nil"}, bottom={"val": "nil"},
                        left={"val": "nil"}, right={"val": "nil"})
        set_cell_margins(cell, top=15, bottom=15, start=30, end=30)
    p = table.cell(0, 0).paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_together = True
    # Editable Cambria Math runs are used instead of Word's linear-math parser.
    # A small parser converts x_y and x^{...} into true Word sub/superscripts,
    # producing stable rendering in both LibreOffice and Microsoft Word.
    def add_math_run(value: str, *, sub=False, sup=False):
        if not value:
            return
        run = p.add_run(value)
        run.font.name = "Cambria Math"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria Math")
        run.font.size = Pt(10.2)
        run.font.subscript = sub
        run.font.superscript = sup

    def grouped(s: str, start: int) -> tuple[str, int]:
        if start >= len(s):
            return "", start
        if s[start] != "{":
            return s[start], start + 1
        depth = 0
        chars = []
        i = start
        while i < len(s):
            ch = s[i]
            if ch == "{":
                depth += 1
                if depth > 1:
                    chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(chars), i + 1
                chars.append(ch)
            else:
                chars.append(ch)
            i += 1
        return "{" + "".join(chars), i

    i = 0
    plain = []
    while i < len(expression):
        ch = expression[i]
        if ch == "\n":
            add_math_run("".join(plain)); plain = []
            p.add_run().add_break()
            i += 1
        elif ch in "_^":
            add_math_run("".join(plain)); plain = []
            value, i = grouped(expression, i + 1)
            add_math_run(value, sub=(ch == "_"), sup=(ch == "^"))
        else:
            plain.append(ch)
            i += 1
    add_math_run("".join(plain))
    np_ = table.cell(0, 1).paragraphs[0]
    np_.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    np_.paragraph_format.space_after = Pt(0)
    nr = np_.add_run(f"({number})")
    nr.font.name = "Times New Roman"
    nr.font.size = Pt(10.5)
    prevent_row_split(table.rows[0])


def add_table(doc: Document, caption: str, headers: list[str], rows: list[list[str]], widths=None,
              font_size=8.2, note: str | None = None, page_break_before=False) -> None:
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(5)
    cap.paragraph_format.space_after = Pt(2)
    cap.paragraph_format.keep_with_next = True
    cap.paragraph_format.page_break_before = page_break_before
    rr = cap.add_run(caption)
    rr.bold = True
    rr.font.name = "Times New Roman"
    rr.font.size = Pt(9)
    rr.font.color.rgb = RGBColor.from_string(NAVY)

    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.style = "Table Grid"
    if widths:
        for i, width in enumerate(widths):
            tbl.columns[i].width = Cm(width)
    hdr = tbl.rows[0]
    set_repeat_table_header(hdr)
    prevent_row_split(hdr)
    for i, header in enumerate(headers):
        cell = hdr.cells[i]
        set_cell_shading(cell, NAVY)
        set_cell_margins(cell, top=75, bottom=75, start=65, end=65)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(header)
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor.from_string(WHITE)
    for ridx, row in enumerate(rows):
        cells = tbl.add_row().cells
        prevent_row_split(tbl.rows[-1])
        for cidx, value in enumerate(row):
            cell = cells[cidx]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell, top=60, bottom=60, start=55, end=55)
            if ridx % 2:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if cidx == 1 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(value))
            run.font.name = "Times New Roman"
            run.font.size = Pt(font_size)
    if note:
        p = doc.add_paragraph(style="Figure Caption")
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(6)
        p.add_run("Note. ").bold = True
        p.add_run(note)


def add_figure(doc: Document, path: Path, caption: str, width=6.55) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(style="Figure Caption")
    cap.add_run(caption)


def fmt(row, metric: str, digits=3) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} ± {row[f'{metric}_ci95']:.{digits}f}"


def fmt_p(value: float) -> str:
    """Format p-values compactly without programming-style exponent notation."""
    value = float(value)
    if value >= 1e-3:
        return f"{value:.3g}"
    exponent = int(math.floor(math.log10(value)))
    coefficient = value / (10 ** exponent)
    superscript = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{coefficient:.2g}×10{superscript}"


def get_row(df, **filters):
    s = df
    for key, value in filters.items():
        s = s[s[key] == value]
    if len(s) != 1:
        raise RuntimeError(f"Expected one row for {filters}, found {len(s)}")
    return s.iloc[0]


def add_reference(doc: Document, number: int, text: str, doi: str | None = None, url: str | None = None) -> None:
    p = doc.add_paragraph(style="Reference")
    p.add_run(f"[{number}] {text}")
    if doi:
        p.add_run(" ")
        add_hyperlink(p, f"https://doi.org/{doi}", f"https://doi.org/{doi}")
    elif url:
        p.add_run(" ")
        add_hyperlink(p, url, url)


def build() -> Path:
    global_df = pd.read_csv(RESULTS / "summary_global.csv")
    env_df = pd.read_csv(RESULTS / "summary_environment.csv")
    cal_df = pd.read_csv(RESULTS / "summary_calibration_size.csv")
    runs_df = pd.read_csv(RESULTS / "runs_global.csv")
    alpha_df = pd.read_csv(RESULTS / "summary_alpha_sweep.csv")
    modulation_df = pd.read_csv(RESULTS / "summary_modulation.csv")
    predicted_df = pd.read_csv(RESULTS / "summary_predicted_group.csv")
    gate_df = pd.read_csv(RESULTS / "summary_gate_confusion.csv")
    mismatch_df = pd.read_csv(RESULTS / "summary_reference_mismatch.csv")
    contamination_df = pd.read_csv(RESULTS / "summary_reference_contamination.csv")
    paired_df = pd.read_csv(RESULTS / "paired_comparisons.csv")
    latency_df = pd.read_csv(LATENCY)

    doc = Document()
    configure_document(doc)

    # Front matter
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("PHYSICAL COMMUNICATION — FULL LENGTH ARTICLE")
    r.bold = True
    r.font.name = "Arial"
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor.from_string(BLUE)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.add_run("C³FAR-Sense: Context-conditioned conformal false-alarm control for short-window spectrum sensing under receiver mismatch")

    authors = doc.add_paragraph()
    authors.paragraph_format.space_after = Pt(3)
    authors.alignment = WD_ALIGN_PARAGRAPH.LEFT
    ar = authors.add_run("[Author 1 Name]ᵃ,* · [Author 2 Name]ᵇ · [Author 3 Name]ᶜ")
    ar.bold = True
    ar.font.size = Pt(11)
    ar.font.color.rgb = RGBColor.from_string(NAVY)
    for line in (
        "ᵃ [Department, Institution, City, Country]",
        "ᵇ [Department, Institution, City, Country]",
        "ᶜ [Department, Institution, City, Country]",
        "* Corresponding author: [name and e-mail]",
    ):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.first_line_indent = Cm(0)
        rr = p.add_run(line)
        rr.font.size = Pt(9.5)
        rr.font.color.rgb = RGBColor.from_string(TEXT_GRAY)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(3)
    rr = p.add_run("Abstract")
    rr.bold = True
    rr.font.size = Pt(12)
    rr.font.color.rgb = RGBColor.from_string(NAVY)
    abstract = (
        "Short-window spectrum sensors are often evaluated with thresholds selected under the same noise law as the test data, although deployed receivers face noise-power drift, colored noise, impulses, and front-end mismatch. This paper presents C³FAR-Sense, a context-conditioned conformal detector that pairs each sensing window with an independent noise-reference window. A 43-dimensional evidence bank and 10 reference-only context statistics form a compact 484-parameter bilinear occupancy scorer; a 33-parameter gate selects one of three calibration regimes. Groupwise split-conformal order statistics use only held-out noise-only windows and control false alarms under within-group exchangeability. Thresholds can be refreshed without retraining the scorer or gate. Ten-seed simulations span ten modulation families, native windows of 64–256 samples, −20 to 0 dB nominal SNR, channel and RF impairments, and unseen colored, impulsive, and Student-t noise. At L = 128, matched performance is Pᵈ = 0.533 ± 0.007 at Pᶠ = 0.102 ± 0.005. Under stress, a global refreshed threshold gives Pᵈ = 0.355 ± 0.024 at Pᶠ = 0.099 ± 0.004, whereas C³FAR-Sense reaches Pᵈ = 0.444 ± 0.014 at Pᶠ = 0.098 ± 0.002; the paired gain is 8.84 ± 1.96 percentage points (Holm-adjusted p = 1.8×10⁻⁵). Across targets α ∈ {0.01,0.05,0.10,0.20}, contextual calibration reduces worst-environment false-alarm error, including 0.188 to 0.031 at α = 0.10. Label-free K-means grouping matches the supervised gate within 0.06 percentage points of Pᵈ, while an oracle partition adds only 0.23 points. Modulation-resolved, calibration-size, reference-mismatch, and contamination experiments identify both robustness and failure boundaries. A frozen 15,000-pair raw-I/Q dataset, derived features, code, manifests, and validation checks accompany the study."
    )
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_after = Pt(5)
    p.add_run(abstract)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.add_run("Keywords: ").bold = True
    p.add_run("spectrum sensing; conformal calibration; false-alarm control; cognitive radio; receiver mismatch; noise context")

    # Highlights in a compact shaded box.
    htbl = doc.add_table(rows=1, cols=1)
    htbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = htbl.cell(0, 0)
    set_cell_shading(cell, LIGHT_BLUE)
    set_cell_border(cell,
                    top={"val": "single", "sz": "6", "color": "9DC3E6"},
                    bottom={"val": "single", "sz": "6", "color": "9DC3E6"},
                    left={"val": "single", "sz": "6", "color": "9DC3E6"},
                    right={"val": "single", "sz": "6", "color": "9DC3E6"})
    set_cell_margins(cell, top=90, bottom=90, start=120, end=120)
    hp = cell.paragraphs[0]
    hp.paragraph_format.space_after = Pt(2)
    hp.add_run("Research highlights").bold = True
    for item in (
        "Finite-sample false-alarm control is obtained within learned noise contexts.",
        "A paired noise reference conditions both the score and its decision threshold.",
        "A label-free context partition nearly matches the supervised gate and oracle ceiling.",
        "Ten-seed stress tests, frozen raw I/Q data, and executable validation are supplied.",
    ):
        pp = cell.add_paragraph(style="List Bullet")
        pp.paragraph_format.space_after = Pt(0)
        pp.paragraph_format.left_indent = Cm(0.5)
        pp.add_run(item)

    doc.add_heading("1. Introduction", level=1)
    add_body(doc, (
        "Dynamic spectrum access depends on a receiver deciding, within a short observation interval, whether a channel is idle or occupied. The decision must be made before transmission begins and is therefore asymmetric: a missed incumbent can cause harmful interference, while an excessive false-alarm rate wastes spectrum. Surveys of cognitive-radio sensing accordingly report detection probability Pᵈ at a prescribed false-alarm probability Pᶠ, rather than accuracy alone [2,3]. Classical energy detection remains attractive because it is cheap and modulation-agnostic, but its threshold depends on an accurate noise model. Even small noise-power errors can create an SNR wall below which additional samples do not restore reliable detection [4]. Eigenvalue and cyclostationary detectors exploit richer structure [5,6], yet they introduce their own observation-length, stationarity, and computational requirements."
    ))
    add_body(doc, (
        "Learning-based sensors replace a hand-derived test statistic with a score learned from in-phase/quadrature (I/Q) windows. Convolutional, recurrent, attention-based, and fused time–frequency models have improved benchmark discrimination across modulation and channel diversity [7–12,24,25]. A recent two-decade survey also identifies robustness, adaptivity, and reproducible evaluation as continuing spectrum-sensing priorities [23]. The base study for this work, InitStateSenseNet [1], asks whether a short raw prefix can initialize a CNN–RNN state and thereby improve short-window inference. It reports gains on RadioML2016.10b while identifying matched synthetic H₀ noise, derived long windows, threshold scope, and absent hardware validation as limitations. Those limits motivate a different question: how should a learned score become a dependable operating decision when the receiver noise law changes?"
    ))
    add_body(doc, (
        "That conversion is often treated as a plotting detail. A threshold is selected on an evaluation slice so that its empirical Pᶠ is close to a target, and the resulting Pᵈ is reported. Such a protocol is useful diagnostically but cannot be reproduced online without labels from the same test distribution. A single threshold calibrated under white Gaussian noise may also average over incompatible regimes: it can be permissive for colored noise and unnecessarily conservative for impulsive noise. The score can rank H₁ above H₀ reasonably well—hence retain a useful AUC—while its operational false-alarm rate becomes unacceptable. In other words, representation accuracy and decision calibration are separate engineering problems."
    ))
    add_body(doc, (
        "Conformal prediction offers distribution-free finite-sample calibration under exchangeability [14,15]. Recent communications work applies conformal risk control to wideband sub-Nyquist sensing with a false-negative constraint [16] and develops context-dependent weighting for general wireless AI under distribution shift [17,18]. These works establish that calibration can be treated as a first-class component. They do not, however, provide a short-window narrowband detector that (i) conditions its representation on an explicit receiver-noise reference, (ii) controls Pᶠ by noise context, and (iii) permits threshold-only refresh after deployment. The present paper addresses that combination."
    ))
    add_body(doc, "The main contributions are:", first_line=False)
    add_bullets(doc, [
        "A paired-window sensing formulation in which an occupancy window y and a trusted noise-only reference r are processed jointly. Unlike per-window unit-energy normalization, the formulation preserves energy relative to the receiver context.",
        "A compact, auditable detector comprising 43 evidence features, 10 reference-context features, and explicit evidence–context interactions. The logistic bilinear scorer has 484 trainable parameters; with the 33-parameter context gate, the complete learned detector has 517 parameters.",
        "A context-conditioned split-conformal threshold rule with a finite-sample Pᶠ guarantee for each predicted context under within-context exchangeability. Under drift, thresholds alone are recomputed from a small independent H₀ bank.",
        "A supervised gate, label-free K-means partition, and simulator-label oracle that separate the value of contextual partitioning from the value of regime supervision and expose the attainable ceiling.",
        "A ten-seed study across four Pᶠ targets, three window lengths, ten modulations, calibration banks from 100 to 3000 windows, five reference-mismatch levels, and five contamination rates, with paired tests and Holm correction.",
        "A frozen synthetic release containing 15,000 paired raw-I/Q windows, 43 evidence and 10 context features, sample metadata, checksums, executable Python code, and a numerical validation script.",
    ])
    add_body(doc, (
        "The remainder of the paper reviews sensing and calibration, formulates the paired-window problem, derives C³FAR-Sense and its guarantee, describes the reproducible simulation, presents matched and mismatch results, and closes with deployment implications and limitations."
    ))

    doc.add_heading("2. Related work and research gap", level=1)
    doc.add_heading("2.1. Model-based spectrum sensing", level=2)
    add_body(doc, (
        "For a complex observation y[n], energy detection compares the average |y[n]|² with a threshold derived from a noise-power estimate [3,22]. Its principal advantage is universality: no modulation or pilot is required. Its principal weakness is that uncertainty in noise power shifts the entire H₀ distribution. Tandra and Sahai formalized the resulting SNR-wall phenomenon [4]. Eigenvalue methods mitigate explicit noise-power dependence by exploiting covariance structure [5], and cyclostationary detectors use periodic second-order statistics [6]. These alternatives can be effective when enough samples are available, but the relevant estimates are noisy for short L and may fail when the assumed structure is absent. A practical receiver therefore still needs a mechanism for tracking its operating threshold."
    ))
    doc.add_heading("2.2. Learned short-window detectors", level=2)
    add_body(doc, (
        "Deep spectrum sensors learn non-linear statistics from raw I/Q samples or engineered transforms. Gao et al. demonstrate learned sensing beyond an energy statistic [7]; Cheng et al. specialize deep detection to OFDM structure [8]; and Xie et al. combine convolution with an LSTM [9]. Hybrid and multi-feature models continue this direction [10–12]. More recent work uses a gated recurrent unit with attention and multiple descriptors [24] or fused time–frequency representations [25]. The architectural trend is toward richer representations, but a stronger ranking model does not by itself supply a reliable operating threshold under a changed receiver."
    ))
    add_body(doc, (
        "InitStateSenseNet [1] is especially relevant because it isolates the short-window regime and conditions a recurrent initial state on an early I/Q prefix. Its evaluation also makes several realism limits visible: H₀ is matched synthetic circular Gaussian noise, energy is removed by per-window normalization, and diagnostic operating thresholds can be selected within evaluation slices. C³FAR-Sense takes the same latency-sensitive sensing problem as its point of departure, but changes the research axis from hidden-state initialization to receiver-aware decision calibration. It does not reuse the base architecture or its reported measurements."
    ))
    doc.add_heading("2.3. Conformal calibration for communication systems", level=2)
    add_body(doc, (
        "Split conformal methods reserve a calibration set and use its order statistics to obtain finite-sample guarantees without specifying the score distribution [14,15]. The guarantee is marginal unless conditioning is introduced, and exact conditional validity for arbitrary continuous covariates is generally impossible without additional assumptions [19]. Mondrian conformal prediction addresses a tractable version by partitioning examples into pre-defined groups and calibrating within each group. This is the strategy adopted here: the groups are receiver-noise regimes predicted from r, and the formal claim is conditional on the predicted group, not on every possible analog noise parameter."
    ))
    add_body(doc, (
        "Lee et al. [16] use conformal risk control to bound false negatives in sub-Nyquist wideband recovery. Yoo et al. [17] address cross-context wireless calibration through meta-learned weighting, while Cohen et al. [18] apply conformal prediction to calibrated communication decisions. Our target and mechanism differ: the controlled event is a false alarm under H₀; the task is a single short narrowband window; and the runtime context is measured by a paired noise reference. Table 1 positions the proposed method without implying that conformal calibration itself is new."
    ))
    add_table(doc, "Table 1. Positioning of C³FAR-Sense relative to representative sensing and calibration approaches.",
              ["Approach", "Learned score", "Explicit noise context", "Finite-sample target", "Deployment update"],
              [
                  ["Model-based ED/CFAR [3–5]", "No", "Noise estimate", "Model-dependent Pᶠ", "Noise/threshold estimate"],
                  ["Deep sensing [7–12,24,25]", "Yes", "Usually no", "None by default", "Retrain or retune"],
                  ["InitStateSenseNet [1]", "Yes", "No", "None", "Fixed model/threshold"],
                  ["CRC sub-Nyquist sensing [16]", "Yes or reconstructed", "No paired reference", "False-negative risk", "Calibration threshold"],
                  ["Context-dependent wireless CP [17]", "Yes", "Metadata context", "Prediction-set error", "Meta-weighting"],
                  ["C³FAR-Sense (this work)", "Yes; 484 parameters", "Paired r window", "Group-conditional Pᶠ", "Thresholds only"],
              ], widths=[4.1, 2.5, 3.2, 3.1, 3.0], font_size=7.7,
              note="CP denotes conformal prediction; CRC denotes conformal risk control. The guarantee in this work requires H₀ exchangeability within the predicted context group.")

    doc.add_heading("3. System model and objectives", level=1)
    doc.add_heading("3.1. Paired sensing and reference windows", level=2)
    add_body(doc, (
        "Consider a single receiver and a candidate channel observed for L complex baseband samples. Alongside the sensing window y = [y[0], …, y[L−1]]ᵀ, the receiver supplies an independent reference r of the same length that is known to be noise-only. The reference may be acquired from a protected guard subband, a scheduled quiet interval, or a periodically characterized front-end channel. It must not contain the incumbent whose presence is being tested. The binary hypotheses are"
    ))
    add_equation(doc, "H₀:  y[n] = w_y[n],   r[n] = w_r[n]\nH₁:  y[n] = a·e^{j(2πνn+φ)}·(h∗x)[n] + w_y[n],   r[n] = w_r[n]", 1)
    add_body(doc, (
        "where x is an unknown modulated waveform, h is an unknown finite impulse response, a is a signal scale, ν and φ are carrier-frequency and phase offsets, and wᵧ and wᵣ are independent realizations drawn from a shared receiver-noise context. Their instantaneous powers need not be equal. We model residual reference mismatch in decibels as"
    ))
    add_equation(doc, "10·log₁₀(σ_r²/σ_y²) = δ_r,     δ_r ∼ 𝒩(0, σ_δ²)", 2)
    add_body(doc, (
        "which prevents the detector from relying on a perfectly matched noise copy. The reference is therefore side information about the current receiver law, not a sample-by-sample estimate of wᵧ."
    ))

    doc.add_heading("3.2. Operating metrics and design requirements", level=2)
    add_body(doc, "For detector D(y,r) ∈ {0,1}, the operating probabilities are", first_line=True)
    add_equation(doc, "P_f = Pr[D(y,r)=1 | H₀],          P_d = Pr[D(y,r)=1 | H₁]", 3)
    add_body(doc, (
        "The primary target is α = 0.10 for Pᶠ. AUC is also reported because it measures ranking independently of one threshold, but AUC cannot certify the operational constraint. The design requirements are: retain relative energy; exploit non-energy structure; use no test labels for threshold selection; provide a finite-sample statement under a clear assumption; permit low-cost refresh; and expose failure under distribution shift instead of silently recalibrating on the labeled test set."
    ))

    doc.add_heading("4. Proposed C³FAR-Sense detector", level=1)
    add_figure(doc, FIGURES / "fig1_architecture.png",
               "Fig. 1. C³FAR-Sense. The sensing and noise-reference windows generate evidence e and context c. Their explicit interactions feed a frozen logistic scorer. A reference-only gate selects the calibration group, whose H₀ order statistic supplies the decision threshold. At deployment, only the thresholds are refreshed.")

    doc.add_heading("4.1. Robust scaling and feature construction", level=2)
    add_body(doc, (
        "Per-window unit-energy normalization makes a detector invariant to one of the strongest occupancy cues. Conversely, using raw amplitude alone creates severe sensitivity to receiver gain. C³FAR-Sense uses the reference to define a robust scale while preserving the energy ratio. Let"
    ))
    add_equation(doc, "m_r = median_n |r[n]|² / ln 2 + ε\nỹ[n] = y[n]/√m_r,      r̃[n] = r[n]/√m_r", 4)
    add_body(doc, (
        "where division by ln 2 makes mᵣ consistent for the mean power of ideal circular Gaussian noise and ε avoids numerical singularity. This normalization follows the robust-statistics principle of limiting the influence of a small number of extreme samples [20]. It does not assume that the actual noise is Gaussian; under impulsive noise, the median remains less volatile than the mean."
    ))
    add_body(doc, "The first evidence component is the log energy ratio", first_line=True)
    add_equation(doc, "e_E = log[(L⁻¹Σ_n |y[n]|² + ε) / (L⁻¹Σ_n |r[n]|² + ε)]", 5)
    add_body(doc, (
        "supplemented by median- and 0.9-quantile amplitude ratios. Structure is represented by spectral entropy, flatness, peak-to-mean ratio, normalized eight-band spectra, complex pseudo-covariance, normalized fourth-order moment, crest factor, phase-increment concentrations, and normalized autocorrelations. For normalized power spectrum p[k], representative components are"
    ))
    add_equation(doc, "H_s(ỹ) = −(1/log L)·Σ_k p[k] log p[k]\nρ_l(ỹ) = |Σ_{n=l}^{L−1} ỹ[n]ỹ*[n−l]| / Σ_n |ỹ[n]|²", 6)
    add_body(doc, (
        "with lags ℓ ∈ {1,2,4,8,16}. For descriptors sensitive to the noise law, both the sensing value and its difference or log ratio relative to r are included. This yields e ∈ ℝ⁴³. The reference-only context c ∈ ℝ¹⁰ comprises log reference power, spectral entropy and flatness, spectral peak, kurtosis, crest factor, an impulse fraction, and autocorrelations at lags 1, 2, and 4. Appendix A gives the complete grouped inventory."
    ))

    doc.add_heading("4.2. Explicit bilinear context fusion", level=2)
    add_body(doc, (
        "A useful occupancy cue can change sign or importance with the noise context. For example, high spectral concentration is signal evidence in white noise but may be normal under strongly colored noise. We therefore expose every evidence–context interaction rather than asking a large opaque backbone to discover it from raw samples. Define"
    ))
    add_equation(doc, "z = [eᵀ, cᵀ, vec(ecᵀ)ᵀ]ᵀ  ∈  ℝ⁴⁸³", 7)
    add_body(doc, "After featurewise standardization learned on the training set, the occupancy score is", first_line=True)
    add_equation(doc, "s_β(y,r) = σ(β₀ + βᵀz),      σ(u) = 1/(1 + e⁻ᵘ)", 8)
    add_body(doc, (
        "and β is fitted by regularized binary cross-entropy. The scorer contains 483 coefficients plus one intercept. Its linear form is intentional: it is auditable, fast, and isolates the value of context and calibration from sheer model capacity. The regularization inverse strength is C = 0.015."
    ))
    add_equation(doc, "β̂ = arg min_β { −N⁻¹Σ_i [y_i log s_i + (1−y_i) log(1−s_i)]\n                         + λ‖β‖₂² }", 9)

    doc.add_heading("4.3. Reference-only context gate", level=2)
    add_body(doc, (
        "A three-class multinomial logistic gate is trained during receiver characterization to distinguish coarse white, colored, and impulsive noise regimes from c. The gate requires regime labels offline, which may come from controlled calibration captures; it never uses an occupancy label or an oracle environment label at runtime. Its predicted group is"
    ))
    add_equation(doc, "ĝ(c) = arg max_{g∈{1,2,3}} q_θ(g | c)", 10)
    add_body(doc, (
        "where qθ is fitted with C = 2.0. With 10 inputs, three coefficient vectors, and three intercepts, the gate has 33 parameters. Unseen conditions are mapped to one of the learned regimes. The method does not claim that these three classes exhaust physical noise behavior; they provide a finite and operationally calibratable partition."
    ))

    doc.add_heading("4.4. Label-free context partition", level=2)
    add_body(doc, (
        "To test whether receiver-regime labels are necessary, a second gate standardizes c and fits K-means with K = 3, 20 initializations, and a seed fixed before evaluation. It is fitted on training reference contexts only and never sees occupancy labels, noise-family labels, calibration scores, or test data. Calibration and future references are assigned to their nearest learned centroid, after which the same groupwise order statistic is applied. Cluster identities are arbitrary and need not correspond one-to-one with white, colored, or impulsive labels. The supervised gate remains the named C³FAR-Sense configuration; K-means is a deployment-oriented ablation. A third partition uses simulator environment labels as an oracle ceiling and is never considered deployable."
    ))

    doc.add_heading("4.5. Groupwise conformal false-alarm calibration", level=2)
    add_body(doc, (
        "Let 𝒞₀ = {(yᵢ,rᵢ)} be a calibration set containing only trusted H₀ windows, disjoint from training and testing. For group g, collect the scores whose reference is assigned to g and sort them as a₍₁₎ ≤ … ≤ a₍ₙg₎. For target α, set"
    ))
    add_equation(doc, "k_g = ⌈(n_g+1)(1−α)⌉,      τ_g = a[k_g]\nif k_g > n_g, set τ_g = +∞", 11)
    add_body(doc, "The online decision is", first_line=True)
    add_equation(doc, "D(y,r) = 𝟙{s_β(y,r) > τ_{ĝ(c)}}", 12)
    add_body(doc, (
        "The strict inequality makes ties conservative. No H₁ example is required for threshold calibration. Consequently, the same mechanism can wrap a different score model, although all experiments here use the compact bilinear scorer."
    ))
    p = add_body(doc, "Proposition 1 (finite-sample groupwise false-alarm control). ", first_line=False, bold=True)
    p.add_run(
        "Fix a predicted group g. If the n_g calibration pairs and one future H₀ pair are exchangeable conditional on ĝ(c)=g, and the scorer and gate were fitted independently of the calibration set, then the rule in (11)–(12) satisfies"
    )
    add_equation(doc, "Pr[D(y,r)=1 | H₀, ĝ(c)=g] ≤ α", 13)
    p = add_body(doc, "Proof. ", first_line=False, italic=True)
    p.add_run(
        "Conditional on the predicted group, exchangeability makes the rank of the future H₀ score among n_g+1 scores uniform (or super-uniform with ties). At most n_g+1−k_g ranks exceed the kth calibration order statistic. Therefore the probability of a strict exceedance is at most (n_g+1−k_g)/(n_g+1) ≤ α. □"
    )
    add_body(doc, (
        "The proposition is distribution-free but not distribution-shift-free. It applies when calibration and future H₀ scores remain exchangeable within the predicted group. It does not guarantee control for a new context before recalibration, for a contaminated H₀ bank, or conditional on an arbitrary continuous noise parameter. These boundaries are central to the deployment protocol."
    ))

    doc.add_heading("4.6. Threshold-only receiver refresh", level=2)
    add_body(doc, (
        "When a receiver detects drift or enters a new site, it obtains M independent noise-only sensing/reference pairs from a guard band or quiet schedule. The scorer β and gate θ remain frozen. Scores are partitioned by ĝ(c), and (11) is recomputed. Because refresh adjusts only order statistics, no gradient optimization or H₁ collection is needed. If a group is empty, the implementation uses the global conformal threshold for availability and flags that group as lacking a conditional guarantee; if a nonempty group is too small for the requested α, (11) returns +∞, the formally conservative decision. In the primary experiment M = 3000; Section 6.4 studies M down to 100."
    ))

    doc.add_heading("4.7. Complexity", level=2)
    add_body(doc, (
        "Feature extraction is O(L log L) because of the FFT and O(L) for the remaining statistics. Scoring requires 483 multiply–accumulate terms; the gate requires 30. Storing the learned coefficients requires 517 scalar parameters, excluding deterministic standardization statistics and three thresholds. This is orders of magnitude smaller than common CNN–RNN backbones and makes it feasible to inspect which interactions dominate a decision."
    ))

    doc.add_heading("5. Experimental methodology", level=1)
    doc.add_heading("5.1. Independent signal windows", level=2)
    add_body(doc, (
        "The study uses a fully generated dataset so that each window length is native rather than formed by concatenating shorter benchmark examples. Ten waveform families are sampled uniformly: BPSK, QPSK, 8PSK, 4-PAM, 16-QAM, 64-QAM, CPFSK, GFSK, AM-DSB, and WBFM. Symbol rates correspond to 3–6 samples per symbol, and a compact triangular pulse-shaping surrogate suppresses discontinuities for linear modulations. Each H₁ window undergoes either flat complex fading or a normalized three-tap channel, a random phase, carrier-frequency offset, and I/Q imbalance. Signal power is set by nominal SNR in {−20,−16,−12,−8,−4,0} dB. Because noise uncertainty and channel power are sampled separately, effective SNR is deliberately not identical to nominal SNR."
    ))
    doc.add_heading("5.2. Noise and receiver mismatch", level=2)
    add_body(doc, (
        "The calibration-matched family contains circular white noise, complex AR(1) colored noise with correlation coefficient ρ ∈ [0.25,0.62], and Bernoulli–Gaussian impulsive noise with impulse probability 0.012 and impulse-to-background factor 12. Noise power is uncertain by ±2 dB, uniformly in decibels. The paired reference uses the same family but an independent realization and a Gaussian power mismatch with 0.30 dB standard deviation. Training and the primary calibration set draw from this family."
    ))
    add_body(doc, (
        "The unseen stress family is intentionally outside the training support: AR(1) correlation increases to ρ ∈ [0.72,0.90]; impulsive noise uses probability 0.045 and factor 28; and a complex Student-t law with three degrees of freedom introduces heavy tails. Noise uncertainty grows to ±4 dB and reference mismatch to 0.85 dB. Carrier-frequency offset expands from ±0.025 to ±0.055 cycles/sample, I/Q gain imbalance from ±0.07 to ±0.14, and phase imbalance from ±4° to ±9°. These settings are synthetic stress tests, not claims about a particular radio standard."
    ))

    sim_rows = [
        ["Window lengths", "L = 64, 128, 256; independently generated"],
        ["Nominal SNR", "−20, −16, −12, −8, −4, 0 dB"],
        ["H₁ families", "BPSK, QPSK, 8PSK, 4-PAM, 16/64-QAM, CPFSK, GFSK, AM-DSB, WBFM"],
        ["Per seed and L", "18,000 train; 4,500 H₀ calibrate; 12,000 matched test; 9,000 stress test; 3,000 stress H₀ refresh"],
        ["Matched noise", "White; AR(1), ρ 0.25–0.62; impulsive, p 0.012 and factor 12"],
        ["Stress noise", "AR(1), ρ 0.72–0.90; impulsive, p 0.045 and factor 28; complex Student-t, df 3"],
        ["Noise uncertainty", "Matched ±2 dB; stress ±4 dB"],
    ]
    add_table(doc, "Table 2. Simulation and evaluation configuration.", ["Item", "Setting"], sim_rows,
              widths=[4.0, 12.2], font_size=8.2)

    doc.add_heading("5.3. Data splits, fitting and uncertainty", level=2)
    add_body(doc, (
        "For every seed and L, 18,000 balanced windows fit the score models. A separate 4,500-window H₀ set calibrates matched thresholds. The matched test set has 12,000 balanced windows. The stress test has 9,000 balanced windows and is paired with an independent 3,000-window stress H₀ refresh bank. The same generated observations are used by every method within a seed. Standardization, model fitting, gating, and threshold selection are completed before test labels are accessed."
    ))
    add_body(doc, (
        "Results are means over ten independent seeds. Error bars and table intervals are two-sided 95% Student-t intervals, mean ± t₀.₉₇₅,₉·s/√10. They quantify seed-to-seed Monte Carlo and fitting variability; they are not confidence bounds on all possible physical channels. Primary method contrasts use two-sided paired t-tests on seed-matched Pᵈ differences; six reported comparisons are corrected by Holm's step-down procedure, and standardized paired effect sizes d_z are supplied. For a model score s and decisions D, reported metrics are"
    ))
    add_equation(doc, "ÂUC = ROC area;    P̂_f = N₀⁻¹Σ_{i:H₀}D_i;    P̂_d = N₁⁻¹Σ_{i:H₁}D_i\nBA = ½[(1−P̂_f)+P̂_d]", 14)

    doc.add_heading("5.4. Baselines and ablations", level=2)
    add_body(doc, (
        "The core rules are ED, ENR-ED, a signal-only MLP, linear fusion, and bilinear fusion. ED uses log mean sensing energy. ENR-ED uses the log sensing/reference energy ratio in (5). The MLP receives 20 signal-only evidence features and has hidden widths 64 and 32 (3457 parameters). Linear fusion applies logistic regression to [e;c] (54 parameters including intercept), while bilinear fusion uses (7)–(9). Under stress, ENR-ED and the bilinear scorer each receive an independent H₀ refresh bank. Adaptive global applies one refreshed threshold; linear-context and C³FAR-Sense apply groupwise thresholds. The latter two isolate interaction modeling from calibration structure. Because C³FAR-Sense and adaptive global share scores, their AUC is identical."
    ))
    add_body(doc, (
        "Two additional partitions use the identical bilinear scores. K-means clustering learns three groups from unlabeled training reference contexts and therefore removes regime labels; cluster identities may permute without affecting calibration. Oracle conformal partitions by simulator environment label and is an evaluation-only ceiling, not a deployable method. The target sweep uses α ∈ {0.01,0.05,0.10,0.20}. Reference robustness varies the unrefreshed sensing/reference power-mismatch standard deviation over {0,0.30,0.85,1.50,3.00} dB and contaminates {0,1,5,10,20}% of reference windows with signals at uniformly sampled −8 to 2 dB leakage SNR. An exploratory integrity guard is calibrated to 1% clean-reference flag rate and forces a signal decision when the reference is flagged; it is reported as a diagnostic, not part of the proposed guarantee."
    ))
    add_body(doc, (
        "Logistic models use the L-BFGS solver with 700 maximum iterations for the bilinear score and 500 for the linear score and gate. The signal-only MLP uses early stopping, a validation fraction of 0.12, minibatches of 256, and an initial learning rate of 10⁻³. All pseudo-random generators and model fits receive the repetition seed. The implementation uses NumPy, SciPy, and scikit-learn [21]."
    ))

    doc.add_heading("5.5. Frozen data release and reproducibility", level=2)
    add_body(doc, (
        "The full Monte Carlo study is generated deterministically because storing every window from 30 large experiments would be unnecessarily costly. To make the signal model and feature pipeline directly inspectable, the accompanying frozen release contains 15,000 paired L = 128 observations: 6000 training pairs, 1500 matched H₀ calibration pairs, 1500 stress H₀ refresh pairs, 3000 matched test pairs, and 3000 stress test pairs. Complex y and r are stored as float32 I/Q arrays of shape N×128×2. The release also stores labels, nominal SNR, modulation, true noise environment, split identifiers, the 43 evidence features, the 10 context features, a row-level CSV, a machine-readable manifest, and SHA-256 checksums."
    ))
    add_body(doc, (
        "The supplied scripts regenerate the frozen release, reproduce either a fast check or the full ten-seed paper experiment, rebuild all figures, and validate hashes, shapes, feature recomputation, seed count, and window lengths. The frozen release is a synthetic benchmark generated by this study and must not be described as measured over-the-air data. Its purpose is exact pipeline verification and extension; external RF captures remain necessary for deployment evidence."
    ))

    doc.add_heading("6. Results", level=1)
    doc.add_heading("6.1. Calibration-matched performance", level=2)
    add_figure(doc, FIGURES / "fig2_matched_pd_snr.png",
               "Fig. 2. Detection probability versus nominal SNR in the calibration-matched mixture at α = 0.10. Thresholds come from the disjoint H₀ calibration set. Markers are ten-seed means; bars are 95% Student-t intervals.")

    matched_models = ["ED", "ENR-ED", "Signal-only MLP", "Linear fusion",
                      "Bilinear fusion + global conformal", "C3FAR-Sense"]
    matched_labels = {
        "Bilinear fusion + global conformal": "Bilinear, global",
        "C3FAR-Sense": "C³FAR-Sense",
    }
    matched_rows = []
    for length in (64, 128, 256):
        for model in matched_models:
            row = get_row(global_df, length=length, split="matched", model=model)
            label = matched_labels.get(model, model)
            matched_rows.append([str(length), label, fmt(row, "auc"), fmt(row, "pf"), fmt(row, "pd")])
    add_table(doc, "Table 3. Calibration-matched global performance (mean ± 95% interval across ten seeds).",
              ["L", "Method", "AUC", "Pᶠ", "Pᵈ"], matched_rows,
              widths=[1.0, 5.2, 3.0, 3.0, 3.0], font_size=7.8,
              note="Thresholds are selected only on disjoint H₀ calibration data. The bilinear-global and C³FAR rows share a score but use global and contextual thresholds, respectively.",
              page_break_before=True)

    add_body(doc, (
        "The ten-seed results preserve the original trends while narrowing uncertainty. At L = 128, ENR-ED improves Pᵈ over ED from 0.318 ± 0.005 to 0.396 ± 0.005 without increasing Pᶠ. The signal-only MLP reaches 0.476 ± 0.005 and linear fusion reaches 0.530 ± 0.006. Bilinear fusion has AUC 0.782 ± 0.003 and Pᵈ 0.535 ± 0.006. C³FAR-Sense produces Pᵈ 0.533 ± 0.007 at Pᶠ 0.102 ± 0.005. Thus the interaction expansion and contextual thresholds add little under matched conditions; the gain under shift cannot be attributed to an easier matched operating point."
    ))
    add_body(doc, (
        "Figure 2 locates the benefit mainly between −12 and −4 dB, where structural and relative-energy evidence remain informative but ED is limited by sampled noise uncertainty. Longer windows improve every learned method. The generator nevertheless prevents near-perfect aggregate results by varying noise power, channel gain, carrier offset, and I/Q imbalance independently."
    ))

    doc.add_heading("6.2. Unseen receiver-mismatch stress test and ablations", level=2)
    add_figure(doc, FIGURES / "fig3_stress_pd_snr.png",
               "Fig. 3. L = 128 stress detection versus nominal SNR at α = 0.10. Refreshed methods use the same independent 3000-window H₀ bank. The adaptive-global and contextual rules share frozen bilinear scores and differ only in threshold assignment.")
    stress_models = [
        ("Bilinear fusion + global conformal", "Bilinear, frozen threshold"),
        ("ENR-ED + adaptive conformal", "ENR-ED, global refresh"),
        ("Bilinear fusion + adaptive global", "Bilinear, global refresh"),
        ("Linear fusion + context conformal", "Linear, supervised context"),
        ("Bilinear fusion + KMeans conformal", "Bilinear, K-means context"),
        ("Bilinear fusion + oracle conformal", "Bilinear, oracle context"),
        ("C3FAR-Sense", "C³FAR-Sense"),
    ]
    stress_rows = []
    for model, label in stress_models:
        row = get_row(global_df, length=128, split="stress", model=model)
        stress_rows.append([label, fmt(row, "auc"), fmt(row, "pf"), fmt(row, "pd"), fmt(row, "balanced_accuracy")])
    add_table(doc, "Table 4. L = 128 global performance under unseen receiver mismatch.",
              ["Operating rule", "AUC", "Pᶠ", "Pᵈ", "Balanced accuracy"], stress_rows,
              widths=[5.8, 2.6, 2.6, 2.6, 3.3], font_size=7.6,
              note="The frozen-threshold row is diagnostic only: its Pᵈ is not comparable because Pᶠ = 0.395. K-means uses no regime labels; oracle context uses simulator labels only for an evaluation ceiling.")

    length_rows = []
    for length in (64, 128, 256):
        ag = get_row(global_df, length=length, split="stress", model="Bilinear fusion + adaptive global")
        km = get_row(global_df, length=length, split="stress", model="Bilinear fusion + KMeans conformal")
        c3 = get_row(global_df, length=length, split="stress", model="C3FAR-Sense")
        length_rows.append([str(length), fmt(ag, "pf"), fmt(ag, "pd"), fmt(km, "pf"), fmt(km, "pd"), fmt(c3, "pf"), fmt(c3, "pd")])
    add_table(doc, "Table 5. Context-calibration gains across native window lengths.",
              ["L", "Global Pᶠ", "Global Pᵈ", "K-means Pᶠ", "K-means Pᵈ", "C³FAR Pᶠ", "C³FAR Pᵈ"],
              length_rows, widths=[0.9, 2.55, 2.55, 2.55, 2.55, 2.55, 2.55], font_size=7.25)

    comparison_rows = []
    for _, row in paired_df.iterrows():
        short_b = {
            "Bilinear fusion + adaptive global": "Global refresh",
            "Bilinear fusion + KMeans conformal": "K-means context",
            "Linear fusion + context conformal": "Linear context",
        }[row["method_b"]]
        p_text = fmt_p(row["p_value_holm"])
        comparison_rows.append([
            str(int(row["length"])), row["split"].capitalize(), short_b,
            f"{100*row['mean_paired_difference']:.2f} ± {100*row['ci95']:.2f}",
            f"{row['effect_size_dz']:.2f}", p_text,
        ])
    add_table(doc, "Table 6. Seed-paired Pᵈ comparisons for C³FAR-Sense minus the stated comparator.",
              ["L", "Split", "Comparator", "Difference (percentage points)", "d_z", "Holm p"],
              comparison_rows, widths=[1.0, 1.6, 3.7, 5.2, 1.7, 2.5], font_size=7.5,
              note="Two-sided paired t-tests use the same ten generated datasets for both methods. Holm adjustment covers the six planned comparisons shown.")

    main_pair = get_row(paired_df, length=128, split="stress", metric="pd",
                        method_a="C3FAR-Sense", method_b="Bilinear fusion + adaptive global")
    kmeans_pair = get_row(paired_df, length=128, split="stress", metric="pd",
                          method_a="C3FAR-Sense", method_b="Bilinear fusion + KMeans conformal")
    add_body(doc, (
        f"At L = 128, a frozen threshold yields Pᶠ = 0.395 ± 0.035, confirming severe calibration failure. Global refresh restores Pᶠ to 0.099 ± 0.004 but yields Pᵈ = 0.355 ± 0.024. C³FAR-Sense reaches Pᶠ = 0.098 ± 0.002 and Pᵈ = 0.444 ± 0.014. The paired gain is {100*main_pair['mean_paired_difference']:.2f} ± {100*main_pair['ci95']:.2f} percentage points (d_z = {main_pair['effect_size_dz']:.2f}, Holm p = {fmt_p(main_pair['p_value_holm'])}). Gains remain significant at L = 64 and 256."
    ))
    add_body(doc, (
        f"Label-free K-means obtains Pᵈ = 0.444 ± 0.013 and oracle grouping obtains 0.446 ± 0.013. The supervised-minus-K-means difference is only {100*kmeans_pair['mean_paired_difference']:.2f} ± {100*kmeans_pair['ci95']:.2f} points (Holm p = {kmeans_pair['p_value_holm']:.3f}). The result sharpens the mechanism claim: partitioning heterogeneous H₀ scores is valuable, while pre-labeled regime supervision is not essential in this simulation. Linear-context calibration reaches Pᵈ = 0.403 ± 0.009, showing a smaller but significant residual contribution from evidence–context interactions under stress."
    ))

    doc.add_heading("6.3. Environment-resolved false alarms", level=2)
    add_figure(doc, FIGURES / "fig4_environment_pf.png",
               "Fig. 4. Environment-resolved Pᶠ at L = 128. Different panel scales expose matched imbalance and stress failure. The dashed line is α = 0.10; bars are ten-seed 95% intervals.")
    env_rows = []
    for split, environments in (("matched", ["white", "colored", "impulsive"]),
                                ("stress", ["colored-severe", "impulsive-severe", "student-t"])):
        for env in environments:
            adapt = get_row(env_df, length=128, split=split, environment=env,
                            model="Bilinear fusion + adaptive global")
            c3 = get_row(env_df, length=128, split=split, environment=env, model="C3FAR-Sense")
            env_rows.append([
                split.capitalize(), env.replace("-", " "), fmt(adapt, "pf"), fmt(adapt, "pd"),
                fmt(c3, "pf"), fmt(c3, "pd")
            ])
    add_table(doc, "Table 7. Environment-resolved operating performance at L = 128.",
              ["Split", "True environment", "Global Pᶠ", "Global Pᵈ", "C³FAR Pᶠ", "C³FAR Pᵈ"],
              env_rows, widths=[1.6, 3.4, 2.8, 2.8, 2.8, 2.8], font_size=7.5,
              note="True environments are simulator diagnostics. Proposition 1 conditions on predicted groups, which are analyzed in Section 6.6.")
    add_body(doc, (
        "A global threshold meets the aggregate target by allocating errors very unevenly. In matched data its environment Pᶠ values are 0.059, 0.172, and 0.072; C³FAR-Sense compresses them to 0.094–0.110. Under stress, global refresh assigns Pᶠ = 0.288 to severe colored noise but only 0.001 and 0.008 to severe impulsive and Student-t noise. Context calibration changes these to 0.101, 0.070, and 0.123. The largest absolute environment error therefore falls from 0.188 to about 0.031 at α = 0.10 when calculated seedwise."
    ))
    add_body(doc, (
        "The redistribution is not a free improvement in every environment. For severe colored noise, C³FAR-Sense reduces Pᵈ to 0.228 ± 0.025 while correcting the global rule's excessive Pᶠ; in the two heavy-tailed environments it removes excessive conservatism and raises Pᵈ above 0.54. The method is therefore a context-aware allocation of false-alarm risk, not a claim of pointwise dominance."
    ))

    doc.add_heading("6.4. Calibration-bank size", level=2)
    add_figure(doc, FIGURES / "fig5_calibration_size.png",
               "Fig. 5. L = 128 stress sensitivity to H₀ refresh-bank size M. Context methods divide the bank among three groups. Bars show ten-seed 95% intervals.")
    cal_rows = []
    for ncal in (100, 250, 500, 1000, 3000):
        ag = get_row(cal_df, length=128, method="Adaptive global", n_cal=ncal)
        km = get_row(cal_df, length=128, method="KMeans conformal", n_cal=ncal)
        c3 = get_row(cal_df, length=128, method="C3FAR-Sense", n_cal=ncal)
        cal_rows.append([str(ncal), fmt(ag, "pf"), fmt(ag, "pd"), fmt(km, "pf"), fmt(km, "pd"), fmt(c3, "pf"), fmt(c3, "pd")])
    add_table(doc, "Table 8. Calibration-size sensitivity in the L = 128 stress mixture.",
              ["M", "Global Pᶠ", "Global Pᵈ", "K-means Pᶠ", "K-means Pᵈ", "C³FAR Pᶠ", "C³FAR Pᵈ"],
              cal_rows, widths=[1.0, 2.55, 2.55, 2.55, 2.55, 2.55, 2.55], font_size=7.1,
              note="Small groups can return conservative order statistics; uncertainty therefore widens at small M.")
    add_body(doc, (
        "At M = 100, C³FAR-Sense is conservative on average (Pᶠ = 0.086 ± 0.016) but already reaches Pᵈ = 0.424 ± 0.024. From M = 250 onward its mean Pᶠ stays within 0.006 of the target, and at M = 1000 it reaches 0.099 ± 0.007 with Pᵈ = 0.443 ± 0.014. K-means follows the same curve. This evidence supports refresh from a few hundred windows in the present three-group problem, while warning that more or less balanced partitions require correspondingly larger banks."
    ))

    doc.add_heading("6.5. False-alarm target sweep", level=2)
    add_figure(doc, FIGURES / "fig7_alpha_sweep.png",
               "Fig. 6. L = 128 stress performance across target α. Context partitions preserve aggregate control while reducing the worst absolute true-environment deviation |Pᶠ_env−α|. Bars show ten-seed 95% intervals.")
    alpha_rows = []
    for alpha in (0.01, 0.05, 0.10, 0.20):
        for method, label in (("Adaptive global", "Global"), ("KMeans conformal", "K-means"),
                              ("C3FAR-Sense", "C³FAR"), ("Oracle conformal", "Oracle")):
            row = get_row(alpha_df, length=128, split="stress", method=method, alpha=alpha)
            alpha_rows.append([f"{alpha:.2f}", label, fmt(row, "pf"), fmt(row, "pd"),
                               fmt(row, "worst_environment_pf_error")])
    add_table(doc, "Table 9. Target-level sensitivity and worst-environment calibration error.",
              ["α", "Partition", "Aggregate Pᶠ", "Pᵈ", "Worst |Pᶠ_env−α|"],
              alpha_rows, widths=[1.2, 3.2, 3.5, 3.5, 4.4], font_size=7.4,
              note="Oracle uses true simulator environments. Worst-environment error is computed within each seed before averaging.")
    add_body(doc, (
        "Aggregate Pᶠ tracks all four targets for every refreshed rule, but the global threshold hides substantial conditional error. At α = 0.05 its worst-environment deviation is 0.097 ± 0.013, versus 0.021 ± 0.009 for C³FAR-Sense. At α = 0.20 the corresponding values are 0.368 ± 0.019 and 0.052 ± 0.019. Context calibration also increases Pᵈ at every target. Its small aggregate overshoot at α = 0.01 (0.011 ± 0.002) is compatible with Monte Carlo uncertainty and does not replace the conditional exchangeability requirement."
    ))

    doc.add_heading("6.6. Gate behavior and modulation consistency", level=2)
    add_figure(doc, FIGURES / "fig10_gate_confusion.png",
               "Fig. 7. Reference-only gate confusion matrices at L = 128. Columns denote white-like, colored-like, and impulsive-like predicted groups. Stress families are mapped rather than treated as new supervised classes.", width=6.35)
    gate_rows = []
    for split, envs in (("matched", ["white", "colored", "impulsive"]),
                        ("stress", ["colored-severe", "impulsive-severe", "student-t"])):
        vals = []
        for env in envs:
            rates = [get_row(gate_df, length=128, split=split, true_environment=env,
                             predicted_group=g)["rate_mean"] for g in (0, 1, 2)]
            gate_rows.append([split.capitalize(), env.replace("-", " ")] + [f"{100*x:.1f}%" for x in rates])
    add_table(doc, "Table 10. Mean gate allocation rates at L = 128.",
              ["Split", "True environment", "White-like", "Colored-like", "Impulsive-like"],
              gate_rows, widths=[1.8, 4.0, 3.4, 3.4, 3.4], font_size=7.7)
    add_body(doc, (
        "The gate assigns 88.9% of matched white references to the white-like group and 98.7% of colored references to the colored-like group. Matched impulsive references are split between impulsive-like (63.3%) and white-like (36.3%), which explains why predicted groups need not equal physical classes. Under stress, severe colored noise maps entirely to colored-like, whereas severe impulsive and Student-t noise map 99.5% and 98.7% to impulsive-like. The stress white-like group contains only 24.8 H₀ windows on average and therefore has a wide Pᶠ interval; the dominant groups have about 1498 and 2977 H₀ windows and Pᶠ near 0.10."
    ))
    add_figure(doc, FIGURES / "fig8_modulation_pd.png",
               "Fig. 8. L = 128 stress Pᵈ by modulation at α = 0.10. Context thresholds improve every waveform family; bars show ten-seed 95% intervals.", width=6.35)
    modulation_rows = []
    for modulation in ("BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64", "CPFSK", "GFSK", "AM-DSB", "WBFM"):
        ag = get_row(modulation_df, length=128, split="stress",
                     model="Bilinear fusion + adaptive global", modulation=modulation)
        c3 = get_row(modulation_df, length=128, split="stress", model="C3FAR-Sense", modulation=modulation)
        modulation_rows.append([modulation, fmt(ag, "pd"), fmt(c3, "pd"),
                                f"{100*(c3['pd_mean']-ag['pd_mean']):.2f}"])
    add_table(doc, "Table 11. Modulation-resolved stress detection at L = 128.",
              ["Modulation", "Global Pᵈ", "C³FAR Pᵈ", "Gain (points)"],
              modulation_rows, widths=[3.6, 4.2, 4.2, 3.7], font_size=7.7)
    add_body(doc, (
        "C³FAR-Sense ranges from Pᵈ = 0.437 for CPFSK and PAM4 to 0.454 for 8PSK, with no isolated weak waveform. It improves every modulation over global refresh by roughly 7.4–9.9 percentage points. Because modulation labels are never input to the detector or threshold rule, the consistency indicates that the gain comes from noise-context allocation rather than a favored waveform family."
    ))

    doc.add_heading("6.7. Reference mismatch and contamination boundaries", level=2)
    add_figure(doc, FIGURES / "fig9_reference_robustness.png",
               "Fig. 9. Reference robustness at L = 128. Left: test-time power mismatch is increased without refreshing thresholds. Right: signal leakage contaminates reference windows; the exploratory guard is calibrated to a 1% clean-reference flag rate. These are failure-boundary tests, not guaranteed operating regimes.", width=6.45)
    mismatch_rows = []
    for sigma in (0.0, 0.3, 0.85, 1.5, 3.0):
        ag = get_row(mismatch_df, length=128, method="Adaptive global", sigma_delta_db=sigma)
        c3 = get_row(mismatch_df, length=128, method="C3FAR-Sense", sigma_delta_db=sigma)
        mismatch_rows.append([f"{sigma:.2f}", fmt(ag, "pf"), fmt(ag, "pd"), fmt(c3, "pf"), fmt(c3, "pd")])
    add_table(doc, "Table 12. Unrefreshed test-time reference-power mismatch.",
              ["Mismatch σδ (dB)", "Global Pᶠ", "Global Pᵈ", "C³FAR Pᶠ", "C³FAR Pᵈ"],
              mismatch_rows, widths=[3.1, 3.3, 3.3, 3.3, 3.3], font_size=7.6,
              note="The H₀ refresh bank retains the nominal 0.85 dB mismatch law. Rows away from 0.85 dB therefore deliberately violate calibration/test exchangeability.")
    contamination_rows = []
    for rate in (0.0, 0.01, 0.05, 0.10, 0.20):
        row = get_row(contamination_df, length=128, contamination_rate=rate)
        flag = "—" if pd.isna(row["flag_tpr_mean"]) else fmt(row, "flag_tpr")
        contamination_rows.append([
            f"{100*rate:.0f}%", fmt(row, "raw_pf"), fmt(row, "raw_pd"),
            fmt(row, "guarded_pf"), fmt(row, "guarded_pd"), flag,
        ])
    add_table(doc, "Table 13. Signal contamination of test reference windows.",
              ["Contamination", "Raw Pᶠ", "Raw Pᵈ", "Guarded Pᶠ", "Guarded Pᵈ", "Flag TPR"],
              contamination_rows, widths=[2.4, 2.8, 2.8, 2.9, 2.9, 3.0], font_size=7.4,
              note="Leakage SNR is uniform from −8 to 2 dB. The guard's clean-reference flag rate is approximately 0.97%; flag TPR is undefined at zero contamination.")
    add_body(doc, (
        "The nominal stress point is σδ = 0.85 dB. Without threshold refresh, C³FAR-Sense Pᶠ rises from 0.098 there to 0.124 at 1.5 dB and 0.196 at 3 dB. The accompanying Pᵈ increase is not a robustness gain because it is purchased by violating the false-alarm constraint. This experiment directly illustrates the theorem's boundary: exchangeability must be restored by recalibration when the reference-mismatch law changes."
    ))
    add_body(doc, (
        "Reference contamination produces a different failure. At 20% leakage, raw Pᵈ falls from 0.444 to 0.388 while Pᶠ becomes conservative at 0.082. The simple guard recovers Pᵈ to 0.421 but raises Pᶠ to 0.131 and detects only about 22% of contaminated references. It is therefore inadequate as a complete remedy. The defensible operational response is fail-closed acquisition from a protected guard resource, explicit bank-integrity monitoring, and hardware validation of a stronger contamination detector."
    ))

    doc.add_heading("6.8. Parameter count and runtime", level=2)
    complexity_rows = [
        ["ED", "1 statistic", "0", "No", "Threshold"],
        ["ENR-ED", "1 ratio", "0", "Yes", "Threshold"],
        ["Signal-only MLP", "20", "3457", "No", "Threshold"],
        ["Linear fusion", "53", "54", "Yes", "Threshold"],
        ["Bilinear scorer", "483", "484", "Yes", "Global threshold"],
        ["C³FAR-Sense", "483 + 10-input gate", "517", "Yes", "3 thresholds"],
    ]
    add_table(doc, "Table 14. Learned complexity of compared models.",
              ["Model", "Input dimension", "Trainable parameters", "Uses r", "Refreshed state"],
              complexity_rows, widths=[3.7, 3.5, 3.1, 2.0, 3.8], font_size=8.0,
              note="Standardization means/scales and empirical thresholds are stored constants, not trainable model parameters. The C³FAR total is 484 scorer plus 33 gate parameters.")
    add_figure(doc, FIGURES / "fig6_latency.png",
               "Fig. 10. Vectorized batch CPU timing for the frozen proposed detector. Measurements use batches of 4096 windows on an AMD EPYC 9V74 virtual CPU and are throughput, not real-time radio-hardware, measurements.", width=5.95)
    latency_rows = []
    for _, row in latency_df.iterrows():
        latency_rows.append([
            str(int(row["length"])), str(int(row["batch"])), f"{row['feature_ms_per_sample']:.4f}",
            f"{row['decision_ms_per_sample']:.4f}", f"{row['total_ms_per_sample']:.4f}"
        ])
    add_table(doc, "Table 15. Frozen-detector vectorized CPU latency (median of repeated batch measurements).",
              ["L", "Batch", "Feature ms/window", "Decision ms/window", "Total ms/window"],
              latency_rows, widths=[1.4, 2.4, 4.2, 4.2, 4.0], font_size=8.0,
              note="Feature extraction includes both y and r statistics and dominates runtime. Timing excludes sample acquisition, buffering, and calibration-bank collection.")
    add_body(doc, (
        "The measured total is 0.024, 0.044, and 0.087 ms per window for L = 64, 128, and 256 in a vectorized batch. Decision cost remains close to 0.002 ms, while feature extraction scales approximately with L. This confirms the intended complexity profile: threshold refresh and scoring are negligible compared with obtaining FFT and correlation descriptors. A streaming implementation would need incremental feature updates and hardware-specific timing before any latency claim could be made."
    ))

    doc.add_heading("7. Discussion", level=1)
    doc.add_heading("7.1. What the ablation says", level=2)
    add_body(doc, (
        "The strongest conclusion is about partitioned calibration, not model size. In matched data, linear and bilinear fusion have nearly identical AUC and operating Pᵈ. Under stress, using the same bilinear score with contextual rather than global thresholds adds 8.84 percentage points of Pᵈ at L = 128 and sharply reduces environment-wise Pᶠ dispersion. Linear-context calibration also outperforms bilinear-global calibration, confirming that threshold structure is the primary mechanism. The remaining 4.05-point advantage of bilinear over linear contextual calibration under stress indicates a secondary benefit from evidence–context interactions."
    ))
    add_body(doc, (
        "The paired reference contributes in two places. It stabilizes relative-energy and shape features, and it determines which order statistic should be used. Removing it reduces matched L = 128 Pᵈ from 0.533 to 0.476 despite the signal-only MLP having about 6.7 times as many parameters as the complete proposed detector. ENR-ED confirms that reference energy alone is insufficient. Meanwhile, K-means matches the supervised gate and nearly reaches the oracle ceiling, implying that coarse H₀ score homogeneity can be discovered without annotated receiver regimes in this setting."
    ))

    doc.add_heading("7.2. Relationship to the base paper", level=2)
    add_body(doc, (
        "The base paper [1] improves how a CNN–RNN accumulates evidence in a short window. C³FAR-Sense improves how any score is converted into an operating decision across receiver contexts. The approaches are therefore complementary rather than direct substitutes. A future experiment could replace the compact scorer with InitStateSenseNet and retain the proposed reference gate and groupwise thresholds. Doing so would test whether the base architecture's high matched-benchmark AUC translates into better Pᵈ after calibration under realistic H₀ diversity. The present study deliberately avoids that combination so that the value of calibration is not confounded by a large backbone."
    ))

    doc.add_heading("7.3. Deployment interpretation", level=2)
    add_body(doc, (
        "Four deployment conditions are non-negotiable. First, the reference channel must be protected from incumbent energy; the contamination experiment shows that a weak integrity classifier is not enough. Second, calibration pairs should be independent enough that their ranks are approximately exchangeable. Third, the reference-mismatch law must remain represented by the active bank; otherwise rising Pᶠ can be mistaken for sensitivity. Fourth, each context group needs adequate support. A monitoring layer should record group counts, score and reference-integrity histograms, mismatch alarms, and time since calibration, and should default to no transmission when the bank is suspect or too small."
    ))
    add_body(doc, (
        "The method is compatible with regulatory asymmetry. If missing an incumbent is more costly than false alarms, α can be lowered or a second conformal constraint can be introduced for false negatives, following the risk-control direction in [16]. That extension requires labeled H₁ calibration data and is outside the current H₀-only refresh design."
    ))

    doc.add_heading("7.4. Reproducibility and extensibility", level=2)
    add_body(doc, (
        "The frozen release separates two evidence layers. Raw paired I/Q enables independent feature engineering and neural replacements, while the derived 43+10 feature archive permits rapid verification of the proposed scorer and gate. Sample identifiers, split labels, hypotheses, SNR, modulation, noise environment, and seed metadata prevent accidental train/test leakage. SHA-256 hashes and feature spot checks make silent file changes detectable. The same scripts can regenerate the larger ten-seed paper study or run a smaller smoke test."
    ))
    add_body(doc, (
        "This structure also makes the method modular. A future study can replace the bilinear score, the context partition, or the acquisition model independently while retaining the same H₀-only operating protocol. Comparisons should continue to distinguish ranking changes from threshold changes and should report Pᶠ before interpreting Pᵈ."
    ))

    doc.add_heading("8. Limitations and threats to validity", level=1)
    add_body(doc, (
        "The evidence is entirely simulation-based. Although the generator includes modulation, fading, frequency offset, I/Q imbalance, noise uncertainty, correlation, impulses, and heavy tails, it cannot reproduce every analog artifact, adjacent-channel interferer, automatic-gain-control transient, quantizer, oscillator instability, or antenna coupling found in a software-defined radio. The latency benchmark is vectorized CPU throughput and excludes acquisition. Hardware capture is required before operational deployment."
    ))
    add_body(doc, (
        "The reference-window assumption is both a strength and a scope limit. Some systems have a protected guard subband or quiet schedule; others do not. At 20% signal leakage, the exploratory guard detects only about 22% of contaminated references and trades partial Pᵈ recovery for Pᶠ = 0.131. This negative result rules out presenting the guard as a solved integrity mechanism. K-means removes the supervised gate-label requirement empirically, but its clusters can permute or become imbalanced across sites; the finite-sample claim remains conditional on the resulting partition and fresh exchangeable H₀ data."
    ))
    add_body(doc, (
        "The proposition requires exchangeability within predicted groups. The unseen stress experiment demonstrates empirical recovery only after a fresh H₀ bank is collected; it is not a theorem for arbitrary covariate shift. The deliberate mismatch sweep confirms this boundary: with an unchanged bank, C³FAR Pᶠ reaches 0.196 at 3 dB reference mismatch. Environment-specific Pᶠ is conditioned on simulator labels, whereas the guarantee is conditioned on predicted groups. Residual Student-t error and the sparsely populated white-like stress group show that a three-way partition does not align perfectly with every physical regime. More groups could reduce heterogeneity but increase calibration demand."
    ))
    add_body(doc, (
        "Ten seeds support paired inference but remain insufficient for rare-tail certification or broad hardware generalization. The baseline set is designed to isolate energy normalization, reference features, interaction modeling, and threshold structure; it is not exhaustive of transformers, fused time–frequency networks, cyclostationary detectors, or eigenvalue methods. The released 15,000-pair dataset is smaller than the full paper simulation and is intended for reproducibility rather than substituting for all run-level samples. Finally, the features, stress ranges, target grid, and guard were fixed for this study. Preregistered external replication and over-the-air captures should test sensitivity to these choices."
    ))

    doc.add_heading("9. Conclusion", level=1)
    add_body(doc, (
        "This paper introduced C³FAR-Sense, a short-window spectrum sensor that treats receiver context and false-alarm calibration as parts of the detector rather than post-processing choices. A paired noise-reference window preserves relative energy, conditions a compact bilinear score, and selects a groupwise conformal threshold. Under within-group H₀ exchangeability, the threshold supplies a finite-sample Pᶠ guarantee; under receiver drift, it can be refreshed without changing learned weights."
    ))
    add_body(doc, (
        "Ten-seed simulations show that context-conditioned calibration matters most under mismatch. At L = 128, it improves stress Pᵈ from 0.355 to 0.444 relative to a globally refreshed version of the identical score while retaining Pᶠ near 0.10; the 8.84-point paired gain remains significant after Holm correction. It reduces worst-environment Pᶠ error from 0.188 to 0.031, and the pattern persists across window lengths and α from 0.01 to 0.20. Label-free K-means performs statistically indistinguishably from the supervised gate, and the oracle adds little, which locates the main value in partitioned calibration rather than privileged labels."
    ))
    add_body(doc, (
        "The extended experiments also define where the method should not be trusted. Unrefreshed reference mismatch breaks exchangeability, and a simple contamination guard is not sufficient. The practical contribution is therefore a small, auditable, threshold-refreshable sensing layer with an explicit operational contract—not a claim of universal out-of-distribution validity. The accompanying raw-I/Q data, derived features, scripts, summaries, checksums, and validation routine make that contract reproducible. The next step is preregistered over-the-air evaluation with protected reference acquisition, correlated streaming windows, and end-to-end radio latency."
    ))

    doc.add_heading("Declarations", level=1)
    doc.add_heading("Funding", level=2)
    add_body(doc, "[Insert funding statement. If none: “This research received no specific grant from funding agencies in the public, commercial, or not-for-profit sectors.”]", first_line=False)
    doc.add_heading("Declaration of competing interest", level=2)
    add_body(doc, "The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.", first_line=False)
    doc.add_heading("CRediT authorship contribution statement", level=2)
    add_body(doc, "[Author 1]: Conceptualization, Methodology, Software, Formal analysis, Writing – original draft. [Author 2]: Validation, Investigation, Writing – review & editing. [Author 3]: Supervision, Resources, Writing – review & editing. Replace roles to reflect actual contributions before submission.", first_line=False)
    doc.add_heading("Data and code availability", level=2)
    add_body(doc, "The submission package includes executable Python code, dependency specifications, all ten-seed summary and run-level tables, figure-generation code, and a frozen synthetic release with 15,000 paired raw-I/Q sensing/reference windows, derived evidence/context features, sample metadata, manifests, and SHA-256 hashes. The included validation script checks file integrity, shapes, feature recomputation, seeds, and window lengths. A permanent archival DOI or repository URL should be inserted here after deposit and before publication.", first_line=False)
    doc.add_heading("Ethics statement", level=2)
    add_body(doc, "This study used computer-generated communication signals and did not involve human participants, personal data, animals, or clinical intervention.", first_line=False)

    appendix_a = doc.add_heading("Appendix A. Feature inventory", level=1)
    appendix_a.paragraph_format.page_break_before = True
    feature_rows = [
        ["Evidence", "Robust amplitude and energy ratios", "3", "Mean energy, median power, and 0.9-quantile ratios y:r"],
        ["Evidence", "Global spectral descriptors", "6", "Entropy, flatness, and peak; y values and reference contrasts"],
        ["Evidence", "Tail and amplitude shape", "4", "Kurtosis and crest factor; y values and reference contrasts"],
        ["Evidence", "Circular and phase structure", "4", "Pseudo-covariance, fourth circular moment, two phase concentrations"],
        ["Evidence", "Temporal correlation", "10", "Normalized ACF and reference difference at lags 1,2,4,8,16"],
        ["Evidence", "Eight-band spectral shape", "16", "Normalized y band power and log y:r ratio in each band"],
        ["Context", "Power/spectral/tail descriptors", "7", "Log power, entropy, flatness, peak, kurtosis, crest, impulse fraction"],
        ["Context", "Temporal correlation", "3", "Reference normalized ACF at lags 1,2,4"],
    ]
    add_table(doc, "Table A.1. Grouped inventory of the 43 evidence and 10 context features.",
              ["Vector", "Feature group", "Count", "Contents"], feature_rows,
              widths=[2.1, 4.3, 1.3, 8.5], font_size=7.8,
              note="The bilinear expansion contains 43×10 = 430 interactions; 43 + 10 + 430 = 483 scorer inputs.")

    doc.add_heading("Appendix B. Reproducible detector protocol", level=1)
    algo_rows = [
        ["1", "Generate or collect disjoint training, H₀ calibration, and test sets for each L."],
        ["2", "Compute robustly scaled evidence e and reference context c; fit standardization on training only."],
        ["3", "Fit β on z = [e;c;vec(ecᵀ)] with regularized binary cross-entropy."],
        ["4", "Fit supervised reference-only gate θ, or fit the label-free K-means partition on training contexts."],
        ["5", "For each H₀ calibration pair, compute sβ and predicted group ĝ(c)."],
        ["6", "For each nonempty group, store τg using the order statistic in (11)."],
        ["7", "For a new pair, declare H₁ iff sβ(y,r) > τĝ(c)."],
        ["8", "After detected receiver drift, freeze β and θ; repeat steps 5–6 on a new trusted H₀ bank."],
        ["9", "Log group counts; abstain from transmission if reference integrity or calibration support is inadequate."],
    ]
    add_table(doc, "Algorithm B.1. Offline training, online decision, and threshold-only refresh.",
              ["Step", "Operation"], algo_rows, widths=[1.3, 14.9], font_size=8.1)

    doc.add_heading("Appendix C. Frozen reproducibility dataset", level=1)
    release_rows = [
        ["train_matched", "6000", "Balanced H₀/H₁ model fitting"],
        ["calibration_h0_matched", "1500", "Matched H₀ threshold calibration"],
        ["refresh_h0_stress", "1500", "Stress H₀ threshold refresh"],
        ["test_matched", "3000", "Balanced matched evaluation"],
        ["test_stress", "3000", "Balanced unseen-stress evaluation"],
    ]
    add_table(doc, "Table C.1. C³FAR-Sense Paired-IQ Benchmark v1 split inventory (L = 128).",
              ["Split", "Pairs", "Purpose"], release_rows,
              widths=[5.0, 2.0, 9.0], font_size=8.0,
              note="Every pair contains independent sensing and reference arrays with I and Q channels. Calibration and refresh splits contain H₀ only by design.")
    add_body(doc, (
        "The raw archive stores sensing_iq and reference_iq as float32 arrays of shape 15,000×128×2. The feature archive stores evidence (15,000×43), context (15,000×10), names, labels, split codes, environment codes, modulation codes, nominal SNR, and sample identifiers. The 430 bilinear interactions are reconstructed on demand to avoid redundant storage. A CSV mirrors the sample-level metadata for inspection without NumPy."
    ))
    add_body(doc, (
        "The manifest records generator version, seed 260908, schemas, label dictionaries, file sizes, and SHA-256 hashes. Validation recomputes all 53 features for a deterministic 128-row subset and verifies them against the stored archive within floating-point tolerance. The paper-scale summary manifest separately verifies ten seeds and L ∈ {64,128,256}; the frozen benchmark is intentionally compact and does not numerically duplicate every full-study run."
    ))
    doc.add_heading("References", level=1)
    add_reference(doc, 1, "M.K. Dixit, A.K. Singh, S. Dixit, InitStateSenseNet: Input-conditioned CNN–RNN for window-based spectrum sensing, Physical Communication 77 (2026) 103125.", doi="10.1016/j.phycom.2026.103125")
    add_reference(doc, 2, "I.F. Akyildiz, W.-Y. Lee, M.C. Vuran, S. Mohanty, NeXt generation/dynamic spectrum access/cognitive radio wireless networks: A survey, Computer Networks 50 (13) (2006) 2127–2159.", doi="10.1016/j.comnet.2006.05.001")
    add_reference(doc, 3, "E. Axell, G. Leus, E.G. Larsson, H.V. Poor, Spectrum sensing for cognitive radio: State-of-the-art and recent advances, IEEE Signal Processing Magazine 29 (3) (2012) 101–116.", doi="10.1109/MSP.2012.2183771")
    add_reference(doc, 4, "R. Tandra, A. Sahai, SNR walls for signal detection, IEEE Journal of Selected Topics in Signal Processing 2 (1) (2008) 4–17.", doi="10.1109/JSTSP.2007.914879")
    add_reference(doc, 5, "Y. Zeng, Y.-C. Liang, Eigenvalue-based spectrum sensing algorithms for cognitive radio, IEEE Transactions on Communications 57 (6) (2009) 1784–1793.", doi="10.1109/TCOMM.2009.06.070402")
    add_reference(doc, 6, "W.A. Gardner, Exploitation of spectral redundancy in cyclostationary signals, IEEE Signal Processing Magazine 8 (2) (1991) 14–36.", doi="10.1109/79.81007")
    add_reference(doc, 7, "J. Gao, X. Yi, C. Zhong, X. Chen, Z. Zhang, Deep learning for spectrum sensing, IEEE Wireless Communications Letters 8 (6) (2019) 1727–1730.", doi="10.1109/LWC.2019.2939314")
    add_reference(doc, 8, "Q. Cheng, Z. Shi, D.N. Nguyen, E. Dutkiewicz, Sensing OFDM signal: A deep learning approach, IEEE Transactions on Communications 67 (11) (2019) 7785–7798.", doi="10.1109/TCOMM.2019.2940013")
    add_reference(doc, 9, "J. Xie, J. Fang, C. Liu, X. Li, Deep learning-based spectrum sensing in cognitive radio: A CNN-LSTM approach, IEEE Communications Letters 24 (10) (2020) 2196–2200.", doi="10.1109/LCOMM.2020.3002073")
    add_reference(doc, 10, "S. Mondal, M.P. Dutta, S.K. Chakraborty, A hybrid deep learning based approach for spectrum sensing in cognitive radio, Physical Communication 67 (2024) 102497.", doi="10.1016/j.phycom.2024.102497")
    add_reference(doc, 11, "S.E. Abdelbaset, H.M. Kasem, A.A. Khalaf, A.H. Hussein, A.A. Kabeel, Deep learning-based spectrum sensing for cognitive radio applications, Sensors 24 (24) (2024) 7907.", doi="10.3390/s24247907")
    add_reference(doc, 12, "Y. Zhang, H. Luo, A deep-learning-based method for spectrum sensing with multiple feature combination, Electronics 13 (14) (2024) 2705.", doi="10.3390/electronics13142705")
    add_reference(doc, 13, "T.J. O'Shea, N. West, Radio machine learning dataset generation with GNU Radio, Proceedings of the GNU Radio Conference 1 (1) (2016).", url="https://pubs.gnuradio.org/index.php/grcon/article/view/11")
    add_reference(doc, 14, "V. Vovk, A. Gammerman, G. Shafer, Algorithmic Learning in a Random World, Springer, New York, 2005.", doi="10.1007/b106715")
    add_reference(doc, 15, "A.N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, Conformal risk control, in: International Conference on Learning Representations, 2024.", url="https://openreview.net/forum?id=33XGfHLtZg")
    add_reference(doc, 16, "H. Lee, S. Park, O. Simeone, Y.C. Eldar, J. Kang, Reliable sub-Nyquist spectrum sensing via conformal risk control, arXiv:2405.17071 (2024).", doi="10.48550/arXiv.2405.17071")
    add_reference(doc, 17, "S. Yoo, S. Park, P. Popovski, J. Kang, O. Simeone, Calibrating wireless AI via meta-learned context-dependent conformal prediction, arXiv:2501.14566 (2025).", doi="10.48550/arXiv.2501.14566")
    add_reference(doc, 18, "K.M. Cohen, S. Park, O. Simeone, S. Shamai, Calibrating AI models for wireless communications via conformal prediction, IEEE Transactions on Machine Learning in Communications and Networking 1 (2023) 296–312.", doi="10.1109/TMLCN.2023.3319282")
    add_reference(doc, 19, "R.F. Barber, E.J. Candès, A. Ramdas, R.J. Tibshirani, The limits of distribution-free conditional predictive inference, Information and Inference 10 (2) (2021) 455–482.", doi="10.1093/imaiai/iaaa017")
    add_reference(doc, 20, "P.J. Huber, E.M. Ronchetti, Robust Statistics, second ed., Wiley, Hoboken, 2009.", doi="10.1002/9780470434697")
    add_reference(doc, 21, "F. Pedregosa et al., Scikit-learn: Machine learning in Python, Journal of Machine Learning Research 12 (2011) 2825–2830.", url="https://jmlr.org/papers/v12/pedregosa11a.html")
    add_reference(doc, 22, "S.M. Kay, Fundamentals of Statistical Signal Processing, Volume II: Detection Theory, Prentice Hall, Upper Saddle River, 1998.")
    add_reference(doc, 23, "M. Falco, A. Scarvaglieri, F. Busacca, D. Croce, The evolution of dynamic spectrum sensing: A two-decade survey from foundations to frontiers, Computer Networks 278 (2026) 112095.", doi="10.1016/j.comnet.2026.112095")
    add_reference(doc, 24, "X. Bai, J. Xu, F. Fang, S. Wang, A multi-feature spectrum sensing algorithm based on GRU-AM, IEEE Wireless Communications Letters 14 (7) (2025) 2129–2133.", doi="10.1109/LWC.2025.3564053")
    add_reference(doc, 25, "T. Xu, Y. Wang, M. Yan, Q. Liang, Deep learning-based spectrum sensing with fused time-frequency representations, IEEE Communications Letters 29 (10) (2025) 2386–2390.", doi="10.1109/LCOMM.2025.3597786")

    # Make all body table rows non-splitting and normalize fonts in table cells.
    for table in doc.tables:
        for row in table.rows:
            prevent_row_split(row)

    doc.save(OUTPUT)
    print(OUTPUT)
    return OUTPUT


def main() -> None:
    global RESULTS, FIGURES, LATENCY, OUTPUT
    parser = argparse.ArgumentParser(
        description="Build the editable C3FAR-Sense Physical Communication manuscript."
    )
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="Directory containing the 10-seed result CSV files")
    parser.add_argument("--figures", type=Path, default=FIGURES,
                        help="Directory containing fig1 through fig10")
    parser.add_argument("--latency", type=Path, default=LATENCY,
                        help="CPU latency CSV produced by benchmark_latency.py")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="Destination .docx file")
    args = parser.parse_args()
    RESULTS = args.results.resolve()
    FIGURES = args.figures.resolve()
    LATENCY = args.latency.resolve()
    OUTPUT = args.output.resolve()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build()


if __name__ == "__main__":
    main()
