import re
from pathlib import Path
from xml.sax.saxutils import escape
from .config import SETTINGS

def policy_pdfs(settings=SETTINGS):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    out = settings.root / "data/policies/pdf"
    out.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles["BodyText"].leading = 17
    styles["BodyText"].spaceAfter = 12
    styles["Title"].textColor = colors.HexColor("#123d36")
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64766f"))
        canvas.drawString(48, 30, "FICTIONAL DEMONSTRATION POLICY | Effective 1 November 2026")
        canvas.drawRightString(A4[0]-48, 30, f"Page {document.page}")
        canvas.restoreState()
    paths = []
    for source in sorted((settings.root / "data/policies/source").glob("*.md")):
        text = source.read_text()
        title = text.splitlines()[0].removeprefix("# ")
        sections = re.split(r"^## ", text, flags=re.M)[1:]
        story = []
        for index, section in enumerate(sections):
            if index: story.append(PageBreak())
            heading, body = section.split("\n", 1)
            story.extend([Paragraph(escape(heading), styles["Title"]), Paragraph(escape(title), styles["Heading2"]), Spacer(1, 18)])
            for paragraph in body.strip().split("\n\n"):
                story.append(Paragraph(escape(paragraph.replace("\n", " ")), styles["BodyText"]))
        target = out / (source.stem + ".pdf")
        SimpleDocTemplate(str(target), pagesize=A4, title=title, author="RetailAssist educational project",
                          leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48).build(story, onFirstPage=footer, onLaterPages=footer)
        paths.append(str(target))
    return paths
