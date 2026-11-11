"""Export named sharing copies with common project content and individual Part IV."""
import argparse
import json
import re
from pathlib import Path

from retailassist.config import SETTINGS
from retailassist.reporting import build_reports, common_blocks, export_blocks, individual_blocks, safe_name, SITUATED_SECTIONS
from retailassist.training import dump

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=SETTINGS.root / "submission/group.json")
    parser.add_argument("--output-dir", type=Path, default=SETTINGS.root / "Group_28_Individual_Reports")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    group_id = safe_name(metadata["group_id"])
    members = metadata["members"]
    for member in members:
        missing = [key for key, _ in SITUATED_SECTIONS if not member.get("situated_learning", {}).get(key)]
        if missing: raise ValueError(f"Complete {member['name']}'s sections first: {', '.join(missing)}")
    stems = [safe_name(group_id + "_" + re.sub(r"[^A-Za-z0-9]+", "_", m["name"]).strip("_")) for m in members]
    if len(stems) != len(set(stems)): raise ValueError("Member names produce duplicate filenames.")
    # Retain the standard portal-name exports and their actual readiness record.
    status = build_reports(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = common_blocks(metadata)
    reports, table_rows = [], []
    for member, stem in zip(members, stems):
        target = args.output_dir / stem
        # Sharing copies omit administrative draft stamps; no review result is changed.
        export_blocks(common + individual_blocks(member), target, draft=False, author=member["name"])
        reports.extend(str(target.with_suffix(extension).resolve()) for extension in [".pdf", ".docx"])
        table_rows.append(f"| {member['name']} | {member['bits_id']} | [{stem}.pdf]({stem}.pdf) | [{stem}.docx]({stem}.docx) |")
    notes = [f"# Group {group_id} — Individual reports", "",
             "Each report contains the same shared project sections and a separate first-person Situated Learning section for its named member.", "",
             "| Member | Student ID | PDF | Editable Word |", "|---|---|---|---|", *table_rows, "",
             "Review the individual section and adapt any workplace-specific context before submitting. DemoMart is an industry analogue; the reports do not claim employment at a real retailer.", "",
             "Add the shared screen-recording link after your presentation is recorded and uploaded. The report retains the project's measured results and limitations.", "",
             f"The name-based files are convenient for sharing. For the assignment portal, each member should rename their own final copy to {group_id}.pdf or {group_id}.docx, following the assignment naming instruction.", "",
             "To regenerate these sharing copies after updating submission/group.json:", "",
             "```sh", ".venv/bin/python scripts/export_individual_reports.py", "```", ""]
    (args.output_dir / "README.md").write_text("\n".join(notes))
    manifest = {"group_id": group_id, "folder": str(args.output_dir.resolve()), "reports": reports,
                "individual_sections_complete": True, "submission_ready": status["submission_ready"],
                "pending": status["pending"], "quality_reviews_pending": status["quality_reviews_pending"],
                "automated_semantic_review_complete": status["automated_semantic_review_complete"]}
    dump(SETTINGS.artifacts / "individual_reports_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
