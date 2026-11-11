"""Capture actual browser workflows, without substituting fabricated model outputs."""
import json
import os
from playwright.sync_api import sync_playwright
from retailassist.config import SETTINGS

def capture():
    output = SETTINGS.artifacts / "screenshots"
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(os.environ.get("RETAILASSIST_UI_URL", "http://127.0.0.1:8501"), wait_until="networkidle")
        page.get_by_text("Resolve with context.", exact=True).wait_for(timeout=30000)
        page.get_by_role("combobox", name="Example ticket").click()
        page.get_by_role("option", name="Refund delay", exact=True).click()
        page.get_by_role("button", name="Analyze ticket", exact=True).click()
        page.get_by_text("Ticket summary", exact=True).wait_for(timeout=240000)
        page.get_by_role("button", name="Draft reply from policies", exact=True).click()
        page.get_by_role("textbox", name="Reply draft — edit before use").wait_for(timeout=240000)
        workspace_source = page.locator('[data-testid="stExpander"]').filter(has_text="Returns and refunds").first
        workspace_source.locator('summary').click()
        workspace_source.locator('[data-testid="stExpanderDetails"]').wait_for(state="visible")
        page.wait_for_timeout(500)
        page.screenshot(path=str(output / "01-workspace.png"), full_page=False)
        workspace_source.scroll_into_view_if_needed()
        page.screenshot(path=str(output / "06-workspace-source.png"), full_page=False)
        page.get_by_text("Policy chatbot", exact=True).click()
        page.get_by_role("textbox", name="Ask about refunds, shipping, accounts, or another store policy…").fill("How long does a refund take after inspection?")
        page.get_by_role("textbox", name="Ask about refunds, shipping, accounts, or another store policy…").press("Enter")
        page.locator('[data-testid="stChatMessage"]').nth(1).wait_for(timeout=240000)
        answer_source = page.locator('[data-testid="stChatMessage"]').nth(1).locator('[data-testid="stExpander"]').first
        answer_source.locator('summary').click()
        answer_source.locator('[data-testid="stExpanderDetails"]').wait_for(state="visible")
        page.wait_for_timeout(500)
        page.screenshot(path=str(output / "02-chatbot.png"), full_page=False)
        page.get_by_text("Knowledge library", exact=True).click()
        page.get_by_text("A shared source of clarity.", exact=True).wait_for()
        page.get_by_role("button", name="Open PDF").first.wait_for()
        page.screenshot(path=str(output / "03-knowledge.png"), full_page=False)
        page.get_by_text("Evaluation", exact=True).click()
        page.get_by_text("Evidence over assumptions.", exact=True).wait_for()
        page.get_by_text("Public intent macro-F1", exact=True).wait_for()
        page.screenshot(path=str(output / "04-evaluation.png"), full_page=False)
        page.get_by_text("Policy-answer evaluation", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(output / "05-policy-evaluation.png"), full_page=False)
        output.joinpath("browser-check.json").write_text(json.dumps({"page_errors": errors, "passed": not errors}, indent=2))
        browser.close()
        if errors: raise AssertionError(errors)
        print("Captured six actual application screens; no browser page errors.")

if __name__ == "__main__": capture()
