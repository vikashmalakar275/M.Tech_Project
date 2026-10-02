from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the running OrbitWatch dashboard.")
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    images = root / "docs" / "images"
    images.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=1)
        page.goto(args.url)
        expect(page.get_by_role("heading", name="OrbitWatch", exact=True)).to_be_visible(
            timeout=45000
        )
        expect(page.locator(".js-plotly-plot").first).to_be_visible(timeout=45000)
        expect(page.get_by_test_id("stException")).to_have_count(0)
        page.screenshot(path=str(images / "mission-console.png"))
        page.get_by_role("button", name="Investigate through MCP", exact=True).click()
        report_heading = page.get_by_role("heading", name="OrbitWatch investigation", exact=True)
        expect(report_heading).to_be_visible(timeout=60000)
        expect(page.get_by_test_id("stException")).to_have_count(0)
        report_heading.scroll_into_view_if_needed()
        page.screenshot(path=str(images / "investigation.png"))
        if args.llm:
            page.get_by_text("Evidence-validated local LLM", exact=True).click()
            expect(page.get_by_role("textbox", name="Ollama model", exact=True)).to_be_enabled()
            expect(report_heading).not_to_be_visible()
            page.get_by_role("button", name="Investigate through MCP", exact=True).click()
            expect(report_heading).to_be_visible(timeout=180000)
            expect(page.get_by_test_id("stException")).to_have_count(0)
            with page.expect_download() as download:
                page.get_by_role("button", name="Download evidence (.json)", exact=True).click()
            target = root / "local" / "browser-evidence.json"
            download.value.save_as(str(target))
            evidence = json.loads(target.read_text())
            assert evidence["engine"].startswith("ollama/"), evidence["engine"]
            assert evidence["accepted"], "The UI report must include supported claims."
            assert evidence["evidence"]["facts"]["physical_cause"] == "unknown"
        page.get_by_role("tab", name="EXPERIMENT RESULTS", exact=True).click()
        expect(
            page.get_by_role("heading", name="Measured results, not illustrative scores")
        ).to_be_visible()
        page.screenshot(path=str(images / "experiment-results.png"))
        print("Browser workflow passed: real plots, MCP report, export, and result view.")
        if args.llm:
            print("Local-LLM investigation was generated and its downloaded evidence verified.")
        browser.close()


if __name__ == "__main__":
    main()
