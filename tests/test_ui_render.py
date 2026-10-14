"""
Test UI rendering and header badge integrity for DDoSGuard Streamlit app.
"""

import os
from streamlit.testing.v1 import AppTest

def test_app_header_and_interaction():
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"Exceptions on initial run: {at.exception}"

    # Verify header renders without code-block leakage
    header_found = False
    for m in at.markdown:
        val = m.value
        if '<div class="soc-header">' in val:
            header_found = True
            assert "<pre>" not in val, "Header contains pre code block!"
            assert "<code>" not in val, "Header contains code block!"
            assert '<span class="soc-sub-badge">CHAMPION: XGBOOST (HIST)</span>' in val
            assert '<span class="soc-sub-badge">DATASET: CIC-IDS-2017</span>' in val

    assert header_found, "Header markdown element not found!"

    # Verify primary classify button
    classify_clicked = False
    for b in at.button:
        if "Inspect & Classify" in b.label:
            b.click().run()
            classify_clicked = True
            break

    assert classify_clicked, "Inspect & Classify button not found!"
    assert len(at.exception) == 0, f"Exceptions after classify: {at.exception}"

    # Verify preset loading
    benign_clicked = False
    for b in at.button:
        if "Benign" in b.label:
            b.click().run()
            benign_clicked = True
            break

    assert benign_clicked, "Benign preset button not found!"
    assert len(at.exception) == 0, f"Exceptions after benign click: {at.exception}"
