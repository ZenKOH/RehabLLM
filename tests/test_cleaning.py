from rehab_minillm.cleaning import clean_document, detect_contamination


def test_cleaner_removes_publishing_boilerplate_and_markup():
    text = """Introduction {#sec1}
Robot-assisted rehabilitation may increase training repetitions.

Competing interests
The authors declare that they have no competing interests.

Author contributions
AB designed the study. CD analysed the data. All authors read and approved the final manuscript.

Conclusion
Clinical value depends on population, comparator and outcome.
"""
    before = detect_contamination(text)
    assert "heading:competing_interests" in before
    assert "markup:section_anchor" in before

    result = clean_document(text)
    assert "no competing interests" not in result.text.lower()
    assert "all authors read and approved" not in result.text.lower()
    assert "{#sec1}" not in result.text
    assert "Robot-assisted rehabilitation" in result.text
    assert "Conclusion" in result.text
    assert "Clinical value depends" in result.text


def test_cleaner_preserves_substantive_content_without_markers():
    text = "Introduction\nRehabilitation aims to improve functioning.\n\nResults\nFunction improved."
    result = clean_document(text)
    assert result.text == text
    assert result.removed_chars == 0
