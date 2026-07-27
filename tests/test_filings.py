from synergy_estimator.corpus.filings import clean_text


def test_strips_markup_and_collapses_whitespace():
    html = "<html><body><p>Expected   $450 million</p></body></html>"

    assert clean_text(html) == "Expected $450 million"


def test_drops_script_and_style_content():
    html = "<body><style>p{color:red}</style><p>synergies</p><script>x()</script></body>"

    assert clean_text(html) == "synergies"


def test_rejoins_a_sentence_split_by_inline_markup():
    """Filings routinely wrap part of a phrase in a tag mid-sentence; the
    disclosure patterns only match if the text is reassembled."""
    html = "<p>$450 <font>million</font> in run-rate <b>cost</b> synergies</p>"

    assert clean_text(html) == "$450 million in run-rate cost synergies"


def test_separates_adjacent_table_cells():
    """Without a separator, adjacent cells would fuse into one nonsense token."""
    html = "<table><tr><td>Cost synergies</td><td>$450 million</td></tr></table>"

    assert clean_text(html) == "Cost synergies $450 million"


def test_handles_xml_exhibits():
    assert "synergies" in clean_text("<?xml version='1.0'?><root><a>synergies</a></root>")
