from rehab_minillm.corpus import build_pmc_query


def test_query_excludes_retractions_and_expressions_of_concern():
    query = build_pmc_query("stroke rehabilitation", '"cc by license"[filter]')
    assert "NOT articletyperetraction" in query
    assert "NOT hasretractionin" in query
    assert "NOT articletypeexpressionofconcern" in query
    assert "NOT hasexpressionofconcernin" in query
    assert "NOT articletypecorrection" in query
