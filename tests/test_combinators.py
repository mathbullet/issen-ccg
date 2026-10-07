from issen_ccg.domain.combinators import combine


def parents(left, right):
    return {str(result.category) for result in combine(left, right)}


def test_application_propagates_features_and_erases_nb():
    assert "NP" in parents("NP[nb]/N", "N")
    assert r"S[dcl]\NP" in parents(r"(S\NP)/(S\NP)", r"S[dcl]\NP")
    assert "S[dcl]" in parents("NP", r"S[dcl]\NP")
    assert not parents(r"(S[dcl]\NP)/(S[b]\NP)", r"S[ng]\NP")
    assert r"NP[nb]/N" in parents("NP", r"(NP[nb]/N)\NP")
    assert "N[num]" in parents("N/N", "N[num]")
    assert r"S[dcl]\NP[thr]" in parents(r"S[dcl]\NP[thr]", r"(S\NP)\(S\NP)")


def test_composition_and_crossed_composition():
    assert "S/NP" in parents(r"S/(S\NP)", r"(S\NP)/NP")
    assert r"(S[dcl]\NP)/NP" in parents(r"(S[dcl]\NP)/NP", r"(S\NP)\(S\NP)")
    assert "N/PP" not in parents("N/NP", r"N\N")


def test_coordination_punctuation_and_limited_type_change():
    assert "NP[conj]" in parents("conj", "NP")
    assert "NP" in parents("NP", "NP[conj]")
    assert "S/S" in parents("NP", ",")
    assert r"(S\NP)/(S\NP)" not in parents("NP", ",")
    assert "S[dcl]" in parents("S[dcl]", ".")
    assert "NP[conj]" in parents(",", "NP[conj]")


def test_shared_feature_variable_cannot_bind_to_two_constants():
    assert not parents(r"NP/((S\NP)/(S\NP))", r"(S[dcl]\NP)/(S[b]\NP)")
    assert "NP" in parents(r"NP/((S\NP)/(S\NP))", r"(S[dcl]\NP)/(S[dcl]\NP)")
