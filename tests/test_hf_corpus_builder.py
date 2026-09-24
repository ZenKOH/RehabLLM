from importlib import util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_hf_rehab_corpus.py"
spec = util.spec_from_file_location("build_hf_rehab_corpus", MODULE_PATH)
assert spec and spec.loader
builder = util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_classifies_robotic_rehabilitation():
    text = "Robot-assisted therapy is used during stroke rehabilitation and gait training."
    assert builder.classify_document(text) == "robotics"


def test_classifies_neurotechnology_rehabilitation():
    text = "Functional electrical stimulation was studied during neurorehabilitation after stroke."
    assert builder.classify_document(text) == "neurotechnology"


def test_rejects_non_rehabilitation_text():
    assert builder.classify_document("A molecular study of bacterial metabolism.") is None


def test_license_normalisation_is_conservative():
    assert builder.normalise_license(
        {"license": "https://creativecommons.org/licenses/by/4.0/"}
    ) == "CC-BY"
    assert builder.normalise_license(
        {"license": "https://creativecommons.org/licenses/by-sa/4.0/"}
    ) == "CC-BY-SA"
    assert builder.normalise_license(
        {"license": "https://creativecommons.org/publicdomain/zero/1.0/"}
    ) == "CC0"
    assert (
        builder.normalise_license(
            {"license": "https://creativecommons.org/licenses/by-nc/4.0/"}
        )
        is None
    )


def test_quota_total_matches_target():
    targets = builder.targets_from_total(12000)
    assert sum(targets.values()) == 12000
    assert targets["robotics"] > targets["neurotechnology"]
