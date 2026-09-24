from importlib import util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "fetch_pmc.py"
spec = util.spec_from_file_location("fetch_pmc", MODULE_PATH)
assert spec and spec.loader
fetch_pmc = util.module_from_spec(spec)
spec.loader.exec_module(fetch_pmc)


def test_license_normalisation_is_conservative():
    assert fetch_pmc._normalise_license(["https://creativecommons.org/licenses/by/4.0/"]) == "CC-BY"
    assert fetch_pmc._normalise_license(["https://creativecommons.org/licenses/by-sa/4.0/"]) == "CC-BY-SA"
    assert fetch_pmc._normalise_license(["https://creativecommons.org/publicdomain/zero/1.0/"]) == "CC0"
    assert fetch_pmc._normalise_license(["https://creativecommons.org/licenses/by-nc/4.0/"]) is None
