from bold_dementia.data.memento import Memento

def test_loading_augmented():
    base_phenotype = Memento.load_phenotypes("/bigdata/jlegrand/data/Memento/phenotypes.tsv", augmented=False)
    aumgented_phenotype = Memento.load_phenotypes(
        "/bigdata/jlegrand/data/Memento/output/augmented_phenotypes.csv"
    )
    common_cols = list(set(base_phenotype.columns).intersection(set(aumgented_phenotype.columns)))
    assert all(base_phenotype[common_cols] == aumgented_phenotype[common_cols])