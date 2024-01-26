from bold_dementia.data.memento import Memento

def test_loading_augmented():
    base_phenotype = Memento.load_phenotypes("/bigdata/jlegrand/data/Memento/phenotypes.tsv")
    aumgented_phenotype = Memento.load_phenotypes(
        "/bigdata/jlegrand/data/Memento/output/augmented_phenotypes.csv",
        augmented=True
    )
    assert all(base_phenotype.INCCONSDAT_D == aumgented_phenotype.INCCONSDAT_D)