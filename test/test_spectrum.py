import pandas as pd
from annotix_ml.spectrum import Spectrum

def example_data():
    return {
        "spectral_data_id": 1, "pepmass": 146.06, "num_peaks": 19, 
        "peaks_list": "118.0650177 100\\n146.0600128 43.15889778\\n146.0783691 9.715713778\\n91.05410004 6.332097054\\n117.0570679 1.61180439\\n86.09648132 0.348417004\\n84.58882904 0.169255974\\n101.8007584 0.154802373\\n89.72386169 0.137757455\\n87.3788147 0.133791458\\n76.92423248 0.132084184\\n110.8465347 0.131686586\\n59.6097641 0.12867336\\n54.34722519 0.115051555\\n66.14446259 0.114989303\\n63.52222443 0.111693346\\n118.0475464 0.770947713\\n118.0824585 0.755948745\\n146.1166534 0.133191735\\n", 
        "data_id": 1, "smiles": "O=CC1=CNC=2C=CC=CC12", "compound_name": "3-Formylindole (level 2)", "database_name": "Bacterial_metabolites_database", 
        "path_to_data": "/data/frederic/ToBeInserted/Json_DataBase/", "json_file": "Bacterial_metabolites_database_molecule000003_PEPMASS-146.0600.json", 
        "filename": "/data/frederic/ToBeInserted/Json_DataBase/Bacterial_metabolites_database_molecule000003_PEPMASS-146.0600.json", "charge": "1+"}

def test_spectrum():
    
    data = example_data()
    
    # Convert the first row to a Spectrum object
    spectrum = Spectrum(pd.Series(data))
    
    # Unit test for metadata
    assert spectrum.metadata["id"] == 1
    assert spectrum.metadata["pepmass"] == 146.06
    assert spectrum.metadata["smiles"] == "O=CC1=CNC=2C=CC=CC12"
    assert spectrum.metadata["compound_name"] == "3-Formylindole (level 2)"
    assert spectrum.metadata["charge"] == 1 #the charge is converted into int in matchms
        
    # Unit test for mz and intensities
    assert len(spectrum.mz) == 19
    assert len(spectrum.intensities) == 19
    assert min(spectrum.mz) == 54.34722519
    assert max(spectrum.mz) == 146.1166534
    assert min(spectrum.intensities) == 0.111693346
    assert max(spectrum.intensities) == 100

    # Unit test for document representation
    doc = spectrum.document()
    assert isinstance(doc.words, list)
    assert {type(word) for word in doc.words} == {str}
    assert doc.words == ["peak@54.35", "peak@59.61", "peak@63.52", "peak@66.14", "peak@76.92", "peak@84.59", "peak@86.10", "peak@87.38", "peak@89.72", "peak@91.05", "peak@101.80", "peak@110.85", "peak@117.06", "peak@118.05", "peak@118.07", "peak@118.08", "peak@146.06", "peak@146.08", "peak@146.12", "loss@27.98", "loss@27.99", "loss@28.01", "loss@29.00", "loss@35.21", "loss@44.26", "loss@55.01", "loss@56.34", "loss@58.68", "loss@59.96", "loss@61.47", "loss@69.14", "loss@79.92", "loss@82.54", "loss@86.45", "loss@91.71"]
