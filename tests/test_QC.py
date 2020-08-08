import pytest
import mrqy.QC as QC
import os 
# tests that the sample output matches expected baseline output
class TestQC:
    def test_parse_patient_names(self):
        # execute test subject
        patients, names, dicom_spil, nondicom_spli, nondicom_names = QC.parse_patient_names(os.getcwd() + "/tests/TCIA/")
        print(patients)
        # assert number of images
        assert len(patients) == 11
        # assert the snapshot of 1 image
        # assert that the tsv was generated for UI
        
    # tests that the main exits correctly if no input is provided
    def test_QC_main_missing_input(self):
        # assert that bad folder input fails
        assert 1