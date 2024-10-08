import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('medicine.csv')
print(data.head())


def mol_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [Descriptors.MolWt(mol), Descriptors.MolLog(mol), Descriptors.NumHDonors(mol)]


data['descriptors'] = data['molecule_smiles'].apply(mol_descriptors)
print(data.head())
descriptors_df = pd.DataFrame(data['descriptors'].tolist())
