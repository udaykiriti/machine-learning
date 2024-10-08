import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
path = r"medicine.csv"
data = pd.read_csv(path)
print("Initial Data Preview:")
print(data.head())

# Convert SMILES to molecular descriptors
def mol_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return [Descriptors.HeavyAtomMolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol)]
    else:
        return [0, 0, 0]  # Fallback values for invalid SMILES

# Apply the descriptor function to the SMILES columns
data['block1_descriptors'] = data['buildingblock1_smiles'].apply(mol_descriptors)
data['block2_descriptors'] = data['buildingblock2_smiles'].apply(mol_descriptors)
data['block3_descriptors'] = data['buildingblock3_smiles'].apply(mol_descriptors)

# Split descriptors into separate columns for each block
block1_df = pd.DataFrame(data['block1_descriptors'].tolist(), columns=['Block1_HeavyAtomMolWt', 'Block1_MolLogP', 'Block1_NumHDonors'])
block2_df = pd.DataFrame(data['block2_descriptors'].tolist(), columns=['Block2_HeavyAtomMolWt', 'Block2_MolLogP', 'Block2_NumHDonors'])
block3_df = pd.DataFrame(data['block3_descriptors'].tolist(), columns=['Block3_HeavyAtomMolWt', 'Block3_MolLogP', 'Block3_NumHDonors'])

# Merge descriptors with the original DataFrame
data = pd.concat([data.drop(columns=['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'block1_descriptors', 'block2_descriptors', 'block3_descriptors', 'molecule_smiles']),
                  block1_df, block2_df, block3_df], axis=1)

print("Data after merging descriptors:")
print(data.head())

# Check available columns after preprocessing
print("Available columns after preprocessing:")
print(data.columns)

# Set your target variable
target_column = 'protein_name'

# Split data into features and labels
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = svm_model.predict(X_test)

# Calculate accuracy and print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print("Support Vector Classification Accuracy:", accuracy)
print("Support Vector Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))