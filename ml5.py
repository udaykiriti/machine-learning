from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the Bayesian Network structure
model = BayesianNetwork([('B', 'A'), ('E', 'A'), ('A', 'J'), ('A', 'M')])

# Define the CPD for B
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.001], [0.999]])

# Define the CPD for E
cpd_e = TabularCPD(variable='E', variable_card=2, values=[[0.002], [0.998]])

# Define the CPD for A
cpd_a = TabularCPD(variable='A', variable_card=2,
                   values=[[0.95, 0.94, 0.29, 0.001],
                           [0.05, 0.06, 0.71, 0.999]],
                   evidence=['B', 'E'], evidence_card=[2, 2])

# Define the CPD for J
cpd_j = TabularCPD(variable='J', variable_card=2,
                   values=[[0.9, 0.05],
                           [0.1, 0.95]],
                   evidence=['A'], evidence_card=[2])

# Define the CPD for M
cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[[0.7, 0.01],
                           [0.3, 0.99]],
                   evidence=['A'], evidence_card=[2])

# Add the CPDs to the model
model.add_cpds(cpd_b, cpd_e, cpd_a, cpd_j, cpd_m)

# Check if the model is valid
assert model.check_model()

# Perform inference
inference = VariableElimination(model)

# Example query: What is the probability of J given M=1?
result = inference.query(variables=['J'], evidence={'M': 1})
print(result)
