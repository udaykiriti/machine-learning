from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#define the structure of a earthquake
model = BayesianNetwork([('B','A'),('E','A'),('A','J'),('A','M')])
#CPT for Burglary
cpd_b = TabularCPD(variable='B',variable_card=2, values=[[0.001],[0.999]])
cpd_e = TabularCPD(variable='E',variable_card=2, values=[[0.002],[0.998]])

cpd_a = TabularCPD(variable='A',variable_card=2, values=[[0.95,0.94,0.29,0.001],[0.05,0.06,0.71,0.999]], evidence=['B','E'],evidence_card=[2,2])
cpd_j = TabularCPD(variable='J',variable_card=2,values=[[0.9,0.05],[0.1,0.95]],evidence=['A'],evidence_card=[2])

#marrycalls
cpd_m = TabularCPD(variable='M',variable_card=2,values=[[0.7,0.01],[0.3,0.99]],evidence=['A'],evidence_card=[2])
#Adding the CPD's to the model
model.add_cpds(cpd_b,cpd_m,cpd_j,cpd_a,cpd_e)
print('probability Distribution,P(Burglary)')
print(cpd_b)
print()
print('Probability Distribution,P(Earthquake)')
print(cpd_e)
print()
print('Joint Probability Distribution,P(Alarm | Burglary, Earthquake)')
print(cpd_a)
print()
print(' Joint Probability distribution, P(MaryCells | Alarm) ')
print(cpd_m)
print()

#check if the model is valid
assert model.check_model()
#Performing inference using Variable Elimination
inference = VariableElimination(model)
#Query the probability of burglary gain that both mary and john have called
result = inference.query(variables=['A'],evidence={'M':1,'J':1,'B':0,'E':0})
print(result)
result2 = inference.query(variables=['B'],evidence={'E':1,'A':1,'M':0,'J':0})
print(result2,'The evidence out')