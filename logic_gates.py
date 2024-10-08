import numpy as np

def step_activation(x):
    return 1 if x >= 0 else 0

def mcculloch_pitts(weights, bias, inputs):
    total_input = np.dot(weights, inputs) + bias
    return step_activation(total_input)

inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])


weights_and = np.array([1, 1, 1])
bias_and = -3  # threshold = 3
outputs_and = [mcculloch_pitts(weights_and, bias_and, x) for x in inputs]

weights_or = np.array([1, 1, 1])
bias_or = -1  # threshold = 1
outputs_or = [mcculloch_pitts(weights_or, bias_or, x) for x in inputs]

weights_nor = np.array([-1, -1, -1])
bias_nor = 0  # threshold = 0
outputs_nor = [mcculloch_pitts(weights_nor, bias_nor, x) for x in inputs]

weights_nand = np.array([-1, -1, -1])
bias_nand = 2  # threshold = 2
outputs_nand = [mcculloch_pitts(weights_nand, bias_nand, x) for x in inputs]

print("3-input AND Gate")
for i, x in enumerate(inputs):
    print(f"Input: {x} => Output: {outputs_and[i]}")

print("\n3-input OR Gate")
for i, x in enumerate(inputs):
    print(f"Input: {x} => Output: {outputs_or[i]}")

print("\n3-input NOR Gate")
for i, x in enumerate(inputs):
    print(f"Input: {x} => Output: {outputs_nor[i]}")

print("\n3-input NAND Gate")
for i, x in enumerate(inputs):
    print(f"Input: {x} => Output: {outputs_nand[i]}")
