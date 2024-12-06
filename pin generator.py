# Generate all possible 4-digit PINs
possible_pins = [f"{i:04d}" for i in range(10000)]

# Print all possible PINs
for pin in possible_pins:
    print(pin)
