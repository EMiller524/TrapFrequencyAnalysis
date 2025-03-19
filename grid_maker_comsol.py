# make a function to print a string with numbers seperated by commas with the numbers seperated by x distance and centered at zero


def print_numbers_separated_by_distance(x, n):
    # Generate the numbers centered at zero
    numbers = [i * x for i in range(-n // 2, n // 2 + 1)]

    # round each number to the nearest 8 decimal places
    numbers = [round(i, 8) for i in numbers]

    # Convert numbers to strings and join them with commas
    result = ", ".join(map(str, numbers))

    # Print the result
    print(result)


# Example usage
# print_numbers_separated_by_distance(0.005, 160)
print_numbers_separated_by_distance(0.000125, 640)

# Grid 1
# X --> print_numbers_separated_by_distance(0.01, 80)
# Y --> print_numbers_separated_by_distance(0.0005, 140)
# Z --> print_numbers_separated_by_distance(0.0005, 140)

# Grid 2
# X --> print_numbers_separated_by_distance(0.005, 160)
# Y --> print_numbers_separated_by_distance(0.000125, 640)
# Z --> print_numbers_separated_by_distance(0.000125, 640)


# Z
# -0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1

# X
# -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5

# Y
# -0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1


# Grid 1
