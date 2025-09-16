def create_files(a, b):
    for i in range(a, b + 1):
        filename = f"reformulateText{i}.txt"
        with open(filename, "w") as f:
            f.write(f"This is file {filename}\n")
    print(f"Created files from reformulateText{a}.txt to reformulateText{b}.txt")

# Example usage:
create_files(10, 20)
