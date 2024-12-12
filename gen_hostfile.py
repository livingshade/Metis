import sys

def create_hostfile(n):
    try:
        # Calculate repetitions for each host
        repetitions = n // 4
        if repetitions < 1:
            print("Error: n must be at least 4.")
            return
        
        # Create and write to the hostfile
        with open("hostfile", "w") as f:
            for _ in range(repetitions):
                f.write("0.0.0.1 slots=4\n")
            for _ in range(repetitions):
                f.write("0.0.0.4 slots=4\n")
        print("hostfile created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure a valid number argument is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <number>")
    else:
        try:
            n = int(sys.argv[1])
            create_hostfile(n)
        except ValueError:
            print("Error: The argument must be an integer.")