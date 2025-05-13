# Open file_num.txt to read the current value of n
# This file keeps track of which line from eminem.txt should be processed next
with open("file_num.txt", "r") as file_num:
    n = int(file_num.read().strip())  # Read the value of n and convert it to an integer

# Open eminem.txt to read all lines
# This file contains multiple lines, and we will process one line at a time based on n
with open("combined/eminem.txt", "r") as eminem_file:
    lines = eminem_file.readlines()  # Read all lines from the file into a list

# Ensure that the value of n is within the valid range of line numbers
if 1 <= n <= len(lines):
    # Get the nth line from the file
    # Python uses 0-based indexing, so we need to access lines[n-1]
    nth_line = lines[n - 1]

    # Write the nth line to time.txt
    # This file is updated with the specific line for further processing
    with open("time.txt", "w") as time_file:
        time_file.write(nth_line)

    # Increment n by 1 to process the next line in the next run
    n += 1

    # Write the updated value of n back to file_num.txt
    # This ensures the next iteration will process the next line in eminem.txt
    with open("file_num.txt", "w") as file_num:
        file_num.write(str(n))
else:
    # If n is out of the valid range, print an error message
    # This happens if n is less than 1 or greater than the number of lines in eminem.txt
    print("Error: n is out of bounds of the lines in eminem.txt")
