def combine_intervals_and_overwrite(file_path):
    # Read the intervals from the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    intervals = [line.strip().split() for line in lines]
    combined_intervals = []

    start = intervals[0][0]
    end = intervals[0][1]

    for i in range(1, len(intervals)):
        current_start, current_end = intervals[i]

        if current_start == end:  # Check if intervals are continuous
            end = current_end
        else:
            combined_intervals.append((start, end))
            start = current_start
            end = current_end

    # Append the last interval
    combined_intervals.append((start, end))

    # Overwrite the file with combined intervals
    with open(file_path, 'w') as file:
        for interval in combined_intervals:
            file.write(f"{interval[0]} {interval[1]}\n")


# Apply the function to files from file_01.txt to file_64.txt
for i in range(1, 65):
    file_path = f"file_{i:02}.txt"  # Generates file names like file_01.txt, file_02.txt, etc.
    try:
        combine_intervals_and_overwrite(file_path)
        print(f"File {file_path} has been updated with combined intervals.")
    except FileNotFoundError:
        print(f"File {file_path} not found. Skipping.")
