from datetime import datetime, timedelta

def parse_intervals(file_path):
    """
    Parse intervals from a text file.
    Each line should be in the format: start_time-end_time.
    """
    intervals = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                start, end = line.split('-')
                intervals.append((datetime.fromisoformat(start), datetime.fromisoformat(end)))
    return intervals

def generate_sliding_windows(intervals, window_duration, step_size):
    """
    Generate sliding window intervals within the given intervals.
    
    Parameters:
    - intervals: List of (start, end) tuples defining available intervals.
    - window_duration: Duration of each sliding window (timedelta).
    - step_size: Step size for the sliding window (timedelta).
    
    Returns:
    - List of (start, end) tuples for the sliding windows.
    """
    sliding_windows = []
    for start, end in intervals:
        current_start = start
        while current_start + window_duration <= end:
            current_end = current_start + window_duration
            sliding_windows.append((current_start, current_end))
            current_start += step_size
    return sliding_windows

def format_intervals(intervals):
    """
    Format intervals into a readable string for output.
    """
    return [f"{start.isoformat()}-{end.isoformat()}" for start, end in intervals]

# Read intervals from goes.txt
goes_intervals = parse_intervals('goes.txt')

# Generate sliding windows
window_duration = timedelta(seconds=96)
step_size = timedelta(seconds=8)
sliding_windows = generate_sliding_windows(goes_intervals, window_duration, step_size)

# Save to a text file
output_file = "sliding_windows.txt"
with open(output_file, 'w') as file:
    for interval in format_intervals(sliding_windows):
        file.write(f"{interval}\n")

print(f"Sliding windows have been written to {output_file}.")
