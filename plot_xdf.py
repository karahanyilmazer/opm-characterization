# %%
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pyxdf

plt.rcParams.update(
    {
        "font.size": 14,  # Base font size
        "axes.titlesize": 16,  # Title font size
        "axes.labelsize": 14,  # Axis label font size
        "xtick.labelsize": 12,  # X-axis tick label size
        "ytick.labelsize": 12,  # Y-axis tick label size
    }
)


# %%
def get_wall_clock_time(timestamps):
    # Check if input is a single timestamp (int or float)
    if isinstance(timestamps, (int, float)):
        return RECORDING_START + datetime.timedelta(
            seconds=timestamps - FIRST_TIMESTAMP
        )
    else:
        # Assume iterable of timestamps
        clock_times = [
            RECORDING_START + datetime.timedelta(seconds=t - FIRST_TIMESTAMP)
            for t in timestamps
        ]
        return clock_times

    return clock_times


def get_time_ticks(start_time, end_time, interval_minutes=60):
    """
    Generate time ticks at specified intervals.

    Parameters:
    - start_time: datetime object for the start time
    - end_time: datetime object for the end time
    - interval_minutes: interval in minutes (60=hourly, 30=half-hourly, 15=quarter-hourly)

    Returns:
    - List of datetime objects at the specified intervals
    """
    ticks = []

    # Round start_time to the nearest interval
    if interval_minutes == 60:
        current = start_time.replace(minute=0, second=0, microsecond=0)
    elif interval_minutes == 30:
        minute = 0 if start_time.minute < 30 else 30
        current = start_time.replace(minute=minute, second=0, microsecond=0)
    elif interval_minutes == 15:
        minute = (start_time.minute // 15) * 15
        current = start_time.replace(minute=minute, second=0, microsecond=0)
    else:
        # For other intervals, just use the start time
        current = start_time

    # Generate ticks
    while current <= end_time:
        if current >= start_time:
            ticks.append(current)
        current += datetime.timedelta(minutes=interval_minutes)

    return ticks


# %%
# Load the XDF file
filename = "sub-P001_ses-S001_task-opm-daq_run-001_analog.xdf"
streams, fileheader = pyxdf.load_xdf(filename)

print(f"Loaded {len(streams)} streams from {filename}")

# Inspect available streams
for i, stream in enumerate(streams):
    name = stream["info"]["name"][0]
    type_ = stream["info"]["type"][0]
    rate = stream["info"]["nominal_srate"][0]
    print(f"[{i}] Name: {name}, Type: {type_}, Rate: {rate} Hz")

# Pick the first stream (adjust index if needed)
stream = streams[0]

# Extract timestamps
timestamps = stream["time_stamps"]
RECORDING_START = datetime.datetime.fromisoformat(fileheader["info"]["datetime"][0])
FIRST_TIMESTAMP = timestamps[0]
clock_times = get_wall_clock_time(timestamps)

print("First sample:", get_wall_clock_time(timestamps[0]))
print("Last sample :", get_wall_clock_time(timestamps[-1]))

# Extract data
data = stream["time_series"]
data = data.T  # Transpose to have (channels x timestamps)
n_chs, n_samples = data.shape

print(f"Stream '{stream['info']['name'][0]}' has shape {data.shape}")
print("Statistics Before Conversion:")
print(f"Min: {data.min(axis=1)}")
print(f"Max: {data.max(axis=1)}")
print(f"Mean: {data.mean(axis=1)}")
print(f"Std: {data.std(axis=1)}")

# %%
# Convert voltage to nanoTesla
# First sensor (channels 0-2): 7 mT/V = 7,000,000 nT/V
# Second sensor (channels 3-5): 10 mT/V = 10,000,000 nT/V
conversion_factors = np.array([[7e6, 7e6, 7e6, 10e6, 10e6, 10e6]]).T  # nT/V

# Apply conversion to each channel
data *= conversion_factors

print("Data converted from V to nT")
print("Statistics After Conversion:")
print(f"Min: {data.min(axis=1)}")
print(f"Max: {data.max(axis=1)}")
print(f"Mean: {data.mean(axis=1)}")
print(f"Std: {data.std(axis=1)}")

# %%
# Plot each channel
samples_to_skip = 0
samples_to_plot = n_samples
x_plot = clock_times[samples_to_skip:samples_to_plot]

# Set time interval for x-axis ticks
# Options: 60 (hourly), 30 (half-hourly), 15 (quarter-hourly)
time_interval_minutes = 60

# Generate custom time ticks
time_ticks = get_time_ticks(x_plot[0], x_plot[-1], time_interval_minutes)

# Assign a color for each row
row_colors = [plt.cm.viridis(i / 3) for i in range(3)]  # 3 rows

# Calculate tick indices using sampling rate (much faster!)
sampling_rate = float(stream["info"]["nominal_srate"][0])
time_tick_indices = []
start_time = x_plot[0]

for tick_time in time_ticks:
    # Calculate seconds from start
    seconds_from_start = (tick_time - start_time).total_seconds()
    # Convert to sample index
    sample_index = int(seconds_from_start * sampling_rate)
    # Make sure index is within bounds
    if 0 <= sample_index < len(x_plot):
        time_tick_indices.append(sample_index)

# %%
# Set font sizes
fig, axs = plt.subplots(3, 2, figsize=(20, 12))

# Channel mapping: first 3 channels on left (Inside), second 3 on right (Outside)
axis_labels = ["X-axis", "Y-axis", "Z-axis"]
column_labels = ["Inside", "Outside"]

for row in range(3):  # 3 rows for x, y, z axes
    for col in range(2):  # 2 columns for Inside/Outside
        ch = row + col * 3  # Channel mapping: 0,1,2 for left col; 3,4,5 for right col

        if ch < n_chs:  # Make sure channel exists
            color = row_colors[row]
            axs[row, col].plot(
                data[ch, samples_to_skip:samples_to_plot],
                color=color,
            )

            # Y-label only for left column
            if col == 0:
                axs[row, col].set_ylabel(f"{axis_labels[row]} Amplitude (nT)")

            # Set custom time ticks
            axs[row, col].set_xticks(time_tick_indices)

            # X-label only for bottom row
            if row == 2:
                axs[row, col].set_xlabel("Time (HH:MM)")
                axs[row, col].set_xticklabels(
                    [dt.strftime("%H:%M") for dt in time_ticks], rotation=45
                )
            else:
                axs[row, col].set_xticklabels([])

            axs[row, col].grid(True)

            # Title: column label for top row, axis label for others
            if row == 0:
                axs[row, col].set_title(f"{column_labels[col]}\n{axis_labels[row]}")
            else:
                axs[row, col].set_title(f"{axis_labels[row]}")

plt.tight_layout()
plt.savefig("opm_characterization_plot.png")
plt.show()

# %%
