# %%
import matplotlib.pyplot as plt
import pyxdf

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

# Extract data and timestamps
timestamps = stream["time_stamps"]
data = stream["time_series"]
data = data.T  # Transpose to have (channels x timestamps)
n_chs, n_samples = data.shape

print(f"Stream '{stream['info']['name'][0]}' has shape {data.shape}")

# %%
# Plot each channel
fig, axs = plt.subplots(3, 2)
axs = axs.flatten()
samples_to_plot = 5000

# Assign a color for each row
row_colors = [plt.cm.viridis(i / 3) for i in range(3)]  # 3 rows

for i, ch in enumerate(range(n_chs)):
    row = i // 2  # 0 for top, 1 for middle, 2 for bottom
    col = i % 2  # 0 for left, 1 for right
    color = row_colors[row]
    axs[i].plot(
        timestamps[:samples_to_plot],
        data[ch, :samples_to_plot],
        color=color,
    )

    # Y-label only for left column
    if col == 0:
        axs[i].set_ylabel("Amplitude (V)")
    else:
        axs[i].set_yticklabels([])

    # X-label only for bottom row
    if row == 2:
        axs[i].set_xlabel("Time (s)")
    else:
        axs[i].set_xticklabels([])
    axs[i].grid(True)
    axs[i].set_title(f"Ch{ch+1}")

plt.tight_layout()
plt.show()

# %%
