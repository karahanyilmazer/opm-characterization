import matplotlib.pyplot as plt
import pyxdf

# Load the XDF file
filename = "your_recording.xdf"
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

print(f"Stream '{stream['info']['name'][0]}' has shape {data.shape}")

# Plot each channel
plt.figure(figsize=(10, 6))
for ch in range(data.shape[1]):
    plt.plot(timestamps, data[:, ch], label=f"Ch{ch+1}")

plt.xlabel("Time (s)")
plt.ylabel("Signal")
plt.title(f"Stream: {stream['info']['name'][0]}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
