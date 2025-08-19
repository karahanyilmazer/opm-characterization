import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV
df = pd.read_csv("daq_log.csv")

print(df.head())  # peek at the data

# Plot each channel vs time
plt.figure(figsize=(10, 6))
for col in df.columns[1:]:  # skip the timestamp column
    plt.plot(df["Timestamp (s)"], df[col], label=col)

plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("DAQ Logged Signals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
