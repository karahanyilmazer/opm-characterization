import csv
import signal
import sys
import time

import nidaqmx
from nidaqmx.constants import AcquisitionType

# Graceful shutdown flag
running = True


def signal_handler(sig, frame):
    global running
    print("\nStopping acquisition...")
    running = False


signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C

# CSV file setup
csv_file = open("daq_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp (s)", "Voltage (V)"])  # Header row

# Configure DAQ task
with nidaqmx.Task() as task:
    # Replace cDAQ1Mod1 with your actual device/module name from NI MAX
    task.ai_channels.add_ai_voltage_chan("cDAQ1Mod1/ai0", min_val=-10.0, max_val=10.0)

    # Set timing: 1000 samples per second, continuous
    task.timing.cfg_samp_clk_timing(rate=1000, sample_mode=AcquisitionType.CONTINUOUS)

    print("Starting acquisition. Press Ctrl+C to stop.")

    start_time = time.time()

    while running:
        # Read a block of samples (adjust number_of_samples as needed)
        data = task.read(number_of_samples_per_channel=100)
        timestamp = time.time() - start_time

        # Write each sample with the same timestamp (approximate)
        for sample in data:
            csv_writer.writerow([timestamp, sample])

        # Small sleep to avoid busy-wait
        time.sleep(0.01)

csv_file.close()
print("Data saved to daq_log.csv")
