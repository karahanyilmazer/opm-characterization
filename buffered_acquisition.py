import csv
import signal
import sys
import time

import nidaqmx
from nidaqmx.constants import AcquisitionType

running = True


def signal_handler(sig, frame):
    global running
    print("\nStopping acquisition...")
    running = False


signal.signal(signal.SIGINT, signal_handler)

# Choose your channels here
channels = ["cDAQ2Mod1/ai0", "cDAQ2Mod1/ai1", "cDAQ2Mod1/ai3"]

# Acquisition settings
sample_rate = 1000  # Hz
samples_per_read = 100  # number of samples to fetch per loop (0.1 s worth)

with nidaqmx.Task() as task, open("daq_log.csv", "w", newline="") as f:
    # Add channels
    for ch in channels:
        task.ai_channels.add_ai_voltage_chan(ch, min_val=-10.0, max_val=10.0)

    # Configure timing: 1 kHz, continuous
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate, sample_mode=AcquisitionType.CONTINUOUS
    )

    writer = csv.writer(f)
    writer.writerow(["Timestamp (s)"] + channels)  # header

    print("Starting acquisition at 1 kHz. Press Ctrl+C to stop.")
    start_time = time.time()
    total_samples = 0

    while running:
        # Read a block of samples (shape: channels × samples)
        data = task.read(number_of_samples_per_channel=samples_per_read)

        # task.read() returns nested lists: one list per channel
        # Transpose to samples × channels
        data = list(zip(*data))

        for i, sample in enumerate(data):
            # Compute timestamp from sample index
            timestamp = (total_samples + i) / sample_rate
            writer.writerow([timestamp] + list(sample))

        total_samples += len(data)

    print(f"Stopped. Logged {total_samples} samples per channel.")
