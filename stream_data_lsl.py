import signal
import time

import nidaqmx
from nidaqmx.constants import AcquisitionType
from pylsl import StreamInfo, StreamOutlet

running = True


def signal_handler(sig, frame):
    global running
    print("\nStopping acquisition...")
    running = False


signal.signal(signal.SIGINT, signal_handler)

# Configure your channels
channels = ["cDAQ2Mod1/ai0", "cDAQ2Mod1/ai1", "cDAQ2Mod1/ai3"]

# Acquisition parameters
sample_rate = 1000  # Hz
samples_per_read = 100  # pull 0.1 s of data at a time

# Define LSL stream
info = StreamInfo(
    name="NI9202Stream",
    type="Analog",  # or "Analog" or custom
    channel_count=len(channels),
    nominal_srate=sample_rate,
    channel_format="float32",
    source_id="cDAQ2Mod1",
)

outlet = StreamOutlet(info)

with nidaqmx.Task() as task:
    # Add channels
    for ch in channels:
        task.ai_channels.add_ai_voltage_chan(ch, min_val=-10.0, max_val=10.0)

    # Configure timing: continuous 1 kHz
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate, sample_mode=AcquisitionType.CONTINUOUS
    )

    print(
        f"Streaming {len(channels)} channels over LSL at {sample_rate} Hz. Press Ctrl+C to stop."
    )

    while running:
        # Read block of samples
        data = task.read(number_of_samples_per_channel=samples_per_read)

        # Transpose: samples × channels
        data = list(zip(*data))

        # Push each sample to LSL
        for sample in data:
            outlet.push_sample(sample)
