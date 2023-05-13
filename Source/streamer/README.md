# X-trodes Bluetooth Real-time Streaming API

Python interface for accessing data streamed by an X-trodes data acquisition unit to Windows via the "Bluetooth
Low Energy C# sample" Windows application intermediary.

Also included is a rudimentary real-time data visualizer to demonstrate how one might use streamed data in real-time.

**_NOTE: IMU data streaming still in development. Currently only EXG data available._**

**_Prerequisites:_**
1. Correctly installed "Bluetooth Low Energy C# sample" Windows application
2. Install all Python packages in requirements.txt

**_Usage:_**

1. Install XtrRT: `pip install git+https://github.com/xtrodesorg/XtrRT.git`
2. Run "Bluetooth Low Energy C# sample" Windows application; connect to DAU via application.
3. Run PowerShell as Administrator. Copy the following and paste it in PowerShell by right-clicking inside the PowerShell window, then press enter:
`CheckNetIsolation.exe LoopbackExempt -is -p=S-1-15-2-2022722280-4131399851-3337013219-4054732753-2439233258-3605005605-669734301`
4. Access streamed data via XtrRT like so:

```python
from XtrRT.data import Data
from XtrRT.viz import Viz

host_name = "127.0.0.1"  # IP address from which to receive data
port = 20001             # local port through which to access host
timeout = None           # if streaming is interrupted for this many seconds or longer, terminate program
verbose = True           # if to print to console BT packet summary (as sanity check) upon each received packet

# Stream data in side thread
data = Data(host_name, port, verbose=verbose, timeout_secs=timeout)
data.start()

# Visualize data stream in main thread:
secs = 10             # Time window of plots (in seconds)
ylim = (-1000, 1000)  # y-limits of plots
ica = False           # Perform and visualize ICA alongside raw data
update_interval = 10  # Update plots every X ms
max_points = 250      # Maximum number of data points to visualize per channel (render speed vs. resolution)
viz = Viz(data, window_secs=secs, plot_exg=True, plot_imu=False, plot_ica=ica,
          update_interval_ms=update_interval, ylim_exg=ylim, max_points=250)
viz.start()

print('Process terminated')
```