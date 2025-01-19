## INSTRUCTIONS
### run the command below in a linux terminal
``` sh
bokeh serve imu_bokeh_stream
```

### if running in WSL 

To get an OAK running on WSL 2, you first need to attach USB device to WSL 2. 

On the windows side install the following   
[usbipd-win 3.2.0 Installer](https://github.com/dorssel/usbipd-win/releases/download/v3.2.0/usbipd-win_3.2.0.msi)

Inside WSL 2 you also need to run 
``` sh
sudo apt install linux-tools-virtual hwdata
sudo update-alternatives --install /usr/local/bin/usbip usbip `ls /usr/lib/linux-tools/*/usbip | tail -n1` 20
```

To attach the OAK camera to WSL 2, run the following code in <mark>Python</mark> from an admin Powershell 
``` python
import time
import os

while True:
    output = os.popen('usbipd wsl list').read()
    rows = output.split('\n')
    for row in rows:
        if ('Movidius MyriadX' in row or 'Luxonis Device' in row) and 'Not attached' in row:
            busid = row.split(' ')[0]
            out = os.popen(f'usbipd wsl attach --busid {busid}').read()
            print(out)
            print(f'Usbipd attached Myriad X on bus {busid}')
    time.sleep(.5)

```
The window slider changes the width of the scrolling window ie no of points in view
The delay window can be used to change how many points are plotted per second. This app can comfortably plot at 120Hz

## Example of the running program
https://github.com/hidara2000/depthai/assets/15170494/b15a974c-2955-4f1d-bc45-a0d702d04b09


