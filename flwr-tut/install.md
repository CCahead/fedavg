First match Pytorch and cuda version. In my case, I firstly installed cuda 12.6, but the pytorch 
only supports cuda 11.8,12.1,12.4! 
```url
https://pytorch.org/ 
```
Which cause incompacity. I uninstall cuda in setting/app&features(nvidia cuda) 
```url
https://www.deploymastery.com/2023/10/16/how-to-uninstall-cuda-windows-mac-linux/#:~:text=Uninstall%20CUDA%20on%20Windows%201%20Search%20for%20%E2%80%9CControl,5%20Follow%20the%20instructions%20to%20complete%20the%20uninstallation.
```

and reinstall the recommaned cuda version: 
```url
https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_network
```

however: using this given command, I install a cpu supported torch??
 ```python
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
print(torch.__version__)=> 2.4.1+cpu
```
# SOLUTION
Use CONDA INSTALL!!! instead of PIP!

## SETUP
```bash
pip install  flwr-datasets[vision] 
conda install flwr torch torch-version matplotlib
```
In my case, I ran flwr[simulation] for tut.
https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html#Install-dependencies


---










```log
INFO 2024-09-24 17:36:40,412:      [SUMMARY]
INFO 2024-09-24 17:36:40,413:      Run finished 5 round(s) in 617.64s
INFO 2024-09-24 17:36:40,415:      	History (loss, distributed):
INFO 2024-09-24 17:36:40,416:      		round 1: 0.0640372561454773
INFO 2024-09-24 17:36:40,416:      		round 2: 0.0554464474439621
INFO 2024-09-24 17:36:40,417:      		round 3: 0.05214504785537719
INFO 2024-09-24 17:36:40,419:      		round 4: 0.05038692493438721
INFO 2024-09-24 17:36:40,419:      		round 5: 0.04793982825279236
INFO 2024-09-24 17:36:40,421:      	History (metrics, distributed, evaluate):
INFO 2024-09-24 17:36:40,422:      	{'accuracy': [(1, 0.2964),
INFO 2024-09-24 17:36:40,424:      	              (2, 0.3638),
INFO 2024-09-24 17:36:40,424:      	              (3, 0.40780000000000005),
INFO 2024-09-24 17:36:40,426:      	              (4, 0.42679999999999996),
INFO 2024-09-24 17:36:40,426:      	              (5, 0.45899999999999996)]}
```











there's a incompatible condition between nvcc --version and nvidia-smi
```url
https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi
```
nvcc version:
```nvcc
C:\Users\junjun>nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
```



nvidia-smi shows:
```nivida-smi
C:\Users\junjun>nvidia-smi
Tue Sep 24 06:54:45 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060      WDDM  |   00000000:03:00.0  On |                  N/A |
|  0%   51C    P0             N/A /  115W |    2420MiB /   8188MiB |      2%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2460    C+G   ...crosoft\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A      2508    C+G   ...siveControlPanel\SystemSettings.exe      N/A      |
|    0   N/A  N/A      4928    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A      7580    C+G   ...5n1h2txyewy\ShellExperienceHost.exe      N/A      |
|    0   N/A  N/A      8352    C+G   C:\Windows\explorer.exe                     N/A      |
|    0   N/A  N/A      8524    C+G   ...t.LockApp_cw5n1h2txyewy\LockApp.exe      N/A      |
|    0   N/A  N/A      8836    C+G   ...cal\Microsoft\OneDrive\OneDrive.exe      N/A      |
|    0   N/A  N/A     10264    C+G   ...2txyewy\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A     10364    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A     10796    C+G   ....Search_cw5n1h2txyewy\SearchApp.exe      N/A      |
|    0   N/A  N/A     11468    C+G   ....Search_cw5n1h2txyewy\SearchApp.exe      N/A      |
|    0   N/A  N/A     12092    C+G   ...on\128.0.2739.79\msedgewebview2.exe      N/A      |
|    0   N/A  N/A     17720    C+G   ...\cef\cef.win7x64\steamwebhelper.exe      N/A      |
|    0   N/A  N/A     18180    C+G   ...\cef\cef.win7x64\steamwebhelper.exe      N/A      |
|    0   N/A  N/A     32792    C+G   ...\Docker\frontend\Docker Desktop.exe      N/A      |
|    0   N/A  N/A     35172    C+G   ...s\vscode\Microsoft VS Code\Code.exe      N/A      |
|    0   N/A  N/A     39384    C+G   ....0_x64__8wekyb3d8bbwe\HxOutlook.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```