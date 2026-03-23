# System Profile: lunarlake-ultra7-258v-32gb

This is a sanitized system profile intended to be safe to check into git for benchmark context and reproducibility.

- captured_utc: 20260323T105550Z
- os: Arch Linux
- kernel: Linux 7.0.0-rc3-1-mainline-dirty x86_64

## CPU and memory

~~~text
cpu_model: Intel(R) Core(TM) Ultra 7 258V
cpu_count: 8
memory_total: 30Gi
swap_total: 63Gi
~~~

## Intel accelerator inventory

~~~text
00:02.0 VGA compatible controller [0300]: Intel Corporation Lunar Lake [Intel Arc Graphics 130V / 140V] [8086:64a0] (rev 04)
	DeviceName: Onboard - Video
	Subsystem: Micro-Star International Co., Ltd. [MSI] Device [1462:145f]
	Kernel driver in use: xe
	Kernel modules: xe
~~~

## NPU inventory

~~~text
00:0b.0 Processing accelerators [1200]: Intel Corporation Lunar Lake NPU [8086:643e] (rev 04)
	DeviceName: Onboard - Other
	Subsystem: Micro-Star International Co., Ltd. [MSI] Device [1462:1464]
	Kernel driver in use: intel_vpu
	Kernel modules: intel_vpu
~~~

## DRI nodes

~~~text
total 0
drwxr-xr-x  2 root root         80 Mar 22 02:10 by-path
crw-rw----+ 1 root video  226,   0 Mar 23 12:37 card0
crw-rw-rw-  1 root render 226, 128 Mar 22 02:10 renderD128
~~~

## Tool availability

~~~text
xpu-smi=present
sycl-ls=not-found
clinfo=present
vulkaninfo=present
cmake=present
gcc=present
clang=present
icx=not-found
icpx=not-found
python3=present
~~~

## xpu-smi discovery

~~~text
No device discovered
~~~

## OpenCL summary

~~~text
  Platform Name                                   Intel(R) OpenCL Graphics
  Platform Vendor                                 Intel(R) Corporation
  Platform Version                                OpenCL 3.0 
  Device Name                                     Intel(R) Arc(TM) Graphics
  Device Vendor                                   Intel(R) Corporation
  Device Version                                  OpenCL 3.0 NEO 
  Driver Version                                  26.05.37020
~~~

## Vulkan summary

~~~text
GPU0:
	vendorID           = 0x8086
	deviceID           = 0x64a0
	deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName         = Intel(R) Graphics (LNL)
	driverName         = Intel open-source Mesa driver
	driverInfo         = Mesa 26.1.0-devel (git-1e1d8931c7)
GPU1:
	vendorID           = 0x10005
	deviceID           = 0x0000
	deviceType         = PHYSICAL_DEVICE_TYPE_CPU
	deviceName         = llvmpipe (LLVM 21.1.8, 256 bits)
	driverName         = llvmpipe
	driverInfo         = Mesa 26.1.0-devel (git-1e1d8931c7) (LLVM 21.1.8)
~~~

## Intel env var names

~~~text
none
~~~

## Raw capture

The full raw capture for this run was written to:

~~~text
00-setup/results/system-info-lunarlake-ultra7-258v-32gb-20260323T105550Z.txt
~~~
