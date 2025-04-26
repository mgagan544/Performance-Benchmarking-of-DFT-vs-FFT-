# Performance Benchmarking of FFT and DFT on CPU and GPU using Matlab

This project benchmarks and analyzes the performance of **Fast Fourier Transform (FFT)** computations on both **CPU** and **GPU** using an NVIDIA GeForce RTX 4060 Laptop GPU.  
The benchmark focuses on comparing execution time, memory consumption, energy efficiency, and numerical accuracy across a range of input sizes.
It uses Nvidia SMI for gathering data about the **GPU** like memory usage , energy consumption,etc.
---

##  Project Overview

- **Goal**: Compare CPU and GPU performance for 1D FFT computations.
- **FFT Sizes Tested**: From **256** to **8,388,608** points.
- **Metrics Captured**:
  - Execution Time
  - Memory Usage
  - Power Consumption
  - Maximum Numerical Error between CPU and GPU results
- **Environment**:
  - **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (7.44 GB VRAM)
  - **System Memory**: 16 GB
  - **Software**: MATLAB (with Parallel Computing Toolbox)

---

##  How to Run the Benchmark

### Prerequisites

- MATLAB ( 2022b or latest)
- Parallel Computing Toolbox-NVIDIA SMI addon to be installed from MATLAB
- NVIDIA GPU with CUDA support
- `fft_benchmark_20250426_135246.mat` (Included benchmark result file)

### Running the Benchmark

1. Clone or download the repository:
   ```bash
   git clone https://github.com/yourusername/fft-benchmark-rtx4060.git
   cd fft-benchmark-rtx4060
   ```

2. Open MATLAB and navigate to the project directory.

3. To re-run the benchmark, execute:
   ```matlab
   benchmark_script.m
   ```

4. To load and visualize saved results:
   ```matlab
   load('fft_benchmark_20250426_135246.mat')
   % Use your own visualization code or the provided plot scripts
   ```

>  Monitor GPU power usage during execution using `nvidia-smi` in a separate terminal.

---

##  Key Observations

- **CPU vs GPU Speed**:
  - CPU is faster for small FFT sizes (N ≤ 16384).
  - GPU becomes faster starting from N = 65536.
  - GPU achieves up to **2.73× speedup** over CPU at N = 8388608.

- **Memory Usage**:
  - GPU memory consumption remains relatively stable.
  - CPU memory usage increases linearly with FFT size.

- **Power Consumption**:
  - GPU power draw remains moderate even at large FFT sizes.
  - CPU-only computation results in higher total system power at larger N.

- **Numerical Accuracy**:
  - Maximum difference between CPU and GPU FFT outputs is ~9.4e-10.
  - Confirms high reliability of GPU-based computations.

---

##  Benchmark Results Obtained

| N        | DFT Time (s) | FFT CPU Time (s) | FFT GPU Time (s) | DFT Memory (MB) | FFT CPU Memory (MB) | FFT GPU Memory (MB) |
|----------|--------------|--------------|--------------|-----------------|-----------------|-----------------|
| 256      | 0.0005        | 0.0077        | 0.0278        | 1.48             | 6.51             | 10.48           |
| 512      | 0.0002        | 0.0001        | 0.0060        | 4.20             | 0.14             | 10.48           |
| 1024     | 0.0005        | 0.0001        | 0.0085        | 16.78            | 0.00             | 10.48           |
| 2048     | 0.0017        | 0.0002        | 0.0081        | 67.10            | 0.00             | 10.47           |
| 4096     | 0.0048        | 0.0006        | 0.0154        | 263.47           | 0.00             | 10.45           |
| 8192     | NaN           | 0.0002        | 0.0211        | NaN              | 0.00             | 10.42           |
| 16384    | NaN           | 0.0003        | 0.0127        | NaN              | 0.00             | 10.35           |
| 32768    | NaN           | 0.0003        | 0.0024        | NaN              | 0.00             | 10.22           |
| 65536    | NaN           | 0.0035        | 0.0026        | NaN              | -2.67            | 12.06           |
| 131072   | NaN           | 0.0020        | 0.0027        | NaN              | 2.10             | 13.63           |
| 262144   | NaN           | 0.0024        | 0.0030        | NaN              | 5.02             | 18.87           |
| 524288   | NaN           | 0.0030        | 0.0037        | NaN              | 10.06            | 25.17           |
| 1048576  | NaN           | 0.0050        | 0.0059        | NaN              | 19.40            | 37.75           |
| 2097152  | NaN           | 0.0181        | 0.0090        | NaN              | 35.45            | 62.91           |
| 4194304  | NaN           | 0.0402        | 0.0161        | NaN              | 70.99            | 113.25          |
| 8388608  | NaN           | 0.0802        | 0.0294        | NaN              | 134.22           | 213.91          |


![dftfft](https://github.com/user-attachments/assets/2774100f-81ce-48ec-a619-feb2937e6dcf)


## Analysis And Inferences
- To accommodate a wide range of frequencies, this benchmark utilizes an array of N point DFT values instead of a real-time signal, as the latter may not provide a comprehensive interpretation for all frequencies. 
- 5 trials are run for accurate results for each N point DFT and their average is finally computed and used for benchmarking purpose. 
- From the observations we see that FFT is almost 4x faster than DFT in computation. 
- DFT is benchmarked only till 4096 N point DFT as beyond that there’s no significant inference and it crashes the code due to memory overhead. 
- FFT is run on both CPU and GPU . 
- We find that FFT is initially faster on CPU than on GPU but for large N point DFT such as 65536 GPU computes faster than FFT and the trend continues, this shows that for large values of N GPU is faster due to parallelism. 
- For running FFT on GPU Nvidia Smi Library is used which helps in fetching memory usage, GPU for MATLAB, energy consumption. 
- We observe from the memory graph that the trend of memory suage by the GPU is constant and then increases, while FFT on CPU initially uses more memory and then decreases until it reaches a point where memory usage again increases-At this point the FFT on GPU surpasses the FFT run on CPU.

---

##  Conclusions

- **FFT** is significantly **faster** than traditional **DFT**.
- **GPU acceleration is beneficial for large FFT sizes** and can significantly reduce computation time for sizes ≥ 65536.
- **MATLAB's GPU computing features are reliable** for high-accuracy FFT computations even on laptop GPUs.
- **Energy and thermal behavior** are favorable for practical, portable systems.

---

## Future Enchancements

- Benchmark with **other libraries** (cuFFT, FFTW, Intel MKL).
- Extend analysis to **2D and 3D FFTs**.
- Test **different data precisions** (single vs double precision).
- Include **multi-threaded CPU baselines** for fairer comparison.

---


## Contributing

Contributions are welcome!  
If you have suggestions, improvements, or find bugs, feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

##  Acknowledgments

- MATLAB documentation
- NVIDIA CUDA Toolkit
- Open-source FFT libraries inspiration

---

