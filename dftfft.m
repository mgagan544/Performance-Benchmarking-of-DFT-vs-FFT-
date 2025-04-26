clc;
clear all;
close all;
clearvars;

% Function to check if nvidia-smi is available
function hasNvidiaSmi = checkNvidiaSmi()
    if ispc % for Windows
        [status, ~] = system('where nvidia-smi');
        hasNvidiaSmi = (status == 0);
    else %  if using Linux/Mac
        [status, ~] = system('which nvidia-smi');
        hasNvidiaSmi = (status == 0);
    end
end

% Function measures GPU power using smi 
function power = getGpuPower()
    [status, output] = system('nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits');
    if status == 0
        power = str2double(output);
    else
        power = NaN;
    end
end

% Check for GPU availability
gpuAvailable = (exist('gpuDevice', 'file') == 2) && (gpuDeviceCount > 0);
if gpuAvailable
    g = gpuDevice();
    reset(g);
    fprintf('Using GPU: %s with %.2f GB memory\n', g.Name, g.AvailableMemory/1e9);
    % Check for nvidia-smi for power measurements
    canMeasurePower = checkNvidiaSmi();
    if canMeasurePower
        fprintf('NVIDIA SMI available - will measure GPU power\n');
    else
        fprintf('NVIDIA SMI not available - cannot measure GPU power\n');
    end
else
    fprintf('No GPU available\n');
    canMeasurePower = false;
end

% allocation of size based upon the pc specifications
% this is done in order to prevent crashes and memory overhead
systemMemory = memory;
availableRAM = systemMemory.MemAvailableAllArrays;
fprintf('Available system memory: %.2f GB\n', availableRAM/1e9);

% Base sizes that work on most systems
Ns = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];

%larger sizes based on available memory
if availableRAM > 500e6  % 500MB minimum
    Ns = [Ns, 131072, 262144];
    if availableRAM > 1e9  % 1GB minimum
        Ns = [Ns, 524288, 1048576];
        if availableRAM > 4e9  % 4GB minimum
            Ns = [Ns, 2097152, 4194304];
            if availableRAM > 8e9  % 8GB minimum
                Ns = [Ns, 8388608];
                if availableRAM > 16e9  % 16GB minimum
                    Ns = [Ns, 16777216];
                    if availableRAM > 32e9  % 32GB minimum
                        Ns = [Ns, 33554432];
                        if availableRAM > 64e9  % 64GB minimum
                            Ns = [Ns, 67108864, 134217728];
                        end
                    end
                end
            end
        end
    end
end

num_trials = 5;  % 5 trials are taken for accurate readings 

%  result arrays are initially padded to zeros
cpuFFT_times = zeros(size(Ns));
gpuFFT_times = nan(size(Ns));
dft_times = nan(size(Ns));

cpuFFT_memory = zeros(size(Ns));
gpuFFT_memory = nan(size(Ns));
dft_memory = nan(size(Ns));

cpuFFT_energy = nan(size(Ns));  % Will remain NaN unless system-specific code is added
gpuFFT_energy = nan(size(Ns));
dft_energy = nan(size(Ns));     % Will remain NaN unless system-specific code is added

fprintf('\n==== Comprehensive FFT Benchmark ====\n');
fprintf('Testing %d sizes from %d to %d points\n', length(Ns), min(Ns), max(Ns));

% Create figures for real-time plotting
figure('Name', 'FFT Benchmarks: Time');
timeAxes = gca;
title('Execution Time');
xlabel('Array Size (N)');
ylabel('Time (seconds)');
set(gca, 'XScale', 'log', 'YScale', 'log', 'XGrid', 'on', 'YGrid', 'on');
hold(timeAxes, 'on');

figure('Name', 'FFT Benchmarks: Memory');
memAxes = gca;
title('Memory Usage');
xlabel('Array Size (N)');
ylabel('Memory (MB)');
set(gca, 'XScale', 'log', 'YScale', 'log', 'XGrid', 'on', 'YGrid', 'on');
hold(memAxes, 'on');

if canMeasurePower
    figure('Name', 'FFT Benchmarks: Energy');
    energyAxes = gca;
    title('Energy Consumption');
    xlabel('Array Size (N)');
    ylabel('Energy (Joules)');
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XGrid', 'on', 'YGrid', 'on');
    hold(energyAxes, 'on');
end

% Main benchmarking loop
for i = 1:length(Ns)
    N = Ns(i);
    fprintf('\nTesting N = %d\n', N);
    
    try
        % Create signal with actual frequency components 
        t = linspace(0, 1, N);
        f1 = min(50, N/10);          % Base frequency scaled to avoid aliasing
        f2 = min(120, N/4);          % Mid frequency
        f3 = min(375, N/2-10);       % High frequency
        x = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t) + 0.25*sin(2*pi*f3*t);
        
        % 1. CPU FFT Benchmarking
        fprintf('  CPU FFT: ');
        t_cpu = zeros(1, num_trials);
        mem_cpu = zeros(1, num_trials);
        
        for j = 1:num_trials
            % Force garbage collection
            pause(0.1);
            clear X_fft;
            pause(0.1);
            
            % Measure memory before
            memBefore = memory;
            memUsedBefore = memBefore.MemUsedMATLAB;
            
            % Time the operation
            t0 = tic;
            X_fft = fft(x);
            t_cpu(j) = toc(t0);
            
            % Measure memory after
            memAfter = memory;
            memUsedAfter = memAfter.MemUsedMATLAB;
            mem_cpu(j) = (memUsedAfter - memUsedBefore) / 1e6; % Convert to MB
            
            fprintf('.');
        end
        
        cpuFFT_times(i) = mean(t_cpu);
        cpuFFT_memory(i) = mean(mem_cpu);
        fprintf(' %.4f s, %.2f MB\n', cpuFFT_times(i), cpuFFT_memory(i));
        
        % 2. DFT Benchmarking (only for small N)
        if N <= 4096
            fprintf('  DFT:     ');
            t_dft = zeros(1, num_trials);
            mem_dft = zeros(1, num_trials);
            
            for j = 1:num_trials
                % Force garbage collection
                pause(0.1);
                clear twiddle X_dft;
                pause(0.1);
                
                % Measure memory before twiddle creation
                memBefore = memory;
                memUsedBefore = memBefore.MemUsedMATLAB;
                
                % Create twiddle factors
                n = 0:N-1;
                k = n';
                WN = exp(-1j*2*pi/N);
                twiddle = WN .^ (k*n);
                
                % Time the operation
                t0 = tic;
                X_dft = x * twiddle;
                t_dft(j) = toc(t0);
                
                % Measure memory after
                memAfter = memory;
                memUsedAfter = memAfter.MemUsedMATLAB;
                mem_dft(j) = (memUsedAfter - memUsedBefore) / 1e6; % Convert to MB
                
                fprintf('.');
            end
            
            dft_times(i) = mean(t_dft);
            dft_memory(i) = mean(mem_dft);
            fprintf(' %.4f s, %.2f MB\n', dft_times(i), dft_memory(i));
            
            % Verify results match between DFT and FFT
            max_error = max(abs(X_dft - X_fft));
            fprintf('  Max error between DFT and FFT: %.10e\n', max_error);
        end
        
        % 3. GPU FFT Benchmarking
        if gpuAvailable
            fprintf('  GPU FFT: ');
            
            % To check if  array can fit in GPU memory
            mem_needed = N * 16 * 2;  % bytes 
            if mem_needed < g.AvailableMemory * 0.8  % Allow 20% overhead
                t_gpu = zeros(1, num_trials);
                mem_gpu = zeros(1, num_trials);
                energy_gpu = zeros(1, num_trials);
                
                for j = 1:num_trials
                    % Force garbage collection
                    pause(0.1);
                    clear xg X_gpu;
                    pause(0.1);
                    reset(g);
                    
                    % Measure GPU memory before
                    g = gpuDevice();
                    memGpuBefore = g.AvailableMemory;
                    
                    % Transfer data to GPU
                    xg = gpuArray(x);
                    
                    % Measure power before 
                    if canMeasurePower
                        powerBefore = getGpuPower();
                    end
                    
                    % Time the operation
                    wait(g);
                    t0 = tic;
                    X_gpu = fft(xg);
                    wait(g);
                    t_gpu(j) = toc(t0);
                    
                    % Measure power after benchmarking results
                    if canMeasurePower
                        powerAfter = getGpuPower();
                        avgPower = (powerBefore + powerAfter) / 2;
                        energy_gpu(j) = avgPower * t_gpu(j);
                    end
                    
                    % Measure GPU memory after results
                    g = gpuDevice();
                    memGpuAfter = g.AvailableMemory;
                    mem_gpu(j) = (memGpuBefore - memGpuAfter) / 1e6; % Convert to MB
                    
                    fprintf('.');
                end
                
                gpuFFT_times(i) = mean(t_gpu);
                gpuFFT_memory(i) = mean(mem_gpu);
                if canMeasurePower
                    gpuFFT_energy(i) = mean(energy_gpu);
                    fprintf(' %.4f s, %.2f MB, %.2f J\n', gpuFFT_times(i), gpuFFT_memory(i), gpuFFT_energy(i));
                else
                    fprintf(' %.4f s, %.2f MB\n', gpuFFT_times(i), gpuFFT_memory(i));
                end
                
                % Verify GPU results match CPU
                X_from_gpu = gather(X_gpu);
                max_error = max(abs(X_from_gpu - X_fft));
                fprintf('  Max error between GPU and CPU: %.10e\n', max_error);
            else
                fprintf(' [Skipped - insufficient GPU memory]\n');
            end
        end
        
        % Update plots in real-time
        % Time plot
        cla(timeAxes);
        if any(~isnan(dft_times))
            loglog(timeAxes, Ns(1:i), dft_times(1:i), 'r-o', 'LineWidth', 1.5, 'DisplayName', 'DFT');
        end
        loglog(timeAxes, Ns(1:i), cpuFFT_times(1:i), 'b-s', 'LineWidth', 1.5, 'DisplayName', 'CPU FFT');
        if any(~isnan(gpuFFT_times))
            loglog(timeAxes, Ns(1:i), gpuFFT_times(1:i), 'g-^', 'LineWidth', 1.5, 'DisplayName', 'GPU FFT');
        end
        legend(timeAxes, 'Location', 'northwest');
        drawnow;
        
        % Memory plot
        cla(memAxes);
        if any(~isnan(dft_memory))
            loglog(memAxes, Ns(1:i), dft_memory(1:i), 'r-o', 'LineWidth', 1.5, 'DisplayName', 'DFT');
        end
        loglog(memAxes, Ns(1:i), cpuFFT_memory(1:i), 'b-s', 'LineWidth', 1.5, 'DisplayName', 'CPU FFT');
        if any(~isnan(gpuFFT_memory))
            loglog(memAxes, Ns(1:i), gpuFFT_memory(1:i), 'g-^', 'LineWidth', 1.5, 'DisplayName', 'GPU FFT');
        end
        legend(memAxes, 'Location', 'northwest');
        drawnow;
        
        % Energy plot 
        if canMeasurePower
            cla(energyAxes);
            if any(~isnan(gpuFFT_energy))
                loglog(energyAxes, Ns(1:i), gpuFFT_energy(1:i), 'g-^', 'LineWidth', 1.5, 'DisplayName', 'GPU FFT');
                legend(energyAxes, 'Location', 'northwest');
                drawnow;
            end
        end
        
    catch e
        fprintf('\nError for N=%d: %s\n', N, e.message);
    end
end

%% Final Results and Analysis

% Find crossover point where GPU becomes faster than CPU
if any(~isnan(gpuFFT_times)) && any(~isnan(cpuFFT_times))
    valid_indices = ~isnan(gpuFFT_times) & ~isnan(cpuFFT_times);
    gpu_better_idx = find(gpuFFT_times < cpuFFT_times & valid_indices);
    
    if ~isempty(gpu_better_idx)
        gpu_kickin_N = Ns(gpu_better_idx(1));
        fprintf('\n? GPU becomes faster than CPU FFT at N = %d\n', gpu_kickin_N);
        
        % Calculate speedup at max size
        max_valid_idx = find(valid_indices, 1, 'last');
        if ~isempty(max_valid_idx)
            max_speedup = cpuFFT_times(max_valid_idx) / gpuFFT_times(max_valid_idx);
            fprintf('? Maximum GPU speedup: %.2fx at N = %d\n', max_speedup, Ns(max_valid_idx));
        end
    else
        fprintf('\n GPU never outperformed CPU in this range.\n');
    end
else
    fprintf('\n Insufficient data to determine GPU/CPU crossover point.\n');
end


if sum(~isnan(dft_times)) >= 2 && sum(~isnan(cpuFFT_times)) >= 2
    fprintf('\nAlgorithmic Complexity Verification:\n');
    
    
    valid_dft = find(~isnan(dft_times));
    valid_fft = find(~isnan(cpuFFT_times));
    
    if length(valid_dft) >= 2
        idx1 = valid_dft(1);
        idx2 = valid_dft(end);
        expected_ratio_N2 = (Ns(idx2)/Ns(idx1))^2;
        actual_ratio_dft = dft_times(idx2)/dft_times(idx1);
        fprintf('- DFT: Expected O(N²) ratio: %.2f, Actual ratio: %.2f\n', ...
            expected_ratio_N2, actual_ratio_dft);
    end
    
    if length(valid_fft) >= 2
        idx1 = valid_fft(1);
        idx2 = valid_fft(end);
        expected_ratio_NlogN = (Ns(idx2)*log2(Ns(idx2))) / (Ns(idx1)*log2(Ns(idx1)));
        actual_ratio_fft = cpuFFT_times(idx2)/cpuFFT_times(idx1);
        fprintf('- FFT: Expected O(N log N) ratio: %.2f, Actual ratio: %.2f\n', ...
            expected_ratio_NlogN, actual_ratio_fft);
    end
end

% Generate comprehensive report table
fprintf('\n%-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n', ...
    'N', 'DFT Time', 'CPU Time', 'GPU Time', 'DFT Mem', 'CPU Mem', 'GPU Mem');
fprintf(repmat('-', 1, 80));
fprintf('\n');

for i = 1:length(Ns)
    fprintf('%-10d | %-10.4f | %-10.4f | %-10.4f | %-10.2f | %-10.2f | %-10.2f\n', ...
        Ns(i), dft_times(i), cpuFFT_times(i), gpuFFT_times(i), ...
        dft_memory(i), cpuFFT_memory(i), gpuFFT_memory(i));
end

% Saving results localy to MAT file
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = sprintf('fft_benchmark_%s.mat', timestamp);
save(filename, 'Ns', 'dft_times', 'cpuFFT_times', 'gpuFFT_times', ...
    'dft_memory', 'cpuFFT_memory', 'gpuFFT_memory', 'gpuFFT_energy');
fprintf('\nResults saved to %s\n', filename);

% final comparison figure with all metrics
figure('Name', 'Comprehensive FFT Benchmark Results', 'Position', [100, 100, 1200, 800]);

% Time subplot
subplot(2, 2, 1);
if any(~isnan(dft_times))
    loglog(Ns, dft_times, 'r-o', 'LineWidth', 1.5); hold on;
end
loglog(Ns, cpuFFT_times, 'b-s', 'LineWidth', 1.5);
if any(~isnan(gpuFFT_times))
    loglog(Ns, gpuFFT_times, 'g-^', 'LineWidth', 1.5);
end
grid on;
xlabel('Signal Length (N)');
ylabel('Time (seconds)');
title('Execution Time');
legend_entries = {};
if any(~isnan(dft_times))
    legend_entries{end+1} = 'DFT (O(N²))';
end
legend_entries{end+1} = 'FFT CPU (O(N log N))';
if any(~isnan(gpuFFT_times))
    legend_entries{end+1} = 'FFT GPU';
end
legend(legend_entries, 'Location', 'northwest');

% Memory subplot
subplot(2, 2, 2);
if any(~isnan(dft_memory))
    loglog(Ns, dft_memory, 'r-o', 'LineWidth', 1.5); hold on;
end
loglog(Ns, cpuFFT_memory, 'b-s', 'LineWidth', 1.5);
if any(~isnan(gpuFFT_memory))
    loglog(Ns, gpuFFT_memory, 'g-^', 'LineWidth', 1.5);
end
grid on;
xlabel('Signal Length (N)');
ylabel('Memory (MB)');
title('Memory Usage');
legend_entries = {};
if any(~isnan(dft_memory))
    legend_entries{end+1} = 'DFT';
end
legend_entries{end+1} = 'FFT CPU';
if any(~isnan(gpuFFT_memory))
    legend_entries{end+1} = 'FFT GPU';
end
legend(legend_entries, 'Location', 'northwest');

% Energy subplot (if available)
subplot(2, 2, 3);
if any(~isnan(gpuFFT_energy))
    loglog(Ns, gpuFFT_energy, 'g-^', 'LineWidth', 1.5);
    grid on;
    xlabel('Signal Length (N)');
    ylabel('Energy (Joules)');
    title('Energy Consumption (GPU)');
    legend('FFT GPU', 'Location', 'northwest');
else
    text(0.5, 0.5, 'Energy data not available', 'HorizontalAlignment', 'center');
    title('Energy Consumption');
end

% Efficiency subplot (computations per joule)
subplot(2, 2, 4);
if any(~isnan(gpuFFT_energy))
    % Calculate efficiency: N*log2(N) operations per Joule
    valid_energy = ~isnan(gpuFFT_energy) & gpuFFT_energy > 0;
    if any(valid_energy)
        N_valid = Ns(valid_energy);
        ops_valid = N_valid .* log2(N_valid); % Approximate number of operations
        energy_valid = gpuFFT_energy(valid_energy);
        efficiency = ops_valid ./ energy_valid;
        
        loglog(N_valid, efficiency, 'g-^', 'LineWidth', 1.5);
        grid on;
        xlabel('Signal Length (N)');
        ylabel('Operations per Joule');
        title('Computational Efficiency (GPU)');
    else
        text(0.5, 0.5, 'Efficiency data not available', 'HorizontalAlignment', 'center');
        title('Computational Efficiency');
    end
else
    text(0.5, 0.5, 'Energy data not available', 'HorizontalAlignment', 'center');
    title('Computational Efficiency');
end

sgtitle('Comprehensive FFT Benchmark Results');