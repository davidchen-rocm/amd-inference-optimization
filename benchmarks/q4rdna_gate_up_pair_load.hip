#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

constexpr int ROWS = 12288;
constexpr int COLUMNS = 4096;
constexpr int WAVE = 32;
constexpr int SPLIT_WAVES = 8;
constexpr int THREADS = WAVE * SPLIT_WAVES;
constexpr int GROUP = 64;
constexpr int GROUPS = COLUMNS / GROUP;
constexpr int TILE_BYTES = 1088;
constexpr int PAIRED_TILE_BYTES = TILE_BYTES * 2;
constexpr int MATRIX_BYTES = ROWS / WAVE * GROUPS * TILE_BYTES;
constexpr int PAIRED_BYTES = MATRIX_BYTES * 2;
constexpr int WEIGHT_COPIES = 4;
constexpr int PRECONDITION_ITERATIONS = 5000;
constexpr int WARMUP_ITERATIONS = 100;
constexpr int TIMED_ITERATIONS = 1000;
constexpr int TIMED_ROUNDS = 10;

void check(hipError_t status) {
    if (status != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(status));
    }
}

__device__ __forceinline__ int low_int4(uint8_t value) {
    return int(static_cast<int8_t>(value << 4)) >> 4;
}

__device__ __forceinline__ int high_int4(uint8_t value) {
    return int(static_cast<int8_t>(value)) >> 4;
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

__global__ __launch_bounds__(THREADS) void gate_up_separate(
        const uint8_t * up_weights,
        const uint8_t * gate_weights,
        const float * activation,
        float * output) {
    __shared__ float up_partial[SPLIT_WAVES][WAVE];
    __shared__ float gate_partial[SPLIT_WAVES][WAVE];
    const int lane = threadIdx.x & (WAVE - 1);
    const int split = threadIdx.x / WAVE;
    const int row_tile = blockIdx.x;
    float up_sum = 0.0f;
    float gate_sum = 0.0f;

    for (int group = split; group < GROUPS; group += SPLIT_WAVES) {
        const uint64_t tile_offset = uint64_t(row_tile * GROUPS + group) * TILE_BYTES;
        const uint8_t * up_tile = up_weights + tile_offset;
        const uint8_t * gate_tile = gate_weights + tile_offset;
        float up_low = 0.0f;
        float up_high = 0.0f;
        float gate_low = 0.0f;
        float gate_high = 0.0f;
#pragma unroll 4
        for (int i = 0; i < GROUP / 2; ++i) {
            const float activation_low = activation[group * GROUP + 2 * i];
            const float activation_high = activation[group * GROUP + 2 * i + 1];
            const uint8_t up_packed = up_tile[64 + i * WAVE + lane];
            const uint8_t gate_packed = gate_tile[64 + i * WAVE + lane];
            up_low = fmaf(float(low_int4(up_packed)), activation_low, up_low);
            up_high = fmaf(float(high_int4(up_packed)), activation_high, up_high);
            gate_low = fmaf(float(low_int4(gate_packed)), activation_low, gate_low);
            gate_high = fmaf(float(high_int4(gate_packed)), activation_high, gate_high);
        }
        const float up_scale = __half2float(reinterpret_cast<const __half *>(up_tile)[lane]);
        const float gate_scale = __half2float(reinterpret_cast<const __half *>(gate_tile)[lane]);
        up_sum = fmaf(up_scale, up_low + up_high, up_sum);
        gate_sum = fmaf(gate_scale, gate_low + gate_high, gate_sum);
    }

    up_partial[split][lane] = up_sum;
    gate_partial[split][lane] = gate_sum;
    __syncthreads();
    if (split == 0) {
#pragma unroll
        for (int other = 1; other < SPLIT_WAVES; ++other) {
            up_sum += up_partial[other][lane];
            gate_sum += gate_partial[other][lane];
        }
        output[row_tile * WAVE + lane] = up_sum * silu(gate_sum);
    }
}

__global__ __launch_bounds__(THREADS) void gate_up_paired(
        const uint8_t * paired_weights,
        const float * activation,
        float * output) {
    __shared__ float up_partial[SPLIT_WAVES][WAVE];
    __shared__ float gate_partial[SPLIT_WAVES][WAVE];
    const int lane = threadIdx.x & (WAVE - 1);
    const int split = threadIdx.x / WAVE;
    const int row_tile = blockIdx.x;
    float up_sum = 0.0f;
    float gate_sum = 0.0f;

    for (int group = split; group < GROUPS; group += SPLIT_WAVES) {
        const uint8_t * tile =
            paired_weights + uint64_t(row_tile * GROUPS + group) * PAIRED_TILE_BYTES;
        const uint16_t * pairs = reinterpret_cast<const uint16_t *>(tile + 128);
        float up_low = 0.0f;
        float up_high = 0.0f;
        float gate_low = 0.0f;
        float gate_high = 0.0f;
#pragma unroll 4
        for (int i = 0; i < GROUP / 2; ++i) {
            const float activation_low = activation[group * GROUP + 2 * i];
            const float activation_high = activation[group * GROUP + 2 * i + 1];
            const uint16_t packed = pairs[i * WAVE + lane];
            const uint8_t up_packed = packed & 0xff;
            const uint8_t gate_packed = packed >> 8;
            up_low = fmaf(float(low_int4(up_packed)), activation_low, up_low);
            up_high = fmaf(float(high_int4(up_packed)), activation_high, up_high);
            gate_low = fmaf(float(low_int4(gate_packed)), activation_low, gate_low);
            gate_high = fmaf(float(high_int4(gate_packed)), activation_high, gate_high);
        }
        const float up_scale = __half2float(reinterpret_cast<const __half *>(tile)[lane]);
        const float gate_scale = __half2float(reinterpret_cast<const __half *>(tile + 64)[lane]);
        up_sum = fmaf(up_scale, up_low + up_high, up_sum);
        gate_sum = fmaf(gate_scale, gate_low + gate_high, gate_sum);
    }

    up_partial[split][lane] = up_sum;
    gate_partial[split][lane] = gate_sum;
    __syncthreads();
    if (split == 0) {
#pragma unroll
        for (int other = 1; other < SPLIT_WAVES; ++other) {
            up_sum += up_partial[other][lane];
            gate_sum += gate_partial[other][lane];
        }
        output[row_tile * WAVE + lane] = up_sum * silu(gate_sum);
    }
}

uint32_t next_random(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void initialize_weights(std::vector<uint8_t> & weights, uint32_t seed) {
    constexpr uint16_t scale_bits = 0x2800;
    uint32_t state = seed;
    for (int tile = 0; tile < ROWS / WAVE * GROUPS; ++tile) {
        uint8_t * destination = weights.data() + uint64_t(tile) * TILE_BYTES;
        for (int lane = 0; lane < WAVE; ++lane) {
            std::memcpy(destination + lane * sizeof(scale_bits), &scale_bits, sizeof(scale_bits));
        }
        for (int index = 64; index < TILE_BYTES; ++index) {
            destination[index] = static_cast<uint8_t>(next_random(state));
        }
    }
}

std::vector<uint8_t> pair_weights(
        const std::vector<uint8_t> & up,
        const std::vector<uint8_t> & gate) {
    std::vector<uint8_t> paired(PAIRED_BYTES);
    for (int tile = 0; tile < ROWS / WAVE * GROUPS; ++tile) {
        const uint8_t * up_tile = up.data() + uint64_t(tile) * TILE_BYTES;
        const uint8_t * gate_tile = gate.data() + uint64_t(tile) * TILE_BYTES;
        uint8_t * destination = paired.data() + uint64_t(tile) * PAIRED_TILE_BYTES;
        std::memcpy(destination, up_tile, 64);
        std::memcpy(destination + 64, gate_tile, 64);
        uint16_t * pairs = reinterpret_cast<uint16_t *>(destination + 128);
        for (int index = 0; index < TILE_BYTES - 64; ++index) {
            pairs[index] = uint16_t(up_tile[64 + index]) | uint16_t(gate_tile[64 + index]) << 8;
        }
    }
    return paired;
}

template <typename Launch>
double benchmark(Launch launch) {
    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        launch(iteration % WEIGHT_COPIES);
    }
    check(hipDeviceSynchronize());
    hipEvent_t begin;
    hipEvent_t end;
    check(hipEventCreate(&begin));
    check(hipEventCreate(&end));
    check(hipEventRecord(begin));
    for (int iteration = 0; iteration < TIMED_ITERATIONS; ++iteration) {
        launch(iteration % WEIGHT_COPIES);
    }
    check(hipEventRecord(end));
    check(hipEventSynchronize(end));
    float milliseconds = 0.0f;
    check(hipEventElapsedTime(&milliseconds, begin, end));
    check(hipEventDestroy(begin));
    check(hipEventDestroy(end));
    return milliseconds * 1000.0 / TIMED_ITERATIONS;
}

double mean(const std::vector<double> & values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double standard_deviation(const std::vector<double> & values) {
    const double average = mean(values);
    double sum = 0.0;
    for (double value : values) {
        sum += (value - average) * (value - average);
    }
    return std::sqrt(sum / (values.size() - 1));
}

} // namespace

int main() try {
    std::vector<uint8_t> up(MATRIX_BYTES);
    std::vector<uint8_t> gate(MATRIX_BYTES);
    initialize_weights(up, 0x12345678);
    initialize_weights(gate, 0x9abcdef0);
    const std::vector<uint8_t> paired = pair_weights(up, gate);
    std::vector<float> activation(COLUMNS);
    for (int index = 0; index < COLUMNS; ++index) {
        activation[index] = std::sin(float(index) * 0.013f) * 0.25f;
    }

    uint8_t * device_up = nullptr;
    uint8_t * device_gate = nullptr;
    uint8_t * device_paired = nullptr;
    float * device_activation = nullptr;
    float * device_separate_output = nullptr;
    float * device_paired_output = nullptr;
    check(hipSetDevice(0));
    check(hipMalloc(&device_up, uint64_t(MATRIX_BYTES) * WEIGHT_COPIES));
    check(hipMalloc(&device_gate, uint64_t(MATRIX_BYTES) * WEIGHT_COPIES));
    check(hipMalloc(&device_paired, uint64_t(PAIRED_BYTES) * WEIGHT_COPIES));
    check(hipMalloc(&device_activation, activation.size() * sizeof(float)));
    check(hipMalloc(&device_separate_output, ROWS * sizeof(float)));
    check(hipMalloc(&device_paired_output, ROWS * sizeof(float)));
    for (int copy = 0; copy < WEIGHT_COPIES; ++copy) {
        check(hipMemcpy(device_up + uint64_t(copy) * MATRIX_BYTES, up.data(), MATRIX_BYTES, hipMemcpyHostToDevice));
        check(hipMemcpy(device_gate + uint64_t(copy) * MATRIX_BYTES, gate.data(), MATRIX_BYTES, hipMemcpyHostToDevice));
        check(hipMemcpy(device_paired + uint64_t(copy) * PAIRED_BYTES, paired.data(), PAIRED_BYTES, hipMemcpyHostToDevice));
    }
    check(hipMemcpy(device_activation, activation.data(), activation.size() * sizeof(float), hipMemcpyHostToDevice));

    const dim3 grid(ROWS / WAVE);
    const dim3 block(THREADS);
    auto launch_separate = [&](int copy) {
        hipLaunchKernelGGL(
            gate_up_separate,
            grid,
            block,
            0,
            0,
            device_up + uint64_t(copy) * MATRIX_BYTES,
            device_gate + uint64_t(copy) * MATRIX_BYTES,
            device_activation,
            device_separate_output);
    };
    auto launch_paired = [&](int copy) {
        hipLaunchKernelGGL(
            gate_up_paired,
            grid,
            block,
            0,
            0,
            device_paired + uint64_t(copy) * PAIRED_BYTES,
            device_activation,
            device_paired_output);
    };

    launch_separate(0);
    launch_paired(0);
    check(hipDeviceSynchronize());
    std::vector<float> separate_output(ROWS);
    std::vector<float> paired_output(ROWS);
    check(hipMemcpy(separate_output.data(), device_separate_output, ROWS * sizeof(float), hipMemcpyDeviceToHost));
    check(hipMemcpy(paired_output.data(), device_paired_output, ROWS * sizeof(float), hipMemcpyDeviceToHost));
    double max_absolute_error = 0.0;
    for (int index = 0; index < ROWS; ++index) {
        max_absolute_error = std::max(
            max_absolute_error,
            std::abs(double(separate_output[index]) - paired_output[index]));
    }
    if (max_absolute_error != 0.0) {
        throw std::runtime_error("paired layout changed the gate/up result");
    }

    for (int iteration = 0; iteration < PRECONDITION_ITERATIONS; ++iteration) {
        launch_separate(iteration % WEIGHT_COPIES);
        launch_paired(iteration % WEIGHT_COPIES);
    }
    check(hipDeviceSynchronize());

    std::vector<double> separate_us;
    std::vector<double> paired_us;
    for (int round = 0; round < TIMED_ROUNDS; ++round) {
        if ((round & 1) == 0) {
            separate_us.push_back(benchmark(launch_separate));
            paired_us.push_back(benchmark(launch_paired));
        } else {
            paired_us.push_back(benchmark(launch_paired));
            separate_us.push_back(benchmark(launch_separate));
        }
    }
    const double separate_mean = mean(separate_us);
    const double paired_mean = mean(paired_us);
    const double improvement = (separate_mean / paired_mean - 1.0) * 100.0;
    std::vector<double> pairwise_improvement;
    for (int round = 0; round < TIMED_ROUNDS; ++round) {
        pairwise_improvement.push_back((separate_us[round] / paired_us[round] - 1.0) * 100.0);
    }

    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0));
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "{\n";
    std::cout << "  \"schema\": \"gpuopt.q4rdna-gate-up-pair-load.v1\",\n";
    std::cout << "  \"gpu\": \"" << properties.name << "\",\n";
    std::cout << "  \"gfx\": \"" << properties.gcnArchName << "\",\n";
    std::cout << "  \"shape\": {\"rows\": " << ROWS << ", \"columns\": " << COLUMNS << "},\n";
    std::cout << "  \"split_waves\": " << SPLIT_WAVES << ",\n";
    std::cout << "  \"weight_bytes\": " << PAIRED_BYTES << ",\n";
    std::cout << "  \"weight_copies\": " << WEIGHT_COPIES << ",\n";
    std::cout << "  \"precondition_iterations_per_kernel\": " << PRECONDITION_ITERATIONS << ",\n";
    std::cout << "  \"timed_iterations_per_round\": " << TIMED_ITERATIONS << ",\n";
    std::cout << "  \"max_absolute_error\": " << max_absolute_error << ",\n";
    std::cout << "  \"separate_us\": [";
    for (size_t index = 0; index < separate_us.size(); ++index) {
        if (index) std::cout << ", ";
        std::cout << separate_us[index];
    }
    std::cout << "],\n";
    std::cout << "  \"paired_us\": [";
    for (size_t index = 0; index < paired_us.size(); ++index) {
        if (index) std::cout << ", ";
        std::cout << paired_us[index];
    }
    std::cout << "],\n";
    std::cout << "  \"separate_mean_us\": " << separate_mean << ",\n";
    std::cout << "  \"paired_mean_us\": " << paired_mean << ",\n";
    std::cout << "  \"separate_cv_percent\": " << standard_deviation(separate_us) / separate_mean * 100.0 << ",\n";
    std::cout << "  \"paired_cv_percent\": " << standard_deviation(paired_us) / paired_mean * 100.0 << ",\n";
    std::cout << "  \"paired_improvement_percent\": " << improvement << ",\n";
    std::cout << "  \"pairwise_improvement_percent\": [";
    for (size_t index = 0; index < pairwise_improvement.size(); ++index) {
        if (index) std::cout << ", ";
        std::cout << pairwise_improvement[index];
    }
    std::cout << "],\n";
    std::cout << "  \"pairwise_improvement_mean_percent\": " << mean(pairwise_improvement) << ",\n";
    std::cout << "  \"pairwise_improvement_cv_percent\": "
              << standard_deviation(pairwise_improvement) / std::abs(mean(pairwise_improvement)) * 100.0 << "\n";
    std::cout << "}\n";

    check(hipFree(device_paired_output));
    check(hipFree(device_separate_output));
    check(hipFree(device_activation));
    check(hipFree(device_paired));
    check(hipFree(device_gate));
    check(hipFree(device_up));
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
