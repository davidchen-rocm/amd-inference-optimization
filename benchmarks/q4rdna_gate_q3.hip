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
constexpr int Q4_TILE_BYTES = 1088;
constexpr int Q3_TILE_BYTES = 832;
constexpr int TILES = ROWS / WAVE * GROUPS;
constexpr int Q4_MATRIX_BYTES = TILES * Q4_TILE_BYTES;
constexpr int Q3_MATRIX_BYTES = TILES * Q3_TILE_BYTES;
constexpr int WEIGHT_COPIES = 4;
constexpr int PRECONDITION_ITERATIONS = 5000;
constexpr int WARMUP_ITERATIONS = 100;
constexpr int TIMED_ITERATIONS = 1000;
constexpr int TIMED_ROUNDS = 10;

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

__device__ __forceinline__ int int4_low(uint8_t value) {
    return int(static_cast<int8_t>(value << 4)) >> 4;
}

__device__ __forceinline__ int int4_high(uint8_t value) {
    return int(static_cast<int8_t>(value)) >> 4;
}

__device__ __forceinline__ int int3(uint32_t value) {
    return (int(value & 7u) ^ 4) - 4;
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

template <bool gate_q3>
__global__ __launch_bounds__(THREADS) void gate_up(
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
        const uint8_t * up_tile =
            up_weights + uint64_t(row_tile * GROUPS + group) * Q4_TILE_BYTES;
        const uint8_t * gate_tile = gate_weights +
            uint64_t(row_tile * GROUPS + group) *
                (gate_q3 ? Q3_TILE_BYTES : Q4_TILE_BYTES);
        float up_low = 0.0f;
        float up_high = 0.0f;
        float gate_value = 0.0f;
#pragma unroll
        for (int chunk = 0; chunk < 8; ++chunk) {
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int index = chunk * 4 + pair;
                const float activation_low = activation[group * GROUP + 2 * index];
                const float activation_high = activation[group * GROUP + 2 * index + 1];
                const uint8_t up_packed = up_tile[64 + index * WAVE + lane];
                up_low = fmaf(float(int4_low(up_packed)), activation_low, up_low);
                up_high = fmaf(float(int4_high(up_packed)), activation_high, up_high);
            }
            if constexpr (gate_q3) {
                const uint32_t packed =
                    uint32_t(gate_tile[64 + (chunk * 3 + 0) * WAVE + lane]) |
                    uint32_t(gate_tile[64 + (chunk * 3 + 1) * WAVE + lane]) << 8 |
                    uint32_t(gate_tile[64 + (chunk * 3 + 2) * WAVE + lane]) << 16;
#pragma unroll
                for (int value = 0; value < 8; ++value) {
                    gate_value = fmaf(
                        float(int3(packed >> (3 * value))),
                        activation[group * GROUP + chunk * 8 + value],
                        gate_value);
                }
            } else {
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const int index = chunk * 4 + pair;
                    const uint8_t packed = gate_tile[64 + index * WAVE + lane];
                    gate_value = fmaf(
                        float(int4_low(packed)),
                        activation[group * GROUP + 2 * index],
                        gate_value);
                    gate_value = fmaf(
                        float(int4_high(packed)),
                        activation[group * GROUP + 2 * index + 1],
                        gate_value);
                }
            }
        }
        const float up_scale =
            __half2float(reinterpret_cast<const __half *>(up_tile)[lane]);
        const float gate_scale =
            __half2float(reinterpret_cast<const __half *>(gate_tile)[lane]);
        up_sum = fmaf(up_scale, up_low + up_high, up_sum);
        gate_sum = fmaf(gate_scale, gate_value, gate_sum);
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

void set_scales(std::vector<uint8_t> & weights, int tile_bytes) {
    constexpr uint16_t scale = 0x2800;
    for (int tile = 0; tile < TILES; ++tile) {
        uint8_t * destination = weights.data() + uint64_t(tile) * tile_bytes;
        for (int lane = 0; lane < WAVE; ++lane) {
            std::memcpy(destination + lane * sizeof(scale), &scale, sizeof(scale));
        }
    }
}

void initialize_up(std::vector<uint8_t> & up) {
    set_scales(up, Q4_TILE_BYTES);
    uint32_t state = 0x12345678;
    for (int tile = 0; tile < TILES; ++tile) {
        uint8_t * destination = up.data() + uint64_t(tile) * Q4_TILE_BYTES;
        for (int index = 64; index < Q4_TILE_BYTES; ++index) {
            destination[index] = uint8_t(next_random(state));
        }
    }
}

void initialize_gate(
        std::vector<uint8_t> & q4_gate,
        std::vector<uint8_t> & q3_gate) {
    set_scales(q4_gate, Q4_TILE_BYTES);
    set_scales(q3_gate, Q3_TILE_BYTES);
    uint32_t state = 0x9abcdef0;
    for (int tile = 0; tile < TILES; ++tile) {
        uint8_t * q4 = q4_gate.data() + uint64_t(tile) * Q4_TILE_BYTES;
        uint8_t * q3 = q3_gate.data() + uint64_t(tile) * Q3_TILE_BYTES;
        for (int lane = 0; lane < WAVE; ++lane) {
            int values[GROUP];
            for (int index = 0; index < GROUP; ++index) {
                values[index] = int(next_random(state) & 7u) - 4;
            }
            for (int pair = 0; pair < GROUP / 2; ++pair) {
                q4[64 + pair * WAVE + lane] =
                    uint8_t(values[2 * pair] & 15) |
                    uint8_t((values[2 * pair + 1] & 15) << 4);
            }
            for (int chunk = 0; chunk < 8; ++chunk) {
                uint32_t bits = 0;
                for (int value = 0; value < 8; ++value) {
                    bits |= uint32_t(values[chunk * 8 + value] & 7) << (3 * value);
                }
                for (int byte = 0; byte < 3; ++byte) {
                    q3[64 + (chunk * 3 + byte) * WAVE + lane] = uint8_t(bits >> (8 * byte));
                }
            }
        }
    }
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

double cv(const std::vector<double> & values) {
    const double average = mean(values);
    double sum = 0.0;
    for (double value : values) sum += (value - average) * (value - average);
    return std::sqrt(sum / (values.size() - 1)) / average * 100.0;
}

} // namespace

int main() try {
    std::vector<uint8_t> up(Q4_MATRIX_BYTES);
    std::vector<uint8_t> q4_gate(Q4_MATRIX_BYTES);
    std::vector<uint8_t> q3_gate(Q3_MATRIX_BYTES);
    initialize_up(up);
    initialize_gate(q4_gate, q3_gate);
    std::vector<float> activation(COLUMNS);
    for (int index = 0; index < COLUMNS; ++index) {
        activation[index] = std::sin(float(index) * 0.013f) * 0.25f;
    }

    uint8_t * device_up = nullptr;
    uint8_t * device_q4_gate = nullptr;
    uint8_t * device_q3_gate = nullptr;
    float * device_activation = nullptr;
    float * device_q4_output = nullptr;
    float * device_q3_output = nullptr;
    check(hipSetDevice(0));
    check(hipMalloc(&device_up, uint64_t(Q4_MATRIX_BYTES) * WEIGHT_COPIES));
    check(hipMalloc(&device_q4_gate, uint64_t(Q4_MATRIX_BYTES) * WEIGHT_COPIES));
    check(hipMalloc(&device_q3_gate, uint64_t(Q3_MATRIX_BYTES) * WEIGHT_COPIES));
    check(hipMalloc(&device_activation, activation.size() * sizeof(float)));
    check(hipMalloc(&device_q4_output, ROWS * sizeof(float)));
    check(hipMalloc(&device_q3_output, ROWS * sizeof(float)));
    for (int copy = 0; copy < WEIGHT_COPIES; ++copy) {
        check(hipMemcpy(device_up + uint64_t(copy) * Q4_MATRIX_BYTES, up.data(), Q4_MATRIX_BYTES, hipMemcpyHostToDevice));
        check(hipMemcpy(device_q4_gate + uint64_t(copy) * Q4_MATRIX_BYTES, q4_gate.data(), Q4_MATRIX_BYTES, hipMemcpyHostToDevice));
        check(hipMemcpy(device_q3_gate + uint64_t(copy) * Q3_MATRIX_BYTES, q3_gate.data(), Q3_MATRIX_BYTES, hipMemcpyHostToDevice));
    }
    check(hipMemcpy(device_activation, activation.data(), activation.size() * sizeof(float), hipMemcpyHostToDevice));

    const dim3 grid(ROWS / WAVE);
    const dim3 block(THREADS);
    auto launch_q4 = [&](int copy) {
        hipLaunchKernelGGL(
            (gate_up<false>), grid, block, 0, 0,
            device_up + uint64_t(copy) * Q4_MATRIX_BYTES,
            device_q4_gate + uint64_t(copy) * Q4_MATRIX_BYTES,
            device_activation, device_q4_output);
    };
    auto launch_q3 = [&](int copy) {
        hipLaunchKernelGGL(
            (gate_up<true>), grid, block, 0, 0,
            device_up + uint64_t(copy) * Q4_MATRIX_BYTES,
            device_q3_gate + uint64_t(copy) * Q3_MATRIX_BYTES,
            device_activation, device_q3_output);
    };
    launch_q4(0);
    launch_q3(0);
    check(hipDeviceSynchronize());
    std::vector<float> q4_output(ROWS);
    std::vector<float> q3_output(ROWS);
    check(hipMemcpy(q4_output.data(), device_q4_output, ROWS * sizeof(float), hipMemcpyDeviceToHost));
    check(hipMemcpy(q3_output.data(), device_q3_output, ROWS * sizeof(float), hipMemcpyDeviceToHost));
    double max_error = 0.0;
    for (int row = 0; row < ROWS; ++row) {
        max_error = std::max(max_error, std::abs(double(q4_output[row]) - q3_output[row]));
    }

    for (int iteration = 0; iteration < PRECONDITION_ITERATIONS; ++iteration) {
        launch_q4(iteration % WEIGHT_COPIES);
        launch_q3(iteration % WEIGHT_COPIES);
    }
    check(hipDeviceSynchronize());
    std::vector<double> q4_us;
    std::vector<double> q3_us;
    for (int round = 0; round < TIMED_ROUNDS; ++round) {
        if ((round & 1) == 0) {
            q4_us.push_back(benchmark(launch_q4));
            q3_us.push_back(benchmark(launch_q3));
        } else {
            q3_us.push_back(benchmark(launch_q3));
            q4_us.push_back(benchmark(launch_q4));
        }
    }
    const double q4_mean = mean(q4_us);
    const double q3_mean = mean(q3_us);
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0));
    std::cout << std::fixed << std::setprecision(6)
              << "{\n"
              << "  \"schema\": \"gpuopt.q4rdna-gate-q3-microbenchmark.v1\",\n"
              << "  \"gpu\": \"" << properties.name << "\",\n"
              << "  \"gfx\": \"" << properties.gcnArchName << "\",\n"
              << "  \"shape\": {\"rows\": " << ROWS << ", \"columns\": " << COLUMNS << "},\n"
              << "  \"split_waves\": " << SPLIT_WAVES << ",\n"
              << "  \"q4_gate_bytes\": " << Q4_MATRIX_BYTES << ",\n"
              << "  \"q3_gate_bytes\": " << Q3_MATRIX_BYTES << ",\n"
              << "  \"gate_byte_reduction_percent\": "
              << (1.0 - double(Q3_MATRIX_BYTES) / Q4_MATRIX_BYTES) * 100.0 << ",\n"
              << "  \"fused_weight_byte_reduction_percent\": "
              << (1.0 - double(Q4_MATRIX_BYTES + Q3_MATRIX_BYTES) / (2.0 * Q4_MATRIX_BYTES)) * 100.0 << ",\n"
              << "  \"max_absolute_error\": " << max_error << ",\n"
              << "  \"q4_mean_us\": " << q4_mean << ",\n"
              << "  \"q3_mean_us\": " << q3_mean << ",\n"
              << "  \"q4_cv_percent\": " << cv(q4_us) << ",\n"
              << "  \"q3_cv_percent\": " << cv(q3_us) << ",\n"
              << "  \"q3_improvement_percent\": " << (q4_mean / q3_mean - 1.0) * 100.0 << "\n"
              << "}\n";
    return max_error == 0.0 ? 0 : 1;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
