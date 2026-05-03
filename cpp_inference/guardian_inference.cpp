/*
 * guardian_inference.cpp
 * Guardian Drive -- LibTorch C++ Inference
 *
 * Production C++ inference using LibTorch -- same runtime
 * used in Tesla FSD stack. Loads TorchScript model and
 * runs inference with latency benchmarking.
 *
 * Built by Akilan Manivannan & Akila Lourdes Miriyala Francis
 */

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace std::chrono;

// Benchmark inference latency
struct BenchmarkResult {
    double median_ms;
    double p95_ms;
    double p99_ms;
    double min_ms;
    int    n_runs;
    int    batch_size;
};

BenchmarkResult benchmark(
    torch::jit::script::Module& model,
    int batch_size = 1,
    int n_warmup   = 20,
    int n_runs     = 200)
{
    auto input = torch::randn({batch_size, 4, 4200});
    std::vector<torch::jit::IValue> inputs = {input};

    // Warmup
    for (int i = 0; i < n_warmup; ++i) {
        model.forward(inputs);
    }

    // Timed runs
    std::vector<double> times;
    times.reserve(n_runs);

    for (int i = 0; i < n_runs; ++i) {
        auto t0  = high_resolution_clock::now();
        model.forward(inputs);
        auto t1  = high_resolution_clock::now();
        double ms = duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());

    BenchmarkResult res;
    res.n_runs     = n_runs;
    res.batch_size = batch_size;
    res.min_ms     = times[0];
    res.median_ms  = times[n_runs / 2];
    res.p95_ms     = times[static_cast<int>(n_runs * 0.95)];
    res.p99_ms     = times[static_cast<int>(n_runs * 0.99)];
    return res;
}

int main(int argc, char* argv[]) {
    std::string model_path = "wesad_tcn_scripted.pt";
    if (argc > 1) model_path = argv[1];

    std::cout << "================================================\n";
    std::cout << "Guardian Drive -- LibTorch C++ Inference\n";
    std::cout << "Akilan Manivannan & Akila Lourdes Miriyala Francis\n";
    std::cout << "================================================\n\n";

    // Load TorchScript model
    std::cout << "Loading model: " << model_path << "\n";
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        model.eval();
        std::cout << "Model loaded successfully\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        std::cerr << "Run: python benchmark_libtorch.py first\n";
        return 1;
    }

    // Verify inference
    auto test_input = torch::randn({1, 4, 4200});
    std::vector<torch::jit::IValue> test_inputs = {test_input};
    auto output = model.forward(test_inputs).toTensor();
    float prob  = torch::sigmoid(output).item<float>();
    std::cout << "Test inference: logit=" << output.item<float>()
              << " prob=" << prob << "\n\n";

    // Benchmark
    std::vector<int> batch_sizes = {1, 8, 32};
    std::cout << "Benchmarking (200 runs per batch size):\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "Batch  Median(ms)  p95(ms)  p99(ms)  Throughput\n";
    std::cout << std::string(60, '-') << "\n";

    // Save results to JSON
    std::ostringstream json;
    json << "{\n";
    json << "  \"model\": \"WESAD TCN (AUC 0.9738)\",\n";
    json << "  \"runtime\": \"LibTorch C++\",\n";
    json << "  \"authors\": \"Akilan Manivannan & Akila Lourdes Miriyala Francis\",\n";
    json << "  \"results\": [\n";

    for (int i = 0; i < (int)batch_sizes.size(); ++i) {
        int bs = batch_sizes[i];
        auto res = benchmark(model, bs);
        double tput = 1000.0 / res.median_ms * bs;

        std::cout << "  " << bs
                  << "      " << res.median_ms
                  << "       " << res.p95_ms
                  << "     " << res.p99_ms
                  << "    " << tput << " seq/s\n";

        json << "    {\"batch\": " << bs
             << ", \"median_ms\": " << res.median_ms
             << ", \"p95_ms\": " << res.p95_ms
             << ", \"p99_ms\": " << res.p99_ms
             << ", \"throughput\": " << tput << "}";
        if (i < (int)batch_sizes.size()-1) json << ",";
        json << "\n";
    }

    json << "  ]\n}\n";

    // Write results
    std::ofstream f("learned/results/libtorch_cpp_benchmark.json");
    f << json.str();
    f.close();
    std::cout << "\nResults saved: learned/results/libtorch_cpp_benchmark.json\n";
    std::cout << "================================================\n";

    return 0;
}
