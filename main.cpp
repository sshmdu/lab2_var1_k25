// Lab 2: Investigation of std::all_of (Variant 1)
// Tkachenko Anastasiia, Group K-25
// Compiler: MSVC (Visual Studio 2022, v17.x), C++20
// Build (Developer Command Prompt for VS 2022):
//   No optimization:      cl /std:c++20 /EHsc /MD /Od main.cpp /Fe:lab_O0.exe
//   Maximum optimization: cl /std:c++20 /EHsc /MD /O2 main.cpp /Fe:lab_O2.exe
// Run:
//   lab_O0.exe > out_O0.txt
//   lab_O2.exe > out_O2.txt
// Data files data_N.txt: N random integers [1..1000] separated by spaces; created on first run

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace std;
using namespace chrono;

volatile bool sink_val = false;

template <class F>
double ms(F&& f, int reps = 3)
{
    double total = 0;
    for (int i = 0; i < reps; ++i)
    {
        auto t0 = steady_clock::now();
        sink_val = f();
        total += duration<double, milli>(steady_clock::now() - t0).count();
    }
    return total / reps;
}

vector<int> load_or_generate(size_t n)
{
    string name = std::format("data_{}.txt", n);
    vector<int> v(n);
    ifstream check(name);
    bool exists = check.good();
    check.close();
    if (!exists)
    {
        mt19937 gen(12345);
        uniform_int_distribution<int> d(1, 1000);
        ofstream out(name);
        for (auto& x : v)
        {
            x = d(gen);
            out << x << ' ';
        }
    }
    else
    {
        ifstream in(name);
        for (auto& x : v)
            in >> x;
    }
    return v;
}

template <class T, class Pred>
bool my_all_of(const T* data, size_t len, Pred p, unsigned K)
{
    size_t block = len / K;
    vector<char> res(K, 1);
    {
        vector<jthread> threads;
        threads.reserve(K);
        for (unsigned i = 0; i < K; ++i)
        {
            size_t offset = i * block;
            size_t count = (i == K - 1) ? (len - offset) : block;
            const T* sub_data = data + offset;
            threads.emplace_back([sub_data, count, &p, &res, i]
                { res[i] = std::all_of(sub_data, sub_data + count, p); });
        }
    }
    return std::all_of(res.begin(), res.end(), [](char c) { return c != 0; });
}

struct BestResult
{
    vector<double> time;
    vector<unsigned> K;
};

template <class Pred>
void warm_up(Pred p, const vector<int>& v, unsigned hw)
{
    sink_val = std::all_of(execution::par, v.begin(), v.end(), p);
    sink_val = std::all_of(execution::par_unseq, v.begin(), v.end(), p);
    sink_val = my_all_of(v.data(), v.size(), p, hw);
}

template <class Pred>
void print_library_table(Pred p, const vector<vector<int>>& data)
{
    cout << "\n[Standard all_of, time in ms]\n";
    cout << std::format("{:<12}{:<14}{:<14}{:<14}{:<14}{:<14}\n",
        "N", "no policy", "seq", "par", "unseq", "par_unseq");
    for (auto& v : data)
    {
        cout << std::format("{:<12}{:<14.3f}{:<14.3f}{:<14.3f}{:<14.3f}{:<14.3f}\n",
            v.size(),
            ms([&] { return std::all_of(v.begin(), v.end(), p); }),
            ms([&] { return std::all_of(execution::seq, v.begin(), v.end(), p); }),
            ms([&] { return std::all_of(execution::par, v.begin(), v.end(), p); }),
            ms([&] { return std::all_of(execution::unseq, v.begin(), v.end(), p); }),
            ms([&] { return std::all_of(execution::par_unseq, v.begin(), v.end(), p); }));
    }
}

vector<unsigned> make_thread_counts(unsigned hw)
{
    vector<unsigned> Ks;
    for (unsigned k = 1; k <= 2 * hw; ++k) Ks.push_back(k);
    for (unsigned m : {3u, 4u, 8u}) Ks.push_back(hw * m);
    sort(Ks.begin(), Ks.end());
    Ks.erase(unique(Ks.begin(), Ks.end()), Ks.end());
    return Ks;
}

template <class Pred>
BestResult print_custom_table(Pred p, const vector<vector<int>>& data, unsigned hw)
{
    cout << std::format("\n[Custom all_of, time in ms depending on K; hardware_concurrency = {}]\n", hw);
    string header = std::format("{:<8}", "K");
    for (auto& v : data)
        header += std::format("{:<14}", v.size());
    cout << header << "\n";

    BestResult best{ vector<double>(data.size(), 1e18), vector<unsigned>(data.size(), 0) };
    for (unsigned K : make_thread_counts(hw))
    {
        string row = std::format("{:<8}", K);
        for (size_t i = 0; i < data.size(); ++i)
        {
            auto& v = data[i];
            int reps = (v.size() >= 10000000ULL) ? 2 : 3;
            double t = ms([&] { return my_all_of(v.data(), v.size(), p, K); }, reps);
            row += std::format("{:<14.3f}", t);
            if (t < best.time[i])
            {
                best.time[i] = t;
                best.K[i] = K;
            }
        }
        cout << row << "\n";
    }
    return best;
}

void print_best_k(const BestResult& best, const vector<vector<int>>& data, unsigned hw)
{
    cout << "\n[Optimal K values]\n";
    for (size_t i = 0; i < data.size(); ++i)
    {
        cout << std::format("N={}: best K={} ({:.3f} ms), ratio K/hw = {:.2f}\n",
            data[i].size(), best.K[i], best.time[i], double(best.K[i]) / hw);
    }
}

template <class Pred>
void experiment(const string& pname, Pred p, const vector<vector<int>>& data)
{
    unsigned hw = thread::hardware_concurrency();
    cout << std::format("\n=============== Predicate: {} ===============\n", pname);
    warm_up(p, data[0], hw);
    print_library_table(p, data);
    BestResult best = print_custom_table(p, data, hw);
    print_best_k(best, data, hw);
}

int main()
{
    ios_base::sync_with_stdio(false);
    cout << std::format("hardware_concurrency (hardware threads count): {}\n",
        thread::hardware_concurrency());

    vector<vector<int>> data;
    for (size_t n : {100000ULL, 500000ULL, 1000000ULL, 5000000ULL})
        data.push_back(load_or_generate(n));

    experiment("x > 0 (light)", [](int x) { return x > 0; }, data);
    experiment("sin^2 + cos^2 > 0.5 (heavy)",
        [](int x)
        {
            double d = static_cast<double>(x);
            return sin(d) * sin(d) + cos(d) * cos(d) > 0.5;
        },
        data);
    if (sink_val) cout << "\n";
    return 0;
}
