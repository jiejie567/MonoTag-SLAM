// Standalone DUtils regression; run separately with "portable" and "legacy".
#ifdef NDEBUG
#undef NDEBUG // Regression checks must execute in Release builds too.
#endif
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "DUtils/Random.h"

namespace {

using DUtils::Random;

const int seed0[] = {
    520932930, 28925691, 822784415, 890459872, 145532761, 2132723841,
    1040043610, 1643550337, 68362598, 66433441, 2002830094, 1906706780};
const int seed1[] = {
    16807, 282475249, 1622650073, 984943658, 1144108930, 470211272,
    101027544, 1457850878, 1458777923, 2007237709, 823564440, 1115438165};
const int seed42[] = {
    705894, 1126542223, 1579310009, 565444343, 807934826, 421520601,
    2095673201, 1100194760, 1139130650, 552121545, 229968128, 1751246343};

int portableDraw()
{
    // Exact power-of-two mapping exposes each draw through the public API.
    return Random::RandomInt(0, INT_MAX);
}

void checkVector(int seed, const int* expected, int count)
{
    Random::SeedRand(seed);
    for(int i = 0; i < count; ++i) assert(portableDraw() == expected[i]);
}

void portableRegression()
{
    static_assert(INT_MAX == 2147483647, "This native target uses 32-bit int");
    assert(portableDraw() == seed0[0]); // Offline bootstrap seed, before seeding.
    Random::SeedRand(42);              // Explicit seeding does not mark Once.
    Random::SeedRandOnce(0);
    assert(portableDraw() == seed0[0]);
    Random::SeedRandOnce(42);
    assert(portableDraw() == seed0[1]);
    Random::SeedRandOnce();
    assert(portableDraw() == seed0[2]);

    checkVector(0, seed0, 12);
    checkVector(1, seed1, 12);
    checkVector(42, seed42, 12);
    checkVector(-1, seed1, 12);       // uint32 seed 0xffffffff.
    checkVector(INT_MIN, seed1, 12); // uint32 seed 0x80000000.
    Random::SeedRand(INT_MAX);
    assert(portableDraw() == 0);
    assert(portableDraw() == seed0[0]); // Zero state replacement is per draw.

    Random::SeedRand(0);
    std::srand(9876);
    (void)std::rand();
    assert(portableDraw() == seed0[0]); // No libc state sharing/interposition.
    assert(Random::RandomValue<double>() == double(seed0[1]) / INT_MAX);
    assert(Random::RandomValue<float>() == float(seed0[2]) / float(INT_MAX));
    assert(portableDraw() == seed0[3]); // All DUtils APIs share this stream.

    for(int repeat = 0; repeat < 2; ++repeat)
    {
        Random::SeedRand(0);
        const int low[] = {0, -100, 30, -7, 0, -27, 20, 0, -50, 5, 1, -9};
        const int high[] = {2000, 100, 500, -7, 299, 27, 4000, 17, 50, 20, 6, 9};
        for(int i = 0; i < 12; ++i)
        {
            const int expected = int((double(seed0[i]) / 2147483648.0) *
                                     (high[i] - low[i] + 1)) + low[i];
            assert(Random::RandomInt(low[i], high[i]) == expected);
        }
    }

#ifdef __APPLE__
    // On the reference OS, compare real libc including floating arithmetic.
    Random::SeedRand(0);
    std::srand(0);
    for(int i = 0; i < 10000; ++i)
    {
        const int low = -(i % 19), high = 10 + i % 1999;
        const int expected = int((double(std::rand()) / (double(RAND_MAX) + 1)) *
                                 (high - low + 1)) + low;
        assert(Random::RandomInt(low, high) == expected);
        assert(Random::RandomValue<float>() == float(std::rand()) / float(RAND_MAX));
        assert(Random::RandomValue<double>() == double(std::rand()) / double(RAND_MAX));
    }
#endif

    Random::SeedRand(0);
    for(int i = 0; i < 10000; ++i)
    {
        (void)Random::RandomInt(INT_MIN, INT_MAX);
        assert(Random::RandomInt(INT_MIN, INT_MIN) == INT_MIN);
        assert(Random::RandomInt(INT_MAX, INT_MAX) == INT_MAX);
        assert(Random::RandomInt(INT_MIN, INT_MIN + 10) <= INT_MIN + 10);
        assert(Random::RandomInt(INT_MAX - 10, INT_MAX) >= INT_MAX - 10);
    }
    bool rejected = false;
    try { (void)Random::RandomInt(2, 1); }
    catch(const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    // The shared stream is serialized, not assigned deterministically to threads.
    const int threads = 4, draws = 10000;
    std::vector<int> sequential(threads * draws), concurrent(threads * draws);
    Random::SeedRand(42);
    for(int& value : sequential) value = portableDraw();
    Random::SeedRand(42);
    std::vector<std::thread> workers;
    for(int t = 0; t < threads; ++t)
        workers.emplace_back([t, &concurrent]() {
            for(int i = 0; i < draws; ++i) concurrent[t * draws + i] = portableDraw();
        });
    for(auto& worker : workers) worker.join();
    std::sort(sequential.begin(), sequential.end());
    std::sort(concurrent.begin(), concurrent.end());
    assert(sequential == concurrent);

    // Mode is fixed at first use; environment changes do not split RNG state.
    setenv("ORB_SLAM3_PORTABLE_RANDOM", "0", 1);
    checkVector(0, seed0, 12);
}

void legacyRegression()
{
    struct Values { int integer; float realFloat; double realDouble; };
    std::vector<Values> expected;
    std::srand(42);
    for(int i = 0; i < 10000; ++i)
    {
        Values values;
        values.integer = int((double(std::rand()) / (double(RAND_MAX) + 1)) * 2101) - 100;
        values.realFloat = float(std::rand()) / float(RAND_MAX);
        values.realDouble = double(std::rand()) / double(RAND_MAX);
        expected.push_back(values);
    }
    Random::SeedRand(42);
    for(const auto& values : expected)
    {
        assert(Random::RandomInt(-100, 2000) == values.integer);
        assert(Random::RandomValue<float>() == values.realFloat);
        assert(Random::RandomValue<double>() == values.realDouble);
    }
    // Legacy still shares libc state and keeps its seed-once behavior.
    std::srand(0);
    const double first = double(std::rand()) / double(RAND_MAX);
    const double second = double(std::rand()) / double(RAND_MAX);
    Random::SeedRandOnce(0);
    assert(Random::RandomValue<double>() == first);
    Random::SeedRandOnce(42);
    Random::SeedRandOnce();
    assert(Random::RandomValue<double>() == second);
    setenv("ORB_SLAM3_PORTABLE_RANDOM", "1", 1);
    std::srand(0);
    assert(Random::RandomValue<double>() == first);
}

}

int main(int argc, char** argv)
{
    const bool legacy = argc > 1 && std::string(argv[1]) == "legacy";
    setenv("ORB_SLAM3_PORTABLE_RANDOM", legacy ? "0" : "1", 1);
    if(legacy) legacyRegression();
    else portableRegression();
    std::cout << "native portable random regression: " << (legacy ? "legacy" : "portable")
              << " passed\n";
}
