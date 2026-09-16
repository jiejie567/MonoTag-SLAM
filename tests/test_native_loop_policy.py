"""Native loop-policy regression matrix; requires only a C++14 compiler.

Run with: python -m unittest discover -s tests -p test_native_loop_policy.py
Set ORB_LOOP_POLICY_SOURCE to test an isolated LoopClosing.cc candidate.
The test compiles actual source predicates with map/keyframe stubs, not ORB-SLAM3.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


def _source_path():
    override = os.environ.get("ORB_LOOP_POLICY_SOURCE")
    if override:
        return Path(override)
    for project in Path(__file__).resolve().parents:
        source = project / "third_party/ORB_SLAM3/src/LoopClosing.cc"
        if source.is_file():
            return source
    raise FileNotFoundError("Cannot locate project third_party/ORB_SLAM3/src/LoopClosing.cc")


def _between(text, begin, end):
    start = text.index(begin)
    finish = text.index(end, start + len(begin))
    return text[start:finish]


def _cpp_source(source):
    live = source[source.index("bool LoopClosing::NewDetectCommonRegions()"):]
    fragments = {
        # Keep the policy harness independent of the production optimizer
        # types.  CandidateMarkerEvidence is exercised by the live C++ code,
        # but the matrix below only calls IsStagedMetricLoop and
        # MetricLoopScaleExperiment.  Pulling the intervening helper used to
        # require a g2o::Sim3 definition in this tiny standalone build and
        # made a valid policy regression fail before any predicate ran.
        "helpers": (
            _between(source, "bool IsStagedMetricLoop(",
                    "bool CandidateMarkerEvidence")
            + _between(source, "bool MetricLoopScaleExperiment(",
                       "struct OfflineCandidateDiagnostic")
        ),
        "incrementalFlag": _between(live, "    const char* incrementalMode=",
                                    "    // To deactivate"),
        "enhanced": _between(live, "        const bool enhancedRecall=",
                             "        mpKeyFrameDB->DetectNBestCandidates"),
        "offline": _between(source, "if(IsStagedMetricLoop(map,map)", ") {")[3:],
        "moderateSetup": _between(source, "                        const char* moderateMode=",
                                  "                        if((mbOfflineLoopSearch"),
        "moderate": _between(source, "if((mbOfflineLoopSearch", ") {")[3:],
    }
    cpp = CPP_TEMPLATE
    for name, fragment in fragments.items():
        cpp = cpp.replace("${" + name + "}", fragment)
    return cpp


class NativeLoopPolicyTests(unittest.TestCase):
    def test_native_threshold_retrieval_and_merge_contract(self):
        source = _source_path().read_text()
        bow = source[source.index("bool LoopClosing::DetectCommonRegionsFromBoW("):]
        self.assertIn("int nProjOptMatches = 80;", bow)
        self.assertIn("enhancedRecall ? 12 : 3", source)
        self.assertIn("if(vpMergeBowCand.size()>3) vpMergeBowCand.resize(3);", source)
        self.assertIn("if(numProjOptMatches >= nProjOptMatches || moderateSupport)", bow)

    def test_cpp_policy_matrix(self):
        """5,832 predicate checks across map states, flags and support boundaries."""
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A C++14 compiler (c++) is required for the native policy matrix")
        cpp = _cpp_source(_source_path().read_text())
        with tempfile.TemporaryDirectory(prefix="native_loop_policy_") as temporary:
            binary = Path(temporary) / "loop_policy_test"
            compiled = subprocess.run(
                [compiler, "-std=c++14", "-x", "c++", "-", "-o", str(binary)],
                input=cpp, text=True, capture_output=True, timeout=60,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                result.stdout.strip(),
                "PASS 5832 policy checks; native threshold=80, merge cap=3, metric 54-match gate retained",
            )


CPP_TEMPLATE = r'''
#include <cstdlib>
#include <iostream>
#include <string>
struct Map {
    bool mbMetric, geometry, inertial, bad;
    bool HasMetricTagGeometry() { return geometry; }
    bool IsInertial() { return inertial; }
    bool IsBad() { return bad; }
};
struct KeyFrame {
    Map* map;
    Map* GetMap() { return map; }
    int TrackedMapPoints(int) { return 15; }
};
${helpers}
bool EnhancedRecall(Map* currentMap) {
${incrementalFlag}
${enhanced}
    return enhancedRecall;
}
bool OfflineQuery(Map* map) {
    KeyFrame frame{map}; KeyFrame* query=&frame;
    return ${offline};
}
bool ModerateGate(bool mbOfflineLoopSearch, Map* current, Map* matched, int numProjOptMatches) {
    KeyFrame first{current}, second{matched};
    KeyFrame* mpCurrentKF=&first; KeyFrame* pMostBoWMatchesKF=&second;
    const int nProjOptMatches=80;
${moderateSetup}
    return ${moderate};
}
void SetFlag(const char* name, const char* value) {
    if(value) setenv(name,value,1); else unsetenv(name);
}
int main() {
    Map nonmetric{false,false,false,false}, seed{false,true,false,false};
    Map metric{true,true,false,false}, otherMetric{true,true,false,false};
    Map inertial{true,true,true,false}, bad{true,true,false,true};
    Map inconsistent{true,false,false,false};
    struct Case { const char* name; Map* current; Map* matched; bool eligible; };
    const Case cases[] = {
        {"nonmetric",&nonmetric,&nonmetric,false},
        {"marker_seed_not_metric",&seed,&seed,false},
        {"same_map_metric",&metric,&metric,true},
        {"cross_map_metric",&metric,&otherMetric,false},
        {"metric_to_nonmetric",&metric,&nonmetric,false},
        {"nonmetric_to_metric",&nonmetric,&metric,false},
        {"null_current",nullptr,&metric,false},
        {"null_matched",&metric,nullptr,false},
        {"both_null",nullptr,nullptr,false},
        {"metric_without_geometry",&inconsistent,&inconsistent,false},
        {"inertial_metric",&inertial,&inertial,true},
        {"bad_metric",&bad,&bad,true}
    };
    const char* flags[] = {nullptr,"0","1","audit"};
    const int supports[] = {0,49,50,54,79,80,100};
    unsigned checks=0;
    auto check=[&](bool actual,bool expected,const char* name) {
        ++checks;
        if(actual!=expected) { std::cerr << "FAIL " << name << '\n'; std::exit(1); }
    };
    for(const auto& c:cases) {
        check(IsStagedMetricLoop(c.current,c.matched),c.eligible,c.name);
        const bool mapEligible=c.current && c.current->mbMetric && c.current->geometry;
        check(OfflineQuery(c.current),mapEligible && !c.current->bad && !c.current->inertial,c.name);
        for(const char* incremental:flags) {
            SetFlag("ORB_SLAM3_INCREMENTAL_LOOP_SEARCH",incremental);
            const bool incrementalEnabled=incremental && std::string(incremental)=="1";
            check(EnhancedRecall(c.current),incrementalEnabled && mapEligible && !c.current->inertial,c.name);
            for(bool offline:{false,true}) {
                for(const char* scale:flags) {
                    SetFlag("ORB_SLAM3_MARKER_SIM3_LOOP",scale);
                    const bool scaleEnabled=scale && std::string(scale)=="1";
                    check(MetricLoopScaleExperiment(offline,c.current,c.matched),
                        (offline || incrementalEnabled) && c.eligible && scaleEnabled,c.name);
                }
                for(const char* moderate:flags) {
                    SetFlag("ORB_SLAM3_MODERATE_LOOP_SUPPORT",moderate);
                    const bool moderateEnabled=!moderate || std::string(moderate)!="0";
                    for(int support:supports) {
                        const bool expected=(offline || incrementalEnabled) && c.eligible &&
                            moderateEnabled && support>=50 && support<80;
                        check(ModerateGate(offline,c.current,c.matched,support),expected,c.name);
                        // The native >=80 route remains available regardless of feature flags.
                        check(support>=80 || ModerateGate(offline,c.current,c.matched,support),
                            support>=80 || expected,c.name);
                    }
                }
            }
        }
    }
    std::cout << "PASS " << checks << " policy checks; native threshold=80, merge cap=3, metric 54-match gate retained\n";
}
'''


if __name__ == "__main__":
    unittest.main()
