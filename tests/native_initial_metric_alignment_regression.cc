#include "InitialMetricAlignment.h"
#include <Eigen/Geometry>
#include <iostream>
#include <limits>
int main() {
    using namespace ORB_SLAM3;
    using Pair=std::pair<Eigen::Vector3f,Eigen::Vector3f>;
    const Eigen::Matrix3f R=Eigen::AngleAxisf(.6f,Eigen::Vector3f(1,2,3).normalized()).toRotationMatrix();
    const Eigen::Vector3f t(2,-3,.8f);
    std::vector<Pair> pairs;
    for(int i=0;i<9;++i) {
        Eigen::Vector3f p(.6f+.03f*i,-.2f+.02f*i,1.3f+.01f*i);
        pairs.emplace_back(p,4.2f*R*p+t);
    }
    Eigen::Vector3f fitted(99,99,99);
    if(!RefitInitialMetricTranslation(pairs,R,4.2f,fitted) || (fitted-t).norm()>1e-5f)return 1;
    // Reusing an old-scale fit and pivoting about the first view leaves a bias.
    Eigen::Vector3f old;
    if(!RefitInitialMetricTranslation(pairs,R,10.5f,old))return 2;
    const Eigen::Vector3f pivot=old+R*((10.5f-4.2f)*pairs.front().first);
    if((pivot-t).norm()<.5f)return 3;
    // One bad centre cannot move the coordinate-wise median of the correct fit.
    pairs[0].second+=Eigen::Vector3f(10,-20,30);
    if(!RefitInitialMetricTranslation(pairs,R,4.2f,fitted) || (fitted-t).norm()>1e-5f)return 4;
    const Eigen::Vector3f before=fitted;
    if(RefitInitialMetricTranslation(pairs,R,-1,fitted) || fitted!=before)return 5;
    if(RefitInitialMetricTranslation(std::vector<Pair>(pairs.begin(),pairs.begin()+2),R,4.2f,fitted) || fitted!=before)return 7;
    pairs[1].first[0]=std::numeric_limits<float>::quiet_NaN();
    if(RefitInitialMetricTranslation(pairs,R,4.2f,fitted) || fitted!=before)return 6;
    std::cout<<"initial metric translation: scale-consistent, robust, failure preserves state\n";
    return 0;
}
