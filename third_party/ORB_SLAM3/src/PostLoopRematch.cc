#include "PostLoopRematch.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBmatcher.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>

namespace ORB_SLAM3 {
namespace PostLoopRematch {
namespace {
bool Supports(MapPoint* point, const Eigen::Vector3f& position) {
    unsigned int count=0;
    for(const auto& observation:point->GetObservations()) {
        KeyFrame* kf=observation.first;
        if(!kf || kf->isBad() || kf->GetMap()!=point->GetMap()) continue;
        const int index=std::get<0>(observation.second);
        if(kf->mpCamera2 || index<0 || index>=int(kf->mvKeysUn.size())) return false;
        const auto camera=kf->GetPose()*position;
        if(!camera.allFinite() || camera.z()<=0) return false;
        const auto pixel=kf->mpCamera->project(camera);
        const auto& measured=kf->mvKeysUn[index];
        const Eigen::Vector2f error=pixel-Eigen::Vector2f(measured.pt.x,measured.pt.y);
        if(!pixel.allFinite() || measured.octave<0 ||
           measured.octave>=int(kf->mvInvLevelSigma2.size()) ||
           error.squaredNorm()*kf->mvInvLevelSigma2[measured.octave]>5.991f) return false;
        ++count;
    }
    return count>=2;
}
int Cell(KeyFrame* kf, int index) {
    const auto& p=kf->mvKeysUn[index].pt;
    const int x=std::max(0,std::min(3,int(4*(p.x-kf->mnMinX)/(kf->mnMaxX-kf->mnMinX))));
    const int y=std::max(0,std::min(3,int(4*(p.y-kf->mnMinY)/(kf->mnMaxY-kf->mnMinY))));
    return y*4+x;
}
}

bool ConsistentDuplicate(MapPoint* first, MapPoint* second) {
    if(!first || !second || first==second || first->isBad() || second->isBad() ||
       first->GetMap()!=second->GetMap()) return false;
    // Two distinct features already observed in the same image are not a
    // duplicate hypothesis to be resolved by proximity alone.
    const auto a=first->GetObservations(),b=second->GetObservations();
    for(const auto& observation:a) if(b.count(observation.first)) return false;
    const auto x=first->GetWorldPos(),y=second->GetWorldPos();
    return x.allFinite() && y.allFinite() && Supports(first,x) && Supports(first,y) &&
           Supports(second,x) && Supports(second,y);
}

void Run(Map* map, KeyFrame* first, KeyFrame* last, bool apply) {
    if(!map || map->IsBad() || map->IsInertial() || !first || !last) return;
    const auto start=std::chrono::steady_clock::now();
    const double begin=std::min(first->mTimeStamp,last->mTimeStamp);
    const double end=std::max(first->mTimeStamp,last->mTimeStamp);
    auto keyframes=map->GetAllKeyFrames();
    keyframes.erase(std::remove_if(keyframes.begin(),keyframes.end(),[&](KeyFrame* kf){
        return !kf || kf->isBad() || kf->mpCamera2 || kf->mTimeStamp<begin || kf->mTimeStamp>end;
    }),keyframes.end());
    std::sort(keyframes.begin(),keyframes.end(),KeyFrame::lId);
    ORBmatcher matcher(.75,true);
    std::set<std::pair<unsigned long,unsigned long>> visited;
    std::set<MapPoint*> consumed;
    std::set<KeyFrame*> changed;
    unsigned int pairs=0,mutual=0,consistent=0,qualified=0,fused=0;
    for(KeyFrame* query:keyframes) {
        const auto center=query->GetCameraCenter();
        const Eigen::Vector3f forward=query->GetRotation().transpose()*Eigen::Vector3f::UnitZ();
        const float depth=query->ComputeSceneMedianDepth(2);
        if(!std::isfinite(depth) || depth<=0) continue;
        std::vector<std::pair<float,KeyFrame*>> neighbors;
        for(KeyFrame* candidate:keyframes) {
            if(candidate==query || std::abs(query->mTimeStamp-candidate->mTimeStamp)<3 ||
               query->GetWeight(candidate)>=80) continue;
            const Eigen::Vector3f otherForward=candidate->GetRotation().transpose()*Eigen::Vector3f::UnitZ();
            const float distance=(center-candidate->GetCameraCenter()).norm();
            if(forward.dot(otherForward)<.5f || distance>.5f*depth) continue;
            neighbors.emplace_back(distance,candidate);
        }
        std::sort(neighbors.begin(),neighbors.end(),[](const auto& a,const auto& b){
            return a.first==b.first?a.second->mnId<b.second->mnId:a.first<b.first;
        });
        if(neighbors.size()>6) neighbors.resize(6);
        for(const auto& neighbor:neighbors) {
            KeyFrame* candidate=neighbor.second;
            auto ids=std::minmax(query->mnId,candidate->mnId);
            if(!visited.emplace(ids.first,ids.second).second) continue;
            ++pairs;
            auto a=query->GetMapPointMatches(),b=candidate->GetMapPointMatches();
            auto valid=[](MapPoint* p){return p && !p->isBad();};
            std::vector<MapPoint*> pointsA,pointsB;
            for(MapPoint* p:a) if(valid(p)) pointsA.push_back(p);
            for(MapPoint* p:b) if(valid(p)) pointsB.push_back(p);
            std::vector<MapPoint*> intoA(a.size(),nullptr),intoB(b.size(),nullptr);
            auto ta=query->GetPose(),tb=candidate->GetPose();
            Sophus::Sim3f sa(ta.unit_quaternion(),ta.translation());
            Sophus::Sim3f sb(tb.unit_quaternion(),tb.translation());
            matcher.SearchByProjection(query,sa,pointsB,intoA,3,1.f);
            matcher.SearchByProjection(candidate,sb,pointsA,intoB,3,1.f);
            std::vector<std::pair<MapPoint*,MapPoint*>> proposals;
            std::set<int> cellsA,cellsB;
            std::set<MapPoint*> uniqueA,uniqueB;
            for(std::size_t i=0;i<a.size();++i) {
                MapPoint* x=a[i]; MapPoint* y=intoA[i];
                if(!valid(x) || !valid(y) || x==y || consumed.count(x) || consumed.count(y)) continue;
                const int j=std::get<0>(y->GetIndexInKeyFrame(candidate));
                if(j<0 || j>=int(intoB.size()) || intoB[j]!=x) continue;
                ++mutual;
                if(!ConsistentDuplicate(x,y) || uniqueA.count(x) || uniqueB.count(y)) continue;
                ++consistent;
                uniqueA.insert(x); uniqueB.insert(y);
                proposals.emplace_back(x,y);
                cellsA.insert(Cell(query,int(i))); cellsB.insert(Cell(candidate,j));
            }
            if(proposals.size()<15 || cellsA.size()<4 || cellsB.size()<4) continue;
            ++qualified;
            for(auto proposal:proposals) {
                MapPoint* keep=proposal.first; MapPoint* drop=proposal.second;
                if(keep->Observations()<drop->Observations()) std::swap(keep,drop);
                consumed.insert(keep); consumed.insert(drop);
                if(apply) {
                    for(const auto& obs:keep->GetObservations()) changed.insert(obs.first);
                    for(const auto& obs:drop->GetObservations()) changed.insert(obs.first);
                    drop->Replace(keep);
                    keep->UpdateNormalAndDepth();
                    ++fused;
                }
            }
        }
    }
    for(KeyFrame* kf:changed) if(kf && !kf->isBad()) kf->UpdateConnections();
    if(fused) map->IncreaseChangeIndex();
    std::cout << "POST_LOOP_REMATCH first=" << first->mnId << " last=" << last->mnId
        << " keyframes=" << keyframes.size() << " pairs=" << pairs << " mutual=" << mutual
        << " consistent=" << consistent << " qualified=" << qualified << " fused=" << fused
        << " apply=" << apply << " elapsed_ms="
        << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()
        << std::endl;
}
}
}
