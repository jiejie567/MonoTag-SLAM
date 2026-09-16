#include "MarkerGraphCoordinator.h"
#include "MarkerGeometryTolerance.h"
#include "Atlas.h"
#include "Tracking.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "MapDrawer.h"
#include "CameraModels/GeometricCamera.h"
#include <opencv2/calib3d.hpp>
#include <Eigen/SVD>
#include <algorithm>
#include <iterator>
#include <limits>
#include <mutex>
#include <utility>

namespace ORB_SLAM3 {
namespace {
template<class T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;
constexpr double kDenseMarkerGapS = .30;
// Candidate freshness only: these do not interpolate poses, join dense scale
// samples, or relax any raw-corner/background/depth/scale commit gate.
constexpr double kCornerRevisitGapS = 1.0;
constexpr double kCornerRevisitSpanS = 3.0;
double Median(std::vector<double> values) {
    if(values.empty()) return 0;
    std::sort(values.begin(),values.end());
    const std::size_t n=values.size();
    return n%2 ? values[n/2] : .5*(values[n/2-1]+values[n/2]);
}
std::set<int> StrongIds(KeyFrame* kf) {
    std::set<int> result;
    if(!kf || kf->isBad() || !kf->mbTagObservationActive ||
       !kf->mbHasTagObservation || kf->mTagObservationConfidence<.35f ||
       kf->mvTagIds.size()!=kf->mvTagWorldPoints.size() ||
       kf->mvTagImagePoints.size()!=kf->mvTagWorldPoints.size()) return result;
    std::map<int,int> counts;
    for(std::size_t i=0;i<kf->mvTagIds.size();++i)
        if(kf->mvTagPointWeights.empty() ||
           (i<kf->mvTagPointWeights.size() && kf->mvTagPointWeights[i]>=.99f))
            ++counts[kf->mvTagIds[i]];
    for(const auto& item:counts) if(item.second==4) result.insert(item.first);
    return result;
}
std::vector<KeyFrame*> StrongKeyframes(Map* map) {
    std::vector<KeyFrame*> result;
    for(KeyFrame* kf:map->GetAllKeyFrames())
        if(kf && kf->GetMap()==map && !StrongIds(kf).empty()) result.push_back(kf);
    std::sort(result.begin(),result.end(),KeyFrame::lId);
    return result;
}
bool ValidateProposal(const MarkerGraphOptimizer::Proposal& p, Map* target, Map* source=nullptr) {
    if(!p.accepted || p.keyframePoses.empty() || !target || target->IsBad()) return false;
    std::set<MapPoint*> destinations;
    for(const auto& pair:p.pointAliases) {
        if(!pair.first || !pair.second || pair.first==pair.second || pair.first->isBad() || pair.second->isBad() ||
           pair.first->GetMap()!=target || pair.second->GetMap()!=target || p.pointAliases.count(pair.second) ||
           !destinations.insert(pair.second).second || !p.pointPositions.count(pair.first) ||
           !p.pointPositions.count(pair.second) ||
           (p.pointPositions.at(pair.first)-p.pointPositions.at(pair.second)).norm()>1e-6) return false;
    }
    for(const auto& item:p.keyframePoses) {
        if(!item.first || item.first->isBad() || !item.second.matrix().allFinite() ||
           !item.first->GetMap() ||
           (item.first->GetMap()!=target && (!source || item.first->GetMap()!=source))) return false;
        auto scale=p.replayScaleMultipliers.find(item.first);
        if(scale!=p.replayScaleMultipliers.end() &&
           (!std::isfinite(scale->second) || scale->second<=0)) return false;
        const float ratio=scale==p.replayScaleMultipliers.end()?1.f:scale->second;
        const float accumulated=item.first->mReplayUnitScale*ratio;
        const auto delta=MarkerGraphTransform::between(item.first->GetPoseInverse(),item.second.inverse(),ratio,1);
        if(!std::isfinite(accumulated) || accumulated<=0 ||
           !(delta*item.first->mReplayMarkerGraph).finite() ||
           !item.first->mReplayMarkerGauge.finite() ||
           std::abs(item.first->mReplayMarkerGauge.scale-1.)>1e-6) return false;
    }
    for(const auto& item:p.replayScaleMultipliers)
        if(!p.keyframePoses.count(item.first)) return false;
    for(const auto& item:p.pointPositions)
        if(!item.first || item.first->isBad() || !item.second.allFinite() || !item.first->GetMap() ||
           (item.first->GetMap()!=target && (!source || item.first->GetMap()!=source))) return false;
    for(const auto& item:p.tagWorldCorners) {
        if(!p.keyframePoses.count(item.first) || item.second.size()!=item.first->mvTagWorldPoints.size()) return false;
        for(const auto& corner:item.second) if(!corner.allFinite()) return false;
    }
    for(const auto& tag:p.staticTags) {
        if(tag.first<0 || tag.second.size()!=4) return false;
        for(const auto& corner:tag.second) if(!corner.allFinite()) return false;
    }
    for(int id:p.optimizedMarkerIds)
        if(!p.staticTags.count(id)) return false;
    return true;
}
void ApplyStaticTags(Map* map,const MarkerGraphOptimizer::Proposal& p) {
    for(const auto& tag:p.staticTags) {
        auto& values=map->mStaticTags[tag.first];
        values.clear(); values.reserve(12);
        for(const auto& corner:tag.second)
            for(int axis=0;axis<3;++axis) values.push_back(corner(axis));
    }
}
void ApplyProposal(const MarkerGraphOptimizer::Proposal& p, unsigned long sequence) {
    // Validation has completed. All writes occur under the same correction
    // gate/map locks, and no proposal contains aliases to mutable estimates.
    for(const auto& item:p.keyframePoses) {
        KeyFrame* kf=item.first;
        const auto scale=p.replayScaleMultipliers.find(kf);
        const float ratio=scale==p.replayScaleMultipliers.end()?1.f:scale->second;
        const auto delta=MarkerGraphTransform::between(kf->GetPoseInverse(),item.second.inverse(),ratio,sequence);
        kf->mReplayMarkerGraph=delta*kf->mReplayMarkerGraph;
        kf->mReplayUnitScale*=ratio;
        kf->SetPose(item.second);
    }
    for(const auto& item:p.tagWorldCorners) item.first->mvTagWorldPoints=item.second;
    for(const auto& item:p.pointPositions) item.first->SetWorldPos(item.second);
    for(const auto& item:p.pointPositions) item.first->UpdateNormalAndDepth();
}
bool RefineGaugeUnchanged(Map* map, const MarkerGraphOptimizer::Proposal& p) {
    KeyFrame* origin=map ? map->GetOriginKF() : nullptr;
    if(!origin || origin->isBad() || origin->GetMap()!=map) return false;
    const auto camera=p.keyframePoses.find(origin);
    if(camera==p.keyframePoses.end()) return false;
    const MarkerGraphOptimizer::Options limits;
    const auto original=origin->GetPose();
    if((camera->second.inverse().translation()-original.inverse().translation()).norm()>
            limits.maximumAnchorTranslationM ||
       (camera->second.so3()*original.so3().inverse()).log().norm()>limits.maximumAnchorRotationRad)
        return false;
    const auto originIds=StrongIds(origin);
    if(std::find(p.excludedTagKeyFrameIds.begin(),p.excludedTagKeyFrameIds.end(),origin->mnId)!=
            p.excludedTagKeyFrameIds.end()) return false;
    if(!map->mbRigidMarkerLayout) {
        // Gauge is the unchanged origin CAMERA, not its noisy marker pose.
        // Check physical size and registry/factor consistency instead of
        // requiring a free landmark to remain near its initial world XYZ.
        for(int id:originIds)
            if(std::find(p.excludedTagGroupIds.begin(),p.excludedTagGroupIds.end(),
                         std::make_pair(origin->mnId,id))!=p.excludedTagGroupIds.end()) return false;
        for(const auto& tag:p.staticTags) {
            const auto old=map->mStaticTags.find(tag.first);
            if(old==map->mStaticTags.end() || old->second.size()!=12 || tag.second.size()!=4) return false;
            for(int a=0;a<4;++a) for(int b=a+1;b<4;++b) {
                const Eigen::Vector3f pa(old->second[3*a],old->second[3*a+1],old->second[3*a+2]);
                const Eigen::Vector3f pb(old->second[3*b],old->second[3*b+1],old->second[3*b+2]);
                if(std::abs((tag.second[a]-tag.second[b]).norm()-(pa-pb).norm())>1e-4f) return false;
            }
        }
        // Use the actual admitted groups, including weak observations. A
        // changed registry alone must not commit with stale validated factors.
        for(const auto& group:p.after.tagRmsByKeyframeMarker) {
            KeyFrame* k=group.first.first; const int id=group.first.second;
            const auto tag=p.staticTags.find(id);
            const auto corners=p.tagWorldCorners.find(k);
            if(tag==p.staticTags.end() || corners==p.tagWorldCorners.end() ||
               k->mvTagIds.size()!=corners->second.size()) return false;
            std::size_t count=0;
            for(std::size_t i=0;i<k->mvTagIds.size();++i) if(k->mvTagIds[i]==id) {
                const auto& point=corners->second[i];
                float distance=std::numeric_limits<float>::infinity();
                for(const auto& canonical:tag->second) distance=std::min(distance,(point-canonical).norm());
                if(distance>1e-4f) return false;
                ++count;
            }
            if(!count) return false;
        }
        return !p.after.tagRmsByKeyframeMarker.empty();
    }
    if(originIds.empty()) return true;
    // The explicitly configured rigid-board mode retains its existing fixed
    // board policy. This is NOT used for independently placed markers.
    const std::set<int> fixedIds(p.optimizedMarkerIds.begin(),p.optimizedMarkerIds.end());
    AlignedVector<Eigen::Vector3d> before, after;
    for(int id:fixedIds) {
        if(std::find(p.excludedTagGroupIds.begin(),p.excludedTagGroupIds.end(),
                     std::make_pair(origin->mnId,id))!=p.excludedTagGroupIds.end()) return false;
        const auto oldTag=map->mStaticTags.find(id);
        const auto newTag=p.staticTags.find(id);
        if(oldTag==map->mStaticTags.end() || oldTag->second.size()!=12 ||
           newTag==p.staticTags.end() || newTag->second.size()!=4) return false;
        for(int corner=0;corner<4;++corner) {
            before.emplace_back(oldTag->second[3*corner],oldTag->second[3*corner+1],oldTag->second[3*corner+2]);
            after.push_back(newTag->second[corner].cast<double>());
            if((after.back()-before.back()).norm()>limits.maximumAnchorTranslationM) return false;
        }
    }
    if(before.size()<4) return false;
    Eigen::Vector3d beforeCentre=Eigen::Vector3d::Zero(),afterCentre=beforeCentre;
    for(std::size_t i=0;i<before.size();++i) { beforeCentre+=before[i]; afterCentre+=after[i]; }
    beforeCentre/=before.size(); afterCentre/=after.size();
    Eigen::Matrix3d covariance=Eigen::Matrix3d::Zero();
    for(std::size_t i=0;i<before.size();++i)
        covariance+=(before[i]-beforeCentre)*(after[i]-afterCentre).transpose();
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance,Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d correction=svd.matrixV()*svd.matrixU().transpose();
    if(correction.determinant()<0) {
        Eigen::Matrix3d adjusted=svd.matrixV(); adjusted.col(2)*=-1;
        correction=adjusted*svd.matrixU().transpose();
    }
    return correction.allFinite() &&
        Eigen::AngleAxisd(correction).angle()<=limits.maximumAnchorRotationRad;
}
void FillResiduals(MarkerGraphEvent& event, const MarkerGraphOptimizer::Proposal& p) {
    event.beforeTagRms=p.before.tagRmsPx; event.afterTagRms=p.after.tagRmsPx;
    event.beforeBackgroundRms=p.before.backgroundRmsPx;
    event.afterBackgroundRms=p.after.backgroundRmsPx;
    event.affectedKeyframes=p.affectedKeyFrameIds;
    event.excludedTagKeyframes=p.excludedTagKeyFrameIds;
    event.excludedTagGroups=p.excludedTagGroupIds;
    for(const auto& marker:p.after.tagRmsByMarker) {
        event.diagnosticMarkerIds.push_back(marker.first);
        const auto before=p.before.tagRmsByMarker.find(marker.first);
        event.beforeMarkerRms.push_back(before==p.before.tagRmsByMarker.end()?0.:before->second);
        event.afterMarkerRms.push_back(marker.second);
    }
    for(const auto& residual:p.after.tagRmsByKeyframeMarker) {
        if(residual.second<=event.worstTagRms) continue;
        event.worstTagRms=residual.second;
        event.worstTagKeyframeId=long(residual.first.first->mnId);
        event.worstTagMarkerId=residual.first.second;
    }
}
} // namespace

MarkerGraphCoordinator::ScaleEvidence MarkerGraphCoordinator::EstimateScale(
    const ScaleSampleVector& input)
{
    ScaleEvidence e;
    std::vector<const ScaleSample*> samples;
    std::set<unsigned long> frames;
    for(const auto& sample:input)
        if(sample.visualTwc.matrix().allFinite() && sample.markerTwc.matrix().allFinite() &&
           std::isfinite(sample.timestamp) && frames.insert(sample.frameId).second) samples.push_back(&sample);
    e.observations=samples.size();
    if(samples.size()<8) { e.reason="need_eight_independent_marker_observations"; return e; }
    for(const auto* sample:samples)
        if(sample->correctionEpoch!=samples.front()->correctionEpoch) {
            e.reason="mixed_map_correction_epochs"; return e;
        }
    // The many correlated pairs do not count as independent measurements.
    // Estimate conservatively from their dispersion, not dispersion/sqrt(Npairs).
    std::vector<double> logRatios;
    for(std::size_t i=0;i<samples.size();++i) for(std::size_t j=i+1;j<samples.size();++j) {
        const double metric=(samples[i]->markerTwc.translation()-samples[j]->markerTwc.translation()).norm();
        const double visual=(samples[i]->visualTwc.translation()-samples[j]->visualTwc.translation()).norm();
        e.baselineM=std::max(e.baselineM,metric);
        if(metric>=.02 && visual>.001 && std::abs(samples[i]->timestamp-samples[j]->timestamp)>=.05)
            logRatios.push_back(std::log(metric/visual));
    }
    if(e.baselineM<.04 || logRatios.size()<6) { e.reason="insufficient_translation_baseline"; return e; }
    const double centre=Median(logRatios);
    std::vector<double> deviations;
    for(double value:logRatios) deviations.push_back(std::abs(value-centre));
    e.sigma=std::max(.005,1.4826*Median(deviations));
    e.metricPerVisual=std::exp(centre);
    // Pair ratios are correlated: two incompatible pose clusters can create
    // a majority of identical ratios and an erroneously zero MAD. Require a
    // single rigid orientation + scale to explain the independent frames as
    // well. Keep the spatial-fit uncertainty, not its standard error over a
    // fictitious N*(N-1)/2 independent samples.
    Eigen::Vector4d quaternionSum=Eigen::Vector4d::Zero();
    Eigen::Quaterniond firstRotation;
    Eigen::Vector3d visualCentre=Eigen::Vector3d::Zero(),metricCentre=visualCentre;
    for(std::size_t i=0;i<samples.size();++i) {
        Eigen::Quaterniond rotation=samples[i]->markerTwc.unit_quaternion().cast<double>()*
            samples[i]->visualTwc.unit_quaternion().cast<double>().conjugate();
        if(i==0) firstRotation=rotation;
        if(rotation.dot(firstRotation)<0) rotation.coeffs()*=-1;
        quaternionSum+=rotation.coeffs();
        visualCentre+=samples[i]->visualTwc.translation().cast<double>();
        metricCentre+=samples[i]->markerTwc.translation().cast<double>();
    }
    const Eigen::Quaterniond rotation(quaternionSum.normalized());
    visualCentre/=samples.size(); metricCentre/=samples.size();
    double spatialSquared=0;
    for(const auto* sample:samples) {
        const Eigen::Vector3d residual=sample->markerTwc.translation().cast<double>()-metricCentre-
            e.metricPerVisual*(rotation*(sample->visualTwc.translation().cast<double>()-visualCentre));
        spatialSquared+=residual.squaredNorm();
    }
    e.sigma=std::max(e.sigma,std::sqrt(spatialSquared/samples.size())/e.baselineM);
    if(e.metricPerVisual<.5 || e.metricPerVisual>2 || e.sigma>.10) {
        e.reason="inconsistent_or_extreme_scale_observations"; return e;
    }
    e.geometricallyValid=true;
    if(std::abs(e.metricPerVisual-1.0)<=.03 || std::abs(centre)<=3*e.sigma) {
        e.reason="scale_difference_not_significant"; return e;
    }
    e.reliable=true; e.reason="reliable_scale_drift";
    return e;
}

MarkerGraphCoordinator::ScaleEvidence MarkerGraphCoordinator::EstimateScale(
    const std::vector<ScaleSample>& input)
{
    ScaleSampleVector aligned(input.begin(),input.end());
    return EstimateScale(aligned);
}

bool MarkerGraphCoordinator::ShouldScheduleScale(const ScaleEvidence& evidence, bool intervalClosure)
{
    // Significant drift always deserves correction. A new static anchor or
    // a genuine revisit of the current anchor closes a metric interval: even
    // when the ratio is near one, raw corners should jointly check/refine the
    // A->B keyframe/point chain once rather than forcing a visible rescale.
    return evidence.reliable || (intervalClosure && evidence.geometricallyValid);
}

bool MarkerGraphCoordinator::CanTryCornerInterval(const ScaleEvidence& evidence,
                                                 bool intervalClosure, std::size_t keyframes)
{
    // This admits a solve, not a scale measurement or a committed correction.
    // Registered strong corner factors must still pass the full joint BA gates.
    return intervalClosure && keyframes>=3 && evidence.observations>=8 &&
        std::isfinite(evidence.baselineM) && evidence.baselineM>=.04 &&
        evidence.reason=="inconsistent_or_extreme_scale_observations";
}

bool MarkerGraphCoordinator::CanRetryScale(std::size_t previousViews,
        std::size_t currentViews, unsigned attempts, double elapsedSeconds)
{
    // New evidence, not the same rejected graph in an expensive polling loop.
    return attempts<3 && currentViews>=previousViews+2 &&
        std::isfinite(elapsedSeconds) && elapsedSeconds>=.5;
}

void MarkerGraphCoordinator::CornerRevisitWindow::Observe(Map* currentMap,
        unsigned long frame, double timestamp, const std::set<int>& strongIds)
{
    std::set<int> registered;
    if(currentMap && currentMap->mbMetric && !currentMap->IsBad())
        for(int id:strongIds) if(currentMap->mStaticTags.count(id)) registered.insert(id);
    if(registered.empty() || !std::isfinite(timestamp)) {
        *this=CornerRevisitWindow(); return;
    }
    const int epoch=currentMap->GetLastBigChangeIdx();
    std::set<int> common;
    std::set_intersection(markerIds.begin(),markerIds.end(),registered.begin(),registered.end(),
                          std::inserter(common,common.end()));
    const double gap=timestamp-lastTime;
    if(map!=currentMap || correctionEpoch!=epoch || common.empty() || frame<lastFrame ||
       gap<0 || gap>kCornerRevisitGapS+1e-9 || timestamp-firstTime>kCornerRevisitSpanS+1e-9) {
        map=currentMap; correctionEpoch=epoch; firstFrame=frame; firstTime=timestamp;
        fragmented=false; markerIds=registered;
    } else {
        markerIds=common;
        fragmented=fragmented || gap>kDenseMarkerGapS;
    }
    lastFrame=frame; lastTime=timestamp;
}

std::vector<unsigned long> MarkerGraphCoordinator::SelectCornerRevisit(
        const ScaleEvidence& evidence, const CornerRevisitWindow& window,
        KeyFrame* anchor, const std::vector<KeyFrame*>& strongKeyframes)
{
    // Eight CURRENT continuous observations and their >=4 cm marker motion
    // admit a trial, not a cross-gap numeric scale measurement. The trial
    // receives a weak unit prior; its saved raw corners must still validate.
    if(!window.fragmented || !window.map || window.map->IsBad() ||
       window.correctionEpoch!=window.map->GetLastBigChangeIdx() ||
       !anchor || anchor->GetMap()!=window.map || StrongIds(anchor).empty() ||
       evidence.observations<8 || !std::isfinite(evidence.baselineM) || evidence.baselineM<.04 ||
       (evidence.reason!="reliable_scale_drift" &&
        evidence.reason!="scale_difference_not_significant" &&
        evidence.reason!="inconsistent_or_extreme_scale_observations")) return {};
    const auto anchorIds=StrongIds(anchor);
    for(int id:window.markerIds) {
        if(!anchorIds.count(id)) continue;
        std::vector<unsigned long> result;
        for(KeyFrame* k:strongKeyframes)
            if(k && k!=anchor && k->GetMap()==window.map &&
               k->mnFrameId>=window.firstFrame && k->mnFrameId<=window.lastFrame &&
               k->mTimeStamp>=window.firstTime && k->mTimeStamp<=window.lastTime &&
               StrongIds(k).count(id)) result.push_back(k->mnId);
        std::sort(result.begin(),result.end());
        result.erase(std::unique(result.begin(),result.end()),result.end());
        if(result.size()>=2) return result;
    }
    return {};
}

bool MarkerGraphCoordinator::CommitScale(Atlas& atlas, Map* map,
    const MarkerGraphOptimizer::Proposal& p, KeyFrame* nextAnchor, MarkerGraphEvent& event)
{
    if(!ValidateProposal(p,map) || !map->mbMetric || !nextAnchor || nextAnchor->isBad() ||
       nextAnchor->GetMap()!=map || !p.keyframePoses.count(nextAnchor)) return false;
    event.sequence=++atlas.mnMarkerGraphSequence;
    ApplyProposal(p,event.sequence);
    ApplyStaticTags(map,p);
    map->SetMarkerScaleAnchorKFId(long(nextAnchor->mnId));
    map->mnMarkerGraphSequence=event.sequence;
    map->InformNewBigChange(); map->IncreaseChangeIndex();
    event.type="scale_reanchor"; event.status="accepted"; event.reason="joint_graph_and_corner_ba_validated";
    event.mapId=event.targetMapId=long(map->GetId()); event.revision=map->mnRevision;
    FillResiduals(event,p); atlas.mMarkerGraphEvents.push_back(event);
    return true;
}

bool MarkerGraphCoordinator::CommitRefine(Atlas& atlas, Map* map,
    const MarkerGraphOptimizer::Proposal& p, MarkerGraphEvent& event)
{
    if(!ValidateProposal(p,map) || !map->mbMetric) return false;
    if(!RefineGaugeUnchanged(map,p)) {
        event.reason="marker_gauge_shift_validation_failed";
        return false;
    }
    event.sequence=++atlas.mnMarkerGraphSequence;
    MarkerGraphTransform markerGauge; // Identity: fixed-origin BA does not change the world frame.
    markerGauge.sequence=event.sequence;
    ApplyProposal(p,event.sequence);
    // Publish the sequence without moving/scaling all dense marker-anchored
    // history by an average of unrelated free-marker refinements. Re-estimating
    // old B-only poses from its new layout needs their raw pixels separately;
    // this fixed-world transaction does not claim to perform that operation.
    for(const auto& item:p.keyframePoses)
        item.first->mReplayMarkerGauge=markerGauge*item.first->mReplayMarkerGauge;
    ApplyStaticTags(map,p);
    map->mnMarkerGraphSequence=event.sequence;
    map->InformNewBigChange(); map->IncreaseChangeIndex();
    event.type="marker_global_ba"; event.status="accepted";
    event.reason="camera_marker_point_joint_ba_validated";
    event.mapId=event.targetMapId=long(map->GetId()); event.revision=map->mnRevision;
    event.markerIds=p.optimizedMarkerIds;
    FillResiduals(event,p); atlas.mMarkerGraphEvents.push_back(event);
    return true;
}

bool MarkerGraphCoordinator::CommitMerge(Atlas& atlas, Map* target, Map* source,
    const MarkerMapMerge::Proposal& p, MarkerGraphEvent& event)
{
    if(!target || !source || target==source || source->IsBad() || !p.accepted ||
       !source->mbMetric || !target->mbMetric || !ValidateProposal(p.graph,target,source) ||
       !p.sourceToTarget.matrix().allFinite() ||
       p.sourceMapId!=source->GetId() || p.targetMapId!=target->GetId() ||
       p.sourceRevision!=source->mnRevision || p.targetRevision!=target->mnRevision) return false;
    const auto sourceKFs=source->GetAllKeyFrames();
    const auto sourceMPs=source->GetAllMapPoints();
    KeyFrame* targetRoot=target->GetOriginKF();
    if(!targetRoot || targetRoot->isBad()) return false;
    for(KeyFrame* kf:sourceKFs) if(!kf->isBad() && !p.graph.keyframePoses.count(kf)) return false;
    for(MapPoint* point:sourceMPs) if(!point->isBad() && !p.graph.pointPositions.count(point)) return false;
    event.sequence=++atlas.mnMarkerGraphSequence;
    MarkerGraphOptimizer::PoseMap oldWorldPoses;
    for(const auto& item:p.graph.keyframePoses)
        oldWorldPoses[item.first]=item.first->GetPoseInverse();
    ApplyProposal(p.graph,event.sequence);
    const bool sourceActive=atlas.GetCurrentMap()==source;
    for(const auto& item:oldWorldPoses) {
        KeyFrame* kf=item.first;
        const auto delta=MarkerGraphTransform::between(
            item.second,kf->GetPoseInverse(),1.,event.sequence);
        kf->mReplayMarkerGauge=delta*kf->mReplayMarkerGauge;
    }
    for(KeyFrame* kf:sourceKFs) {
        if(kf->isBad()) continue;
        kf->UpdateMap(target); target->AddKeyFrame(kf); source->EraseKeyFrame(kf);
    }
    for(MapPoint* point:sourceMPs) {
        if(point->isBad()) continue;
        point->UpdateMap(target); target->AddMapPoint(point); source->EraseMapPoint(point);
    }
    // Join the source tree, not each KF independently. A common rigid marker
    // connects the two sets of raw corner factors; duplicate background MPs
    // may remain until ordinary local mapping finds descriptor matches.
    KeyFrame* sourceBridge=nullptr;
    for(KeyFrame* kf:sourceKFs) {
        if(kf->isBad()) continue;
        if(!sourceBridge) sourceBridge=kf;
        KeyFrame* parent=kf->GetParent();
        if(!parent || parent->GetMap()!=target) {
            kf->ChangeParent(targetRoot); kf->SetFirstConnection(false);
        }
    }
    if(sourceBridge) {
        targetRoot->AddMergeEdge(sourceBridge); sourceBridge->AddMergeEdge(targetRoot);
    }
    target->mStaticTags=p.staticTags;
    ApplyStaticTags(target,p.graph);
    target->mbBackgroundReady=target->mbBackgroundReady || source->mbBackgroundReady;
    target->mbMarkerSeed=target->mbMarkerSeed || source->mbMarkerSeed;
    target->mbRigidMarkerLayout=target->mbRigidMarkerLayout && source->mbRigidMarkerLayout;
    if(sourceActive) target->mMarkerInputToWorld=p.sourceToTarget*source->mMarkerInputToWorld;
    target->mnMarkerGraphSequence=event.sequence;
    if(target->GetMarkerScaleAnchorKFId()<0) {
        const auto anchors=StrongKeyframes(target);
        if(!anchors.empty()) target->SetMarkerScaleAnchorKFId(long(anchors.front()->mnId));
    }
    source->mvpKeyFrameOrigins.clear();
    for(const auto& item:p.graph.keyframePoses) item.first->UpdateConnections();
    atlas.mMarkerMapAliases[source->GetId()]=target->GetId();
    for(auto& alias:atlas.mMarkerMapAliases)
        if(alias.second==source->GetId()) alias.second=target->GetId();
    if(sourceActive) atlas.ChangeMap(target);
    atlas.SetMapBad(source); atlas.RemoveBadMaps();
    target->InformNewBigChange(); target->IncreaseChangeIndex();
    event.type="marker_map_merge"; event.status="accepted"; event.reason="common_marker_joint_ba_validated";
    event.mapId=event.targetMapId=long(target->GetId()); event.sourceMapId=long(source->GetId());
    event.scale=1.0; event.markerIds=p.verifiedMarkerIds; event.revision=target->mnRevision;
    FillResiduals(event,p.graph); atlas.mMarkerGraphEvents.push_back(event);
    return true;
}

bool MarkerGraphCoordinator::AlignMarkerInput(Map* map, Sophus::SE3f& Twc,
    std::vector<Eigen::Vector3f>& corners, const std::vector<int>& ids,
    const std::vector<float>& weights, bool partial, std::string& reason,
    GeometricCamera* camera, const std::vector<cv::Point2f>& pixels, bool inputInAtlasWorld)
{
    if(!map || !map->mbMetric || map->mStaticTags.empty()) return true;
    // A component just registered against the current visual pose is already
    // in Atlas coordinates. Reapplying the previous component's cached gauge
    // rotates a distant new station about the wrong origin (cm/m of error).
    // Shared registered IDs below still refresh stale layouts after later BA.
    Sophus::SE3f inputToWorld=inputInAtlasWorld ? Sophus::SE3f() : map->mMarkerInputToWorld;
    // Independent marker poses are BA variables. A single rigid transform
    // cannot carry an old cached layout into its independently refined one.
    // Use that transform only as a PnP seed, then use committed Atlas corners.
    // Initial metricization can already optimize these poses while the graph
    // event sequence is still zero. Admission depends on committed geometry,
    // not whether a loop/reanchor event has previously been published.
    const bool refineLayout=!partial && !map->mbRigidMarkerLayout &&
        camera &&
        camera->GetType()==GeometricCamera::CAM_PINHOLE &&
        pixels.size()==corners.size() && ids.size()==corners.size();
    // Pose alignment is strong-only. Geometry references are not: a decoded
    // low-information square still refers to the same four committed corners.
    // Transporting it with another marker's rigid transform would preserve an
    // obsolete relative layout after independent-marker BA.
    using Vec3fEntry=std::pair<const std::size_t,Eigen::Vector3f>;
    std::map<std::size_t,Eigen::Vector3f,std::less<std::size_t>,
             Eigen::aligned_allocator<Vec3fEntry>> registeredCorners,canonicalCorners;
    AlignedVector<Eigen::Vector3d> from,to;
    if(!partial && ids.size()==corners.size()) {
        std::map<int,std::size_t> cornerCounts;
        for(const int id:ids) ++cornerCounts[id];
        for(std::size_t i=0;i+3<ids.size();i+=4) {
            auto tag=map->mStaticTags.find(ids[i]);
            if(tag==map->mStaticTags.end()) continue;
            // No corner indices are carried by partial/tracked observations.
            // Only an unambiguous complete decoded group has the protocol's
            // ordered corner correspondence; never infer one from proximity.
            bool complete=cornerCounts[ids[i]]==4;
            for(std::size_t j=i;j<i+4;++j)
                complete=complete && ids[j]==ids[i];
            if(!complete) continue;
            if(tag->second.size()!=12) {
                reason="registered_marker_geometry_conflict"; return false;
            }
            bool strong=true;
            Eigen::Vector3d stored[4];
            for(int j=0;j<4;++j) {
                stored[j]=Eigen::Vector3d(tag->second[j*3],tag->second[j*3+1],tag->second[j*3+2]);
                if(!stored[j].allFinite() || !corners[i+j].allFinite()) {
                    reason="registered_marker_geometry_conflict"; return false;
                }
                strong=strong &&
                    (weights.empty() || (i+j<weights.size() && weights[i+j]>=.99f));
            }
            const Eigen::Vector3d supplied0=corners[i].cast<double>();
            const Eigen::Vector3d supplied1=corners[i+1].cast<double>();
            const double suppliedSide=(supplied1-supplied0).norm();
            const double storedSide=(stored[1]-stored[0]).norm();
            const double storedEdgeRoundoff=MarkerDistanceRoundoff(stored[0],stored[1]);
            const double edgeRoundoff=storedEdgeRoundoff+MarkerDistanceRoundoff(supplied0,supplied1);
            if(storedSide<=0 || edgeRoundoff>storedSide*.01 ||
               std::abs(suppliedSide-storedSide)>std::max(storedSide*1e-4,edgeRoundoff)) {
                reason="registered_marker_size_conflict"; return false;
            }
            // Keep the physical size gate when a marker becomes weak. Check
            // the entire ordered square, not just its first edge: swapping or
            // repeating individual corners must not silently relabel pixels.
            // Cyclic/reversed square symmetries still rely on decoded order.
            for(int j=0;j<4;++j) for(int k=j+1;k<4;++k) {
                const double distance=(stored[k]-stored[j]).norm();
                const double expected=storedSide*((k-j==2)?std::sqrt(2.):1.);
                const Eigen::Vector3d sj=corners[i+j].cast<double>(),sk=corners[i+k].cast<double>();
                const double supplied=(sk-sj).norm();
                const double pairRoundoff=MarkerDistanceRoundoff(stored[j],stored[k]);
                const double shapeRoundoff=pairRoundoff+storedEdgeRoundoff*((k-j==2)?std::sqrt(2.):1.);
                const double correspondenceRoundoff=pairRoundoff+MarkerDistanceRoundoff(sj,sk);
                // World coordinates can be tens of metres while a marker is
                // only 48 mm wide. Allow float quantization in these added
                // shape checks; the original first-edge size gate is above.
                if(distance<=0 || std::max(shapeRoundoff,correspondenceRoundoff)>storedSide*.01 ||
                   std::abs(distance-expected)>std::max({1e-5,expected*1e-4,shapeRoundoff}) ||
                   std::abs(supplied-distance)>std::max({1e-5,distance*1e-4,correspondenceRoundoff})) {
                    reason="registered_marker_geometry_conflict"; return false;
                }
            }
            for(int j=0;j<4;++j) {
                canonicalCorners.emplace(i+j,stored[j].cast<float>());
                if(strong) {
                    from.push_back(corners[i+j].cast<double>());
                    to.push_back(stored[j]);
                    if(refineLayout) registeredCorners.emplace(i+j,stored[j].cast<float>());
                }
            }
        }
    }
    if(from.size()>=4) {
        Eigen::Vector3d a=Eigen::Vector3d::Zero(),b=a;
        for(std::size_t i=0;i<from.size();++i) { a+=from[i]; b+=to[i]; }
        a/=from.size(); b/=to.size();
        Eigen::Matrix3d covariance=Eigen::Matrix3d::Zero();
        for(std::size_t i=0;i<from.size();++i) covariance+=(from[i]-a)*(to[i]-b).transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance,Eigen::ComputeFullU|Eigen::ComputeFullV);
        if(svd.singularValues()[1]<1e-10) { reason="degenerate_registered_marker"; return false; }
        Eigen::Matrix3d sign=Eigen::Matrix3d::Identity();
        sign(2,2)=(svd.matrixV()*svd.matrixU().transpose()).determinant()<0?-1:1;
        const Eigen::Matrix3d R=svd.matrixV()*sign*svd.matrixU().transpose();
        const Eigen::Vector3d t=b-R*a;
        for(std::size_t i=0;i<from.size();++i)
            if(!refineLayout && (R*from[i]+t-to[i]).norm()>.005) { reason="registered_marker_layout_conflict"; return false; }
        inputToWorld=Sophus::SE3f(Eigen::Quaternionf(R.cast<float>()).normalized(),t.cast<float>());
    }
    if(!inputToWorld.matrix().allFinite()) { reason="invalid_marker_gauge"; return false; }
    Sophus::SE3f alignedPose=inputToWorld*Twc;
    if(!alignedPose.matrix().allFinite()) { reason="invalid_marker_gauge"; return false; }
    auto alignedCorners=corners;
    for(auto& p:alignedCorners) p=inputToWorld*p;
    double layoutChange=0;
    for(const auto& item:registeredCorners)
        layoutChange=std::max(layoutChange,double((alignedCorners[item.first]-item.second).norm()));
    if(refineLayout && registeredCorners.size()>=4 && layoutChange>1e-5) {
        std::vector<cv::Point3d> world;
        std::vector<cv::Point2d> image;
        for(const auto& item:registeredCorners) {
            alignedCorners[item.first]=item.second;
            world.emplace_back(item.second.x(),item.second.y(),item.second.z());
            image.emplace_back(pixels[item.first].x,pixels[item.first].y);
        }
        const auto Tcw=alignedPose.inverse();
        cv::Mat rotation(3,3,CV_64F),translation(3,1,CV_64F),rvec;
        for(int row=0;row<3;++row) {
            translation.at<double>(row)=Tcw.translation()[row];
            for(int col=0;col<3;++col) rotation.at<double>(row,col)=Tcw.rotationMatrix()(row,col);
        }
        cv::Rodrigues(rotation,rvec);
        try {
            if(!cv::solvePnP(world,image,camera->toK(),cv::noArray(),rvec,translation,true,
                             cv::SOLVEPNP_ITERATIVE)) {
                reason="refined_marker_pose_failed"; return false;
            }
            cv::Rodrigues(rvec,rotation);
        } catch(const cv::Exception&) { reason="refined_marker_pose_failed"; return false; }
        Eigen::Matrix3f R; Eigen::Vector3f t;
        for(int row=0;row<3;++row) {
            t[row]=translation.at<double>(row);
            for(int col=0;col<3;++col) R(row,col)=rotation.at<double>(row,col);
        }
        if(!R.allFinite() || !t.allFinite()) { reason="refined_marker_pose_failed"; return false; }
        const Sophus::SE3f refined(Eigen::Quaternionf(R).normalized(),t);
        double error=0;
        for(const auto& item:registeredCorners) {
            const Eigen::Vector3f p=refined*item.second;
            if(!p.allFinite() || p.z()<=0) { reason="refined_marker_depth_failed"; return false; }
            const Eigen::Vector2f uv=camera->project(p);
            error+=(uv-Eigen::Vector2f(pixels[item.first].x,pixels[item.first].y)).squaredNorm();
        }
        if(std::sqrt(error/registeredCorners.size())>3.0) {
            reason="refined_marker_reprojection_failed"; return false;
        }
        alignedPose=refined.inverse();
    }
    // Commit references only after all strong-pose/size/reprojection checks.
    // Weak pixels and information weights remain untouched and never enter
    // the SVD seed or PnP above; downstream admission keeps its original gates.
    for(const auto& item:canonicalCorners) alignedCorners[item.first]=item.second;
    Twc=alignedPose;
    corners=std::move(alignedCorners);
    map->mMarkerInputToWorld=inputToWorld;
    return true;
}

void MarkerGraphCoordinator::Cancel()
{
    if(ownsStop_) tracker_.mpLocalMapper->ReleaseTagAlignmentStop();
    ownsStop_=false; pendingKind_=Kind::None; pendingSource_=pendingTarget_=pendingActive_=nullptr;
    pendingB_.clear(); samples_.clear(); observedMap_=nullptr;
    cornerRevisit_=CornerRevisitWindow();
    observedCorrectionEpoch_=-1; pendingCorrectionEpoch_=-1;
}

void MarkerGraphCoordinator::Schedule(Kind kind, Map* source, Map* target)
{
    pendingKind_=kind; pendingSource_=source; pendingTarget_=target;
    pendingActive_=tracker_.mpAtlas->GetCurrentMap();
    pendingCorrectionEpoch_=source->GetLastBigChangeIdx();
    candidateFrame_=long(tracker_.mCurrentFrame.mnId);
    candidateTime_=tracker_.mCurrentFrame.mTimeStamp;
    if(!std::isfinite(candidateTime_)) {
        candidateTime_=0; candidateFrame_=0;
        for(KeyFrame* k:source->GetAllKeyFrames())
            if(k && !k->isBad() && std::isfinite(k->mTimeStamp) && k->mTimeStamp>=candidateTime_) {
                candidateTime_=k->mTimeStamp; candidateFrame_=long(k->mnFrameId);
            }
    }
    tracker_.mpLocalMapper->RequestTagAlignmentStop(); ownsStop_=true;
}

void MarkerGraphCoordinator::OnFrameEnd(bool final)
{
    if(!final && tracker_.mbHasExternalTagObservation &&
       long(tracker_.mCurrentFrame.mnId)!=lastMarkerObservationFrame_) {
        const double timestamp=tracker_.mCurrentFrame.mTimeStamp;
        if(lastMarkerObservationTime_<0 || timestamp-lastMarkerObservationTime_>kDenseMarkerGapS)
            ++markerEpisode_;
        lastMarkerObservationFrame_=long(tracker_.mCurrentFrame.mnId);
        lastMarkerObservationTime_=timestamp;
    }
    while(true) {
        if(pendingKind_!=Kind::None) {
            ProcessPending(final);
            if(pendingKind_!=Kind::None) return;
            if(!final) return;
        }
        if(tracker_.mbTagAlignmentPending || (!final && tracker_.mbOnlyTracking)) return;
        Map* map=tracker_.mpAtlas->GetCurrentMap();
        if(observedMap_!=map) {
            samples_.clear(); cornerRevisit_=CornerRevisitWindow(); observedMap_=map;
        }
        std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);
        const int correctionEpoch=map->GetLastBigChangeIdx();
        if(observedCorrectionEpoch_!=correctionEpoch) {
            // Dense pose samples are copies, not keyframe-relative poses.
            // A loop/merge/scale correction cannot update those old copies.
            samples_.clear(); cornerRevisit_=CornerRevisitWindow(); observedCorrectionEpoch_=correctionEpoch;
        }
        if(!map->mbMetric || map->IsInertial() || map->IsBad()) return;
        const auto strong=StrongKeyframes(map);
        if(!strong.empty() && std::none_of(strong.begin(),strong.end(),[&](KeyFrame* kf) {
                return long(kf->mnId)==map->GetMarkerScaleAnchorKFId(); }))
            map->SetMarkerScaleAnchorKFId(long(strong.front()->mnId));
        if(map->GetMarkerScaleAnchorKFId()>=0 && !scaleAnchorEpisodes_.count(map->GetId()))
            scaleAnchorEpisodes_[map->GetId()]=markerEpisode_;
        bool scheduledMerge=false;
        // Online, only probe a merge while a current marker observation makes
        // the extra work worthwhile. Offline finalization must instead use
        // all already committed keyframe evidence: requiring the final video
        // frame itself to contain a tag can strand two metric maps that both
        // observed the same registered physical marker minutes earlier.
        if((final || tracker_.mbHasExternalTagObservation) && strong.size()>=3) {
            for(Map* other:tracker_.mpAtlas->GetAllMaps()) {
                if(other==map || other->IsBad() || !other->mbMetric || other->IsInertial() ||
                   other->KeyFramesInMap()<3) continue;
                bool common=false;
                for(const auto& tag:map->mStaticTags) if(other->mStaticTags.count(tag.first)) common=true;
                if(!common) continue;
                // Preserve the oldest map's gauge even when it is already the
                // active map. This also permits 2->0, then inactive 1->0 after
                // map 2 supplied the common-marker connection between 0 and 1.
                Map* target=map->GetId()<other->GetId()?map:other;
                Map* source=target==map?other:map;
                const auto signature=std::make_pair(source->GetMaxKFid(),target->GetMaxKFid());
                const auto key=std::make_pair(source->GetId(),target->GetId());
                auto previous=mergeAttempts_.find(key);
                if(previous!=mergeAttempts_.end() && previous->second==signature) continue;
                Schedule(Kind::Merge,source,target);
                scheduledMerge=true;
                break;
            }
        }
        if(scheduledMerge) {
            if(!final) return;
            lock.unlock();
            continue;
        }
        if(final) {
            // The offline shutdown boundary is the only place where a full-map
            // camera/marker/point BA is requested. A map revision identifies
            // the exact snapshot and prevents an accepted or rejected solve
            // from being repeated indefinitely.
            const auto signature=std::make_pair(map->GetMaxKFid(),map->mnRevision);
            const auto previous=refineAttempts_.find(map->GetId());
            if(map->mbBackgroundReady && strong.size()>=2 &&
               (previous==refineAttempts_.end() || previous->second!=signature)) {
                Schedule(Kind::Refine,map,map);
                lock.unlock();
                continue;
            }
            lock.unlock();
            // Disconnected Atlas maps keep independent gauges, but each metric
            // map still deserves its own final joint BA before serialization.
            for(Map* candidate:tracker_.mpAtlas->GetAllMaps()) {
                if(candidate==map || candidate->IsBad() || !candidate->mbMetric ||
                   candidate->IsInertial() || !candidate->mbBackgroundReady) continue;
                std::unique_lock<std::mutex> candidateLock(candidate->mMutexMapUpdate);
                if(StrongKeyframes(candidate).size()<2) continue;
                const auto candidateSignature=std::make_pair(
                    candidate->GetMaxKFid(),candidate->mnRevision);
                const auto candidatePrevious=refineAttempts_.find(candidate->GetId());
                if(candidatePrevious!=refineAttempts_.end() &&
                   candidatePrevious->second==candidateSignature) continue;
                Schedule(Kind::Refine,candidate,candidate);
                candidateLock.unlock();
                break;
            }
            if(pendingKind_!=Kind::None) continue;
            return;
        }
        if(tracker_.mnMarkerGraphVisualFrameId!=long(tracker_.mCurrentFrame.mnId) ||
           !tracker_.mbHasExternalTagObservation || tracker_.mState!=Tracking::OK || !map->mbBackgroundReady) return;
        if(!samples_.empty() && tracker_.mCurrentFrame.mTimeStamp-samples_.back().timestamp>kDenseMarkerGapS) samples_.clear();
        ScaleSample sample;
        sample.frameId=tracker_.mCurrentFrame.mnId; sample.timestamp=tracker_.mCurrentFrame.mTimeStamp;
        sample.correctionEpoch=correctionEpoch;
        sample.visualTwc=tracker_.mMarkerGraphVisualTwc; sample.markerTwc=tracker_.mExternalTagTwc;
        samples_.push_back(sample);
        if(samples_.size()>240) samples_.erase(samples_.begin());
        auto evidence=EstimateScale(samples_);
        std::map<int,int> strongCounts;
        for(std::size_t i=0;i<tracker_.mvExternalTagIds.size();++i)
            if(tracker_.mvExternalTagPointWeights.empty() ||
               (i<tracker_.mvExternalTagPointWeights.size() && tracker_.mvExternalTagPointWeights[i]>=.99f))
                ++strongCounts[tracker_.mvExternalTagIds[i]];
        std::set<int> currentIds;
        if(tracker_.mExternalTagConfidence>=.35f)
            for(const auto& id:strongCounts) if(id.second==4) currentIds.insert(id.first);
        cornerRevisit_.Observe(map,sample.frameId,sample.timestamp,currentIds);
        if(strong.size()<4) return;
        std::vector<unsigned long> b;
        KeyFrame* anchorA=nullptr;
        std::set<int> bMarkerIds;
        for(KeyFrame* kf:strong)
            if(long(kf->mnId)==map->GetMarkerScaleAnchorKFId()) anchorA=kf;
            else if(kf->mnFrameId>=samples_.front().frameId &&
                    kf->mnFrameId<=samples_.back().frameId) {
                b.push_back(kf->mnId);
                const auto ids=StrongIds(kf);
                bMarkerIds.insert(ids.begin(),ids.end());
            }
        // A is fixed, while two independently timestamped B keyframes give
        // the optimizer a non-degenerate metric interval.  Scale itself is
        // already guarded by >=8 dense observations and >=4 cm translation;
        // requiring a third sparse keyframe needlessly discards valid clips.
        if(!anchorA) return;
        bool revisitDriven=false;
        if(b.size()<2) {
            auto retained=SelectCornerRevisit(evidence,cornerRevisit_,anchorA,strong);
            if(retained.empty()) return;
            b=std::move(retained); revisitDriven=true;
        }
        const auto aMarkerIds=StrongIds(anchorA);
        bool newAnchor=false;
        for(int id:bMarkerIds) if(!aMarkerIds.count(id)) { newAnchor=true; break; }
        const auto anchorEpisode=scaleAnchorEpisodes_.find(map->GetId());
        const auto attemptedEpisode=scaleAttemptEpisodes_.find(map->GetId());
        const bool revisitedAnchor=anchorEpisode!=scaleAnchorEpisodes_.end() &&
            markerEpisode_>anchorEpisode->second &&
            (attemptedEpisode==scaleAttemptEpisodes_.end() || attemptedEpisode->second!=markerEpisode_);
        const bool intervalClosure=newAnchor || revisitedAnchor;
        const bool cornerDriven=revisitDriven || CanTryCornerInterval(evidence,intervalClosure,b.size());
        if(!ShouldScheduleScale(evidence,intervalClosure) && !cornerDriven) return;
        if(cornerDriven) {
            // Do not feed a rejected PnP motion ratio into the graph. A weak
            // unit prior only regularizes initialization; raw rigid corners
            // and multi-view background geometry decide whether to commit.
            evidence.metricPerVisual=1.; evidence.sigma=.1;
            evidence.reliable=false;
        }
        // A near-unit closure is useful once per marker episode. Repeating it
        // while the same marker stays visible only adds runtime and correlated
        // factors. Significant drift may retry as the keyframe path matures.
        if(!evidence.reliable && attemptedEpisode!=scaleAttemptEpisodes_.end() &&
           attemptedEpisode->second==markerEpisode_) return;
        const auto retry=scaleRetries_.find(map->GetId());
        if(retry!=scaleRetries_.end() && retry->second.episode==markerEpisode_) {
            std::size_t newViews=0;
            for(KeyFrame* k:strong) if(!retry->second.frames.count(k->mnFrameId)) ++newViews;
            // Count genuinely new measurements, not a net KF count: culling
            // old keyframes must not hide the arrival of fresh evidence.
            if(!CanRetryScale(0,newViews,retry->second.attempts,
                tracker_.mCurrentFrame.mTimeStamp-retry->second.timestamp)) return;
        }
        auto previous=scaleAttempts_.find(map->GetId());
        if(previous!=scaleAttempts_.end() && previous->second==b.back()) return;
        pendingB_=b; pendingScale_=evidence;
        if(cornerDriven) std::cout << "MARKER_CORNER_INTERVAL_CANDIDATE map=" << map->GetId()
            << " observations=" << evidence.observations << " keyframes=" << b.size()
            << " baseline_m=" << evidence.baselineM << " retained_revisit=" << revisitDriven
            << " raw_first_frame=" << cornerRevisit_.firstFrame << std::endl;
        Schedule(Kind::Scale,map,map);
        return;
    }
}

void MarkerGraphCoordinator::ProcessPending(bool final)
{
    auto& tracker=tracker_;
    Atlas& atlas=*tracker.mpAtlas;
    if(tracker.mbTagAlignmentPending || !pendingSource_ || !pendingTarget_ ||
       atlas.GetCurrentMap()!=pendingActive_ || pendingSource_->IsBad() || pendingTarget_->IsBad()) {
        Cancel(); return;
    }
    if(!(tracker.mpLocalMapper->isStoppedForTagAlignment() || (final && tracker.mpLocalMapper->isFinished()))) return;
    // RequestFinish can end the mapper with an unprocessed KF in its queue.
    // Drain it after the worker has returned, before acquiring map locks, so
    // a current-frame reference cannot be left in the retired source map.
    if(final && tracker.mpLocalMapper->isFinished()) tracker.mpLocalMapper->EmptyQueue();
    // No blocking wait while holding a map lock: a mapper's last BA must be
    // allowed to finish before we acquire the stopped-map snapshot.
    Map* first=pendingSource_->GetId()<pendingTarget_->GetId()?pendingSource_:pendingTarget_;
    Map* second=first==pendingSource_?pendingTarget_:pendingSource_;
    std::unique_lock<std::mutex> lock1(first->mMutexMapUpdate);
    std::unique_lock<std::mutex> lock2;
    if(second!=first) lock2=std::unique_lock<std::mutex>(second->mMutexMapUpdate);
    if(pendingKind_==Kind::Scale &&
       pendingSource_->GetLastBigChangeIdx()!=pendingCorrectionEpoch_) {
        // A pending dense scale cue belongs to the pre-correction geometry.
        // Recollect it rather than applying it to newly corrected keyframes.
        std::cout << "MARKER_SCALE_GATE stage=epoch reason=map_corrected_while_pending" << std::endl;
        Cancel(); return;
    }
    MarkerGraphEvent event;
    event.frameId=long(tracker.mCurrentFrame.mnId); event.timestamp=tracker.mCurrentFrame.mTimeStamp;
    if(!std::isfinite(event.timestamp)) {event.timestamp=candidateTime_; event.frameId=candidateFrame_;}
    event.candidateFrameId=candidateFrame_; event.candidateTimestamp=candidateTime_;
    event.mapId=event.targetMapId=long(pendingTarget_->GetId());
    event.sourceMapId=pendingKind_==Kind::Merge?long(pendingSource_->GetId()):-1;
    event.type=pendingKind_==Kind::Merge?"marker_map_merge":
        (pendingKind_==Kind::Refine?"marker_global_ba":"scale_reanchor");
    event.scale=pendingKind_==Kind::Scale?pendingScale_.metricPerVisual:1.0;
    event.sigma=pendingKind_==Kind::Scale?pendingScale_.sigma:0.0;
    struct Reference { Sophus::SE3f pose; float scale; MarkerGraphTransform graph,gauge; };
    // Reference contains fixed-size Eigen/Sophus values. In this C++14 build,
    // libstdc++'s default map node allocator does not promise their required
    // over-alignment; an ordinary std::map corrupted the node during offline
    // finalization on Linux. Use Eigen's allocator explicitly.
    using ReferenceEntry=std::pair<KeyFrame* const,Reference>;
    std::map<KeyFrame*,Reference,std::less<KeyFrame*>,
             Eigen::aligned_allocator<ReferenceEntry>> references;
    // Tracking history stores raw reference pointers. A queued bootstrap
    // keyframe may be retired while a final marker-graph request is pending,
    // so validate pointer identity against the locked maps before any
    // dereference. Only persisted map keyframes can participate in a graph
    // correction anyway.
    std::set<KeyFrame*> liveKeyFrames;
    for(KeyFrame* kf:first->GetAllKeyFrames()) if(kf) liveKeyFrames.insert(kf);
    if(second!=first)
        for(KeyFrame* kf:second->GetAllKeyFrames()) if(kf) liveKeyFrames.insert(kf);
    auto saveReference=[&](KeyFrame* kf) {
        if(!kf || !liveKeyFrames.count(kf) || references.count(kf)) return;
        Reference r; Map* owner=nullptr;
        if(kf->GetReplayReference(r.pose,r.scale,owner) && kf->GetReplayMarkerGraph(r.graph) &&
           kf->GetReplayMarkerGauge(r.gauge)) references[kf]=r;
    };
    for(KeyFrame* kf:tracker.mlpReferences) saveReference(kf);
    saveReference(tracker.mCurrentFrame.mpReferenceKF); saveReference(tracker.mLastFrame.mpReferenceKF);
    MarkerGraphOptimizer::Proposal graph;
    Sophus::SE3f sourceToTarget;
    bool accepted=false;
    try {
        if(pendingKind_==Kind::Merge) {
            const auto proposal=MarkerMapMerge::Propose(pendingTarget_,pendingSource_);
            graph=proposal.graph; event.reason=proposal.reason;
            event.markerIds=proposal.verifiedMarkerIds; sourceToTarget=proposal.sourceToTarget;
            if(proposal.accepted) {
                tracker.mpLoopClosing->InvalidateBAForMarkerGraph();
                accepted=CommitMerge(atlas,pendingTarget_,pendingSource_,proposal,event);
            }
        } else if(pendingKind_==Kind::Scale) {
            KeyFrame* a=nullptr;
            std::vector<KeyFrame*> b;
            for(KeyFrame* kf:StrongKeyframes(pendingSource_)) {
                if(long(kf->mnId)==pendingSource_->GetMarkerScaleAnchorKFId()) a=kf;
                if(std::find(pendingB_.begin(),pendingB_.end(),kf->mnId)!=pendingB_.end()) b.push_back(kf);
            }
            if(a && b.size()>=2) {
                graph=MarkerGraphOptimizer::Reanchor(pendingSource_,a,b,pendingScale_.metricPerVisual,pendingScale_.sigma);
                if(graph.cornerScale>0) {event.scale=graph.cornerScale; event.sigma=graph.cornerScaleSigma;}
                event.reason=graph.reason;
                for(KeyFrame* kf:b) for(int id:StrongIds(kf))
                    if(std::find(event.markerIds.begin(),event.markerIds.end(),id)==event.markerIds.end()) event.markerIds.push_back(id);
                if(graph.accepted) {
                    tracker.mpLoopClosing->InvalidateBAForMarkerGraph();
                    accepted=CommitScale(atlas,pendingSource_,graph,b.back(),event);
                }
            } else event.reason="anchor_or_two_independent_keyframes_unavailable";
        } else {
            graph=MarkerGraphOptimizer::RefineMetricMap(pendingSource_);
            event.reason=graph.reason;
            event.markerIds=graph.optimizedMarkerIds;
            if(graph.accepted) {
                tracker.mpLoopClosing->InvalidateBAForMarkerGraph();
                accepted=CommitRefine(atlas,pendingSource_,graph,event);
            }
        }
    } catch(const std::exception&) { event.reason="marker_graph_solver_exception"; }
    if(accepted) {
        // Correct native relative-history units once. Metric wrist/hand data
        // is downstream of the camera and is never touched by this operation.
        std::map<std::size_t,Tracking::MarkerMetricFrame*> anchoredHistory;
        for(auto& item:tracker.mMarkerMetricFrames) anchoredHistory[item.second.historyIndex]=&item.second;
        if(tracker.mlReferenceUnitScales.size()!=tracker.mlpReferences.size()) {
            // Compatibility for an in-memory trajectory created before unit
            // stamps were introduced (and for focused native fixtures).
            tracker.mlReferenceUnitScales.clear();
            for(KeyFrame* historicalReference:tracker.mlpReferences) {
                const auto old=references.find(historicalReference);
                tracker.mlReferenceUnitScales.push_back(
                    old!=references.end()?old->second.scale:1.f);
            }
        }
        auto ref=tracker.mlpReferences.begin();
        auto referenceScale=tracker.mlReferenceUnitScales.begin();
        std::size_t historyIndex=0;
        for(auto pose=tracker.mlRelativeFramePoses.begin();
            pose!=tracker.mlRelativeFramePoses.end() && ref!=tracker.mlpReferences.end() &&
                referenceScale!=tracker.mlReferenceUnitScales.end();
            ++pose,++ref,++referenceScale) {
            auto old=references.find(*ref);
            Sophus::SE3f newPose; float newScale; Map* owner=nullptr;
            if(old!=references.end() && (*ref)->GetReplayReference(newPose,newScale,owner)) {
                const auto anchored=anchoredHistory.find(historyIndex);
                if(anchored!=anchoredHistory.end()) {
                    MarkerGraphTransform nowGauge;
                    if((*ref)->GetReplayMarkerGauge(nowGauge)) {
                        auto& world=anchored->second->worldFromCamera;
                        world=(nowGauge*old->second.gauge.inverse()).apply(world);
                        *pose=world.inverse()*newPose;
                    }
                    if(anchored->second->hasVisualRelative)
                        anchored->second->visualCameraFromReference.translation()*=
                            newScale/old->second.scale;
                } else pose->translation()*=newScale/old->second.scale;
                // The stored relative translation is now in the new units;
                // advance its acquisition stamp so replay will not apply the
                // same scale ratio a second time.
                *referenceScale=newScale;
            }
            ++historyIndex;
        }
        auto correctFrame=[&](Frame& frame) {
            if(!frame.HasPose() || !frame.mpReferenceKF) return;
            const auto anchored=tracker.mMarkerMetricFrames.find(frame.mnId);
            if(anchored!=tracker.mMarkerMetricFrames.end()) {
                frame.SetPose(anchored->second.worldFromCamera.inverse());
                return;
            }
            auto old=references.find(frame.mpReferenceKF); MarkerGraphTransform now;
            if(old!=references.end() && frame.mpReferenceKF->GetReplayMarkerGraph(now))
                frame.SetPose((now*old->second.graph.inverse()).apply(frame.GetPose().inverse()).inverse());
        };
        correctFrame(tracker.mCurrentFrame); correctFrame(tracker.mLastFrame);
        if(pendingKind_==Kind::Merge && pendingActive_==pendingSource_) {
            tracker.mExternalTagTwc=sourceToTarget*tracker.mExternalTagTwc;
            tracker.mInitialTagTwc=sourceToTarget*tracker.mInitialTagTwc;
            for(auto& p:tracker.mvExternalTagWorldPoints) p=sourceToTarget*p;
            for(auto& p:tracker.mvInitialTagWorldPoints) p=sourceToTarget*p;
            if(tracker.mInitialFrame.HasPose())
                tracker.mInitialFrame.SetPose((sourceToTarget*tracker.mInitialFrame.GetPose().inverse()).inverse());
            tracker.mpMarkerSeedKF=pendingTarget_->GetOriginKF();
            tracker.mRecoveredTagMetricScale=pendingTarget_->mMetricScale;
            tracker.mbTagMetricAligned=true;
        }
        // World-frame SE(3) corrections cancel in the camera-relative motion
        // model. Its translation only needs the local unit ratio after Sim(3).
        // Throwing away a valid model here forced next-frame BoW against a
        // sparse marker keyframe, bypassing the many last-frame map matches.
        // This is a search prior, never a published pose or a motion constraint.
        // A graph correction changes both the current and previous Tcw poses.
        // Reusing the pre-correction velocity (or disabling it whenever a
        // marker was present) leaves the next high-motion frame with a stale
        // prediction and can turn a recoverable frame into a new submap.
        // Rebuild the ordinary ORB motion prior from the corrected pair.  This
        // is only a relative visual prior: it does not freeze the wrist or
        // publish a pose, and is disabled across map merges/finalization.
        bool keepVelocity=!final && tracker.mState==Tracking::OK &&
            tracker.mCurrentFrame.HasPose() && tracker.mLastFrame.HasPose() &&
            tracker.mCurrentFrame.mpReferenceKF && tracker.mLastFrame.mpReferenceKF &&
            tracker.mCurrentFrame.mpReferenceKF->GetMap()==atlas.GetCurrentMap() &&
            tracker.mLastFrame.mpReferenceKF->GetMap()==atlas.GetCurrentMap();
        if(keepVelocity) {
            const Sophus::SE3f correctedVelocity =
                tracker.mCurrentFrame.GetPose() * tracker.mLastFrame.GetPose().inverse();
            keepVelocity=correctedVelocity.matrix().allFinite();
            if(keepVelocity) tracker.mVelocity=correctedVelocity;
        }
        tracker.mbVelocity=keepVelocity;
        tracker.mnMarkerGraphVisualFrameId=-1;
        tracker.mvpLocalKeyFrames=atlas.GetCurrentMap()->GetAllKeyFrames();
        tracker.mvpLocalMapPoints=atlas.GetCurrentMap()->GetAllMapPoints();
        atlas.GetCurrentMap()->SetReferenceMapPoints(tracker.mvpLocalMapPoints);
        if(tracker.mpMapDrawer && tracker.mCurrentFrame.HasPose() &&
           (tracker.mState==Tracking::OK || tracker.mState==Tracking::MARKER_TRACKING))
            tracker.mpMapDrawer->SetCurrentCameraPose(tracker.mCurrentFrame.GetPose());
    } else {
        if(event.reason.empty()) event.reason="proposal_commit_precondition_changed";
        event.sequence=++atlas.mnMarkerGraphSequence; event.status="rejected";
        event.revision=pendingTarget_->mnRevision;
        FillResiduals(event,graph); atlas.mMarkerGraphEvents.push_back(event);
    }
    if(pendingKind_==Kind::Merge) {
        const auto key=std::make_pair(pendingSource_->GetId(),pendingTarget_->GetId());
        mergeAttempts_[key]=std::make_pair(
            pendingSource_->GetMaxKFid(),pendingTarget_->GetMaxKFid());
    } else if(pendingKind_==Kind::Scale && !pendingB_.empty()) {
        scaleAttempts_[pendingSource_->GetId()]=pendingB_.back();
        auto& retry=scaleRetries_[pendingSource_->GetId()];
        if(retry.episode!=markerEpisode_) retry=ScaleRetry();
        retry.episode=markerEpisode_; ++retry.attempts;
        for(KeyFrame* k:StrongKeyframes(pendingSource_)) retry.frames.insert(k->mnFrameId);
        retry.timestamp=event.timestamp;
        // Only a successful near-unit closure consumes the whole episode.
        // A rejection can be retried after more independent keyframes arrive.
        if(accepted) {
            scaleAttemptEpisodes_[pendingSource_->GetId()]=markerEpisode_;
            scaleAnchorEpisodes_[pendingSource_->GetId()]=markerEpisode_;
            scaleRetries_.erase(pendingSource_->GetId());
        }
        if(!final && !accepted && event.reason=="insufficient_scale_geometry")
            deferredScaleWindows_[pendingSource_->GetId()]=pendingB_;
        else if(accepted) deferredScaleWindows_.erase(pendingSource_->GetId());
    } else if(pendingKind_==Kind::Refine) {
        refineAttempts_[pendingSource_->GetId()]=std::make_pair(
            pendingSource_->GetMaxKFid(),pendingSource_->mnRevision);
    }
    Map* retryMap=nullptr;
    std::vector<unsigned long> retryWindow;
    if(final && accepted && pendingKind_==Kind::Refine) {
        const auto retry=deferredScaleWindows_.find(pendingSource_->GetId());
        if(retry!=deferredScaleWindows_.end()) {
            retryMap=pendingSource_; retryWindow=retry->second;
            deferredScaleWindows_.erase(retry); // consume even if retry fails
        }
    }
    Cancel();
    if(retryMap) {
        pendingB_=retryWindow;
        pendingScale_=ScaleEvidence();
        // Not a reused pre-loop measurement: weak unit prior in the current
        // metric map. Raw multi-view marker corners determine the correction;
        // all depth, anchor and reprojection checks still apply.
        pendingScale_.metricPerVisual=1.0; pendingScale_.sigma=.1;
        std::cout << "MARKER_SCALE_POST_BA_RETRY map=" << retryMap->GetId() << std::endl;
        Schedule(Kind::Scale,retryMap,retryMap);
    }
}
} // namespace ORB_SLAM3
