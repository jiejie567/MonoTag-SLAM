#include "MarkerMapMerge.h"
#include "CameraModels/GeometricCamera.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"

#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <iostream>

namespace ORB_SLAM3 {
namespace {

using Quad = std::array<Eigen::Vector3f, 4>;
using QuadMap = std::map<int, Quad>;

bool readMarker(const std::vector<float>& values, Quad& corners, double& side)
{
    if(values.size()!=12) return false;
    side=0;
    for(std::size_t i=0;i<4;++i) {
        corners[i]=Eigen::Vector3f(values[i*3],values[i*3+1],values[i*3+2]);
        if(!corners[i].allFinite()) return false;
    }
    for(std::size_t i=0;i<4;++i) side+=(corners[(i+1)%4]-corners[i]).norm()/4.0;
    if(!std::isfinite(side) || side<1e-4) return false;
    // A rigid physical square, including ordered corners and planarity.
    for(std::size_t i=0;i<4;++i) {
        const Eigen::Vector3f edge=corners[(i+1)%4]-corners[i];
        const Eigen::Vector3f next=corners[(i+2)%4]-corners[(i+1)%4];
        if(std::abs(edge.norm()/side-1)>0.01 ||
           std::abs(edge.dot(next))/(side*side)>0.01) return false;
    }
    return (corners[0]+corners[2]-corners[1]-corners[3]).norm()/side<=0.01;
}

bool readRegistry(const MarkerMapMerge::StaticTags& input, QuadMap& quads,
                  std::map<int,double>& sides)
{
    for(const auto& tag:input) {
        if(tag.first<0 || !readMarker(tag.second,quads[tag.first],sides[tag.first])) return false;
    }
    return true;
}

struct EvidenceSamples {
    std::set<unsigned long> frames;
    double first=std::numeric_limits<double>::infinity();
    double last=-std::numeric_limits<double>::infinity();
    double squaredError=0;
    std::size_t corners=0;
    double span() const {return frames.empty()?0:last-first;}
    double rms() const {return corners?std::sqrt(squaredError/corners):0;}
};

bool validPose(const Sophus::SE3f& pose)
{
    return pose.matrix().allFinite() &&
        std::abs(pose.rotationMatrix().determinant()-1)<1e-4f;
}

int markerCorner(const Eigen::Vector3f& point,const Quad& registered)
{
    for(int corner=0;corner<4;++corner)
        if((point-registered[corner]).norm()<=1e-4f) return corner;
    return -1;
}

// Validate every registered observation's identity/geometry. Only complete,
// strong decoded observations count as merge evidence; a weak corner cannot
// turn a single accidental detection into a verified common marker.
std::string collectEvidence(const std::vector<KeyFrame*>& keyframes,
                            const QuadMap& registry, const std::set<int>& common,
                            const MarkerMapMerge::Options& options,
                            std::map<int,EvidenceSamples>& samples,
                            const MarkerGraphOptimizer::PoseMap* poses=nullptr,
                            const QuadMap* projectionRegistry=nullptr)
{
    for(KeyFrame* keyframe:keyframes) {
        if(!keyframe->mbHasTagObservation) continue;
        const std::size_t count=keyframe->mvTagWorldPoints.size();
        if(!keyframe->mpCamera || !count ||
           keyframe->mvTagImagePoints.size()!=count || keyframe->mvTagIds.size()!=count ||
           (!keyframe->mvTagPointWeights.empty() && keyframe->mvTagPointWeights.size()!=count) ||
           !std::isfinite(keyframe->mTagObservationConfidence) ||
           keyframe->mTagObservationConfidence<0 || keyframe->mTagObservationConfidence>1 ||
           !std::isfinite(keyframe->mTimeStamp)) return "invalid_marker_observation";
        if(poses && !poses->count(keyframe)) return "missing_staged_keyframe_pose";
        const Sophus::SE3f pose=poses?poses->at(keyframe):keyframe->GetPose();
        if(!validPose(pose)) return "invalid_keyframe_pose";
        std::set<std::pair<int,int>> seen;
        std::map<int,std::array<int,4>> complete;
        for(std::size_t i=0;i<count;++i) {
            const int id=keyframe->mvTagIds[i];
            auto registered=registry.find(id);
            if(registered==registry.end())
                return "marker_observation_identity_mismatch";
            const auto& point=keyframe->mvTagWorldPoints[i];
            const auto& pixel=keyframe->mvTagImagePoints[i];
            const float weight=keyframe->mvTagPointWeights.empty()?1:keyframe->mvTagPointWeights[i];
            if(!point.allFinite() || !std::isfinite(pixel.x) || !std::isfinite(pixel.y) ||
               !std::isfinite(weight) || weight<=0 || weight>1) return "invalid_marker_observation";
            const int corner=markerCorner(point,registered->second);
            if(corner<0) return "marker_observation_layout_mismatch";
            if(!seen.emplace(id,corner).second) return "marker_observation_identity_mismatch";
            if(common.count(id) && weight>=.99f && keyframe->mbTagObservationActive &&
               keyframe->mTagObservationConfidence+1e-7>=options.minimumConfidence) {
                auto insertion=complete.emplace(id,std::array<int,4>{{-1,-1,-1,-1}});
                insertion.first->second[corner]=static_cast<int>(i);
            }
        }
        for(const auto& tag:complete) {
            const int id=tag.first;
            if(std::find(tag.second.begin(),tag.second.end(),-1)!=tag.second.end()) continue;
            const Quad& corners=(projectionRegistry?*projectionRegistry:registry).at(id);
            double squared=0;
            for(std::size_t j=0;j<4;++j) {
                const Eigen::Vector3f cameraPoint=pose*corners[j];
                if(!cameraPoint.allFinite() || cameraPoint.z()<=1e-5f)
                    return "marker_nonpositive_depth";
                const Eigen::Vector2f predicted=keyframe->mpCamera->project(cameraPoint);
                const auto& observed=keyframe->mvTagImagePoints[tag.second[j]];
                if(!predicted.allFinite()) return "invalid_marker_projection";
                squared+=(predicted-Eigen::Vector2f(observed.x,observed.y)).squaredNorm();
            }
            const double rms=std::sqrt(squared/4);
            if(rms>options.maximumMarkerRmsPx) {
                // A single stale/blurred keyframe must not veto an otherwise
                // well-supported common-anchor merge.  Keep the observation
                // out of the evidence set; the later minimum-frame/span gate
                // still rejects a map when too little consistent evidence
                // remains.  This is an outlier filter, not a relaxed merge
                // acceptance threshold.
                std::cout << "COMMON_MARKER_OUTLIER id=" << id
                          << " frame=" << keyframe->mnFrameId
                          << " rms_px=" << rms << std::endl;
                continue;
            }
            auto& evidence=samples[id];
            if(evidence.frames.insert(keyframe->mnFrameId).second) {
                evidence.first=std::min(evidence.first,keyframe->mTimeStamp);
                evidence.last=std::max(evidence.last,keyframe->mTimeStamp);
                evidence.squaredError+=squared;
                evidence.corners+=4;
            }
        }
    }
    return "";
}

Sophus::SE3f alignRigid(const QuadMap& target, const QuadMap& source,
                       const std::vector<int>& ids)
{
    Eigen::Vector3d a=Eigen::Vector3d::Zero(),b=Eigen::Vector3d::Zero();
    const double count=4.0*ids.size();
    for(int id:ids) for(std::size_t i=0;i<4;++i) {
        a+=source.at(id)[i].cast<double>()/count;
        b+=target.at(id)[i].cast<double>()/count;
    }
    Eigen::Matrix3d covariance=Eigen::Matrix3d::Zero();
    for(int id:ids) for(std::size_t i=0;i<4;++i)
        covariance+=(source.at(id)[i].cast<double>()-a)*(target.at(id)[i].cast<double>()-b).transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance,Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d sign=Eigen::Matrix3d::Identity();
    sign(2,2)=(svd.matrixV()*svd.matrixU().transpose()).determinant()<0?-1:1;
    const Eigen::Matrix3d rotation=svd.matrixV()*sign*svd.matrixU().transpose();
    return Sophus::SE3f(rotation.cast<float>(),(b-rotation*a).cast<float>());
}

} // namespace

MarkerMapMerge::Proposal MarkerMapMerge::Propose(Map* target, Map* source,
                                                const Options& options)
{
    Proposal result;
    auto reject=[&result](const std::string& reason) {result.reason=reason; return result;};
    if(!target || !source || target->IsBad() || source->IsBad()) return reject("map_unavailable");
    if(target==source) return reject("same_map");
    result.targetMapId=target->GetId(); result.sourceMapId=source->GetId();
    result.targetRevision=target->mnRevision; result.sourceRevision=source->mnRevision;
    if(!target->mbMetric || !std::isfinite(target->mMetricScale) || target->mMetricScale<=0)
        return reject("target_scale_unknown");
    if(!source->mbMetric || !std::isfinite(source->mMetricScale) || source->mMetricScale<=0)
        return reject("source_scale_unknown");
    if(target->IsInertial() || source->IsInertial()) return reject("inertial_merge_unsupported");
    if(options.minimumIndependentFrames<3 || !std::isfinite(options.minimumTimeSpanS) ||
       options.minimumTimeSpanS<=0 || !std::isfinite(options.minimumConfidence) ||
       options.minimumConfidence<0.35 || options.minimumConfidence>1 ||
       !std::isfinite(options.maximumMarkerRmsPx) || options.maximumMarkerRmsPx<=0 ||
       !std::isfinite(options.maximumSizeRelativeError) || options.maximumSizeRelativeError<=0 ||
       !std::isfinite(options.maximumLayoutErrorM) || options.maximumLayoutErrorM<=0)
        return reject("invalid_options");

    QuadMap targetQuads,sourceQuads;
    std::map<int,double> targetSides,sourceSides;
    if(!readRegistry(target->mStaticTags,targetQuads,targetSides) ||
       !readRegistry(source->mStaticTags,sourceQuads,sourceSides)) return reject("invalid_static_marker_geometry");
    for(const auto& tag:sourceQuads) if(targetQuads.count(tag.first)) {
        const int id=tag.first;
        result.commonMarkerIds.push_back(id);
        auto& evidence=result.evidence[id];
        evidence.sourceSideM=sourceSides.at(id); evidence.targetSideM=targetSides.at(id);
        if(std::abs(evidence.sourceSideM/evidence.targetSideM-1)>options.maximumSizeRelativeError)
            return reject("marker_size_mismatch");
    }
    if(result.commonMarkerIds.empty()) return reject("no_common_marker");

    const auto targetFrames=target->GetAllKeyFrames(),sourceFrames=source->GetAllKeyFrames();
    KeyFrame* root=target->GetOriginKF();
    if(!root || root->isBad() || root->GetMap()!=target ||
       std::find(targetFrames.begin(),targetFrames.end(),root)==targetFrames.end()) return reject("target_root_missing");
    std::set<unsigned long> identities;
    for(Map* map:{target,source}) for(KeyFrame* keyframe:map==target?targetFrames:sourceFrames) {
        if(!keyframe || keyframe->isBad() || keyframe->GetMap()!=map ||
           !identities.insert(keyframe->mnId).second || !validPose(keyframe->GetPose()))
            return reject("invalid_keyframe_membership");
    }
    const std::set<int> common(result.commonMarkerIds.begin(),result.commonMarkerIds.end());
    std::map<int,EvidenceSamples> targetSamples,sourceSamples;
    std::string reason=collectEvidence(targetFrames,targetQuads,common,options,targetSamples);
    if(!reason.empty()) return reject("target_"+reason);
    reason=collectEvidence(sourceFrames,sourceQuads,common,options,sourceSamples);
    if(!reason.empty()) return reject("source_"+reason);
    for(int id:result.commonMarkerIds) {
        const auto& a=targetSamples[id]; const auto& b=sourceSamples[id];
        auto& evidence=result.evidence[id];
        evidence.targetFrames=a.frames.size(); evidence.sourceFrames=b.frames.size();
        evidence.targetTimeSpanS=a.span(); evidence.sourceTimeSpanS=b.span();
        evidence.targetRmsPx=a.rms(); evidence.sourceRmsPx=b.rms();
        std::cout << "COMMON_MARKER_EVIDENCE id=" << id
                  << " target_frames=" << a.frames.size()
                  << " target_span_s=" << a.span()
                  << " source_frames=" << b.frames.size()
                  << " source_span_s=" << b.span() << std::endl;
        const auto primary=[&](const EvidenceSamples& sample) {
            return sample.frames.size()>=options.minimumIndependentFrames &&
                sample.span()+1e-9>=options.minimumTimeSpanS;
        };
        // A short recovered submap can legitimately contain only two strong
        // marker keyframes before it reaches the end of an offline clip. Do
        // not demand a third correlated view when the other map already has
        // full historical evidence. Two observations with a real temporal
        // separation still reject a one-frame false decode, and the ensuing
        // raw-corner joint BA remains the final geometric validator.
        const auto supporting=[&](const EvidenceSamples& sample) {
            return sample.frames.size()>=2 &&
                sample.span()+1e-9>=std::min(0.08,options.minimumTimeSpanS);
        };
        if((primary(a) && supporting(b)) || (primary(b) && supporting(a)))
            result.verifiedMarkerIds.push_back(id);
    }
    if(result.verifiedMarkerIds.empty()) return reject("insufficient_common_marker_evidence");

    // Only verified IDs estimate the transform. Every other shared ID remains
    // an independent counter-check, not an unverified vote in the fit.
    result.sourceToTarget=alignRigid(targetQuads,sourceQuads,result.verifiedMarkerIds);
    if(!validPose(result.sourceToTarget)) return reject("invalid_rigid_alignment");
    for(int id:result.commonMarkerIds) {
        double squared=0,maximum=0;
        for(std::size_t j=0;j<4;++j) {
            const double error=(result.sourceToTarget*sourceQuads.at(id)[j]-targetQuads.at(id)[j]).norm();
            squared+=error*error; maximum=std::max(maximum,error);
        }
        result.evidence[id].alignmentRmsM=std::sqrt(squared/4);
        result.evidence[id].alignmentMaximumM=maximum;
        if(maximum>options.maximumLayoutErrorM) return reject("common_marker_layout_conflict");
    }

    result.staticTags=target->mStaticTags;
    QuadMap mergedQuads=targetQuads;
    for(const auto& tag:sourceQuads) if(!mergedQuads.count(tag.first)) {
        auto& values=result.staticTags[tag.first];
        for(std::size_t j=0;j<4;++j) {
            const Eigen::Vector3f point=result.sourceToTarget*tag.second[j];
            mergedQuads[tag.first][j]=point;
            for(int axis=0;axis<3;++axis) values.push_back(point(axis));
        }
    }
    MarkerGraphOptimizer::StagedBAInput staged;
    staged.rigidMarkerLayout = target->mbRigidMarkerLayout && source->mbRigidMarkerLayout;
    staged.fixedKeyframes.insert(root);
    for(Map* map:{target,source}) {
        const bool moving=map==source;
        const auto& frames=moving?sourceFrames:targetFrames;
        const std::set<KeyFrame*> members(frames.begin(),frames.end());
        for(KeyFrame* keyframe:frames) {
            staged.keyframes.push_back(keyframe);
            staged.keyframePoses[keyframe]=moving?
                keyframe->GetPose()*result.sourceToTarget.inverse():keyframe->GetPose();
            staged.replayScaleMultipliers[keyframe]=1.f;
            if(keyframe->mbHasTagObservation) {
                auto& corners=staged.tagWorldCorners[keyframe];
                const QuadMap& original=moving?sourceQuads:targetQuads;
                // The input may append one or two weak LK corners from other
                // IDs after complete markers. Preserve each pixel's ordering.
                for(std::size_t i=0;i<keyframe->mvTagIds.size();++i) {
                    const int id=keyframe->mvTagIds[i];
                    const int corner=markerCorner(keyframe->mvTagWorldPoints[i],original.at(id));
                    corners.push_back(mergedQuads.at(id)[corner]);
                }
            }
        }
        for(MapPoint* point:map->GetAllMapPoints()) {
            if(!point || point->isBad()) continue;
            if(point->GetMap()!=map || !point->GetWorldPos().allFinite() || staged.pointPositions.count(point))
                return reject("invalid_point_membership");
            // An independently stored map must not share an untransformed
            // observer with another gauge. Otherwise BA would ignore that
            // outside pixel while this merge moved its referenced point.
            for(const auto& observation:point->GetObservations()) {
                KeyFrame* observer=observation.first;
                if(!observer) return reject("invalid_point_observation_membership");
                if(observer->isBad()) continue;
                if(observer->GetMap()!=map || !members.count(observer))
                    return reject("invalid_point_observation_membership");
            }
            staged.pointPositions[point]=moving?
                result.sourceToTarget*point->GetWorldPos():point->GetWorldPos();
        }
    }

    // The evidence pass above intentionally ignores a complete common-marker
    // group whose initial projection RMS is too high.  Keep that decision
    // consistent with the staged BA: otherwise the optimizer still receives
    // the stale corner pixels and the final strict tag gate rejects the whole
    // merge.  Exclude only complete strong common-marker groups, cap the
    // removal per map, and never remove the fixed target gauge.  This is a
    // bounded observation-level retry; all remaining marker and ORB factors
    // continue to constrain the candidate.
    const auto excludeInitialTagOutliers=[&](const std::vector<KeyFrame*>& frames,
                                             const QuadMap& original,
                                             const char* label) {
        std::vector<std::pair<KeyFrame*,int>> candidates;
        std::size_t completeGroups=0;
        for(KeyFrame* keyframe:frames) {
            if(!keyframe || keyframe==root || keyframe->isBad() ||
               !keyframe->mbHasTagObservation || !keyframe->mbTagObservationActive ||
               keyframe->mTagObservationConfidence < options.minimumConfidence)
                continue;
            for(std::size_t i=0;i+3<keyframe->mvTagIds.size();) {
                const int id=keyframe->mvTagIds[i];
                if(!common.count(id) || keyframe->mvTagIds[i+1]!=id ||
                   keyframe->mvTagIds[i+2]!=id || keyframe->mvTagIds[i+3]!=id ||
                   std::count(keyframe->mvTagIds.begin(),keyframe->mvTagIds.end(),id)!=4) {
                    ++i; continue;
                }
                bool strong=true;
                for(std::size_t corner=0;corner<4;++corner) {
                    const float weight=keyframe->mvTagPointWeights.empty()?1.f:
                        keyframe->mvTagPointWeights[i+corner];
                    strong=strong && std::isfinite(weight) && weight>=.99f && weight<=1.f;
                }
                ++completeGroups;
                if(strong) {
                    double squared=0.; bool valid=true;
                    for(std::size_t corner=0;corner<4;++corner) {
                        const int registeredCorner=markerCorner(
                            keyframe->mvTagWorldPoints[i+corner],original.at(id));
                        if(registeredCorner<0) {valid=false;break;}
                        const Eigen::Vector3f cameraPoint=staged.keyframePoses.at(keyframe)*
                            mergedQuads.at(id)[registeredCorner];
                        if(!cameraPoint.allFinite() || cameraPoint.z()<=1e-5f) {
                            valid=false;break;
                        }
                        const Eigen::Vector2f predicted=keyframe->mpCamera->project(cameraPoint);
                        const auto& observed=keyframe->mvTagImagePoints[i+corner];
                        if(!predicted.allFinite() || !std::isfinite(observed.x) ||
                           !std::isfinite(observed.y)) {valid=false;break;}
                        squared+=(predicted-Eigen::Vector2f(observed.x,observed.y)).squaredNorm();
                    }
                    const double rms=valid?std::sqrt(squared/4):std::numeric_limits<double>::infinity();
                    if(rms>options.maximumMarkerRmsPx)
                        candidates.emplace_back(keyframe,id);
                }
                i+=4;
            }
        }
        const std::size_t limit=std::max<std::size_t>(1,completeGroups/4);
        if(candidates.size()>limit) candidates.resize(limit);
        for(const auto& candidate:candidates) {
            staged.excludedTagGroups.insert(candidate);
            std::cout << "MARKER_MERGE_TAG_OUTLIER label=" << label
                      << " keyframe=" << candidate.first->mnId
                      << " marker=" << candidate.second << std::endl;
        }
    };
    excludeInitialTagOutliers(targetFrames,targetQuads,"target");
    excludeInitialTagOutliers(sourceFrames,sourceQuads,"source");
    result.graph=MarkerGraphOptimizer::RefineAndValidate(staged,options.optimizer);
    // A merge spans two independently culled maps and can expose one or two
    // stale keyframe-wide ORB groups. Retain those cameras and every marker
    // factor, exclude only their background pixels, and retry with the same
    // strict aggregate/per-keyframe validation. Bound removal to five percent
    // so this cannot manufacture agreement by discarding a bad submap.
    for(int retry=0; retry<3 && !result.graph.accepted &&
        result.graph.reason=="background_reprojection_validation_failed"; ++retry) {
        const std::size_t total=result.graph.after.backgroundRmsByKeyframe.size();
        const std::size_t maximumExcluded=std::max<std::size_t>(1,total/20);
        const std::size_t before=staged.excludedBackgroundKeyframes.size();
        for(const auto& residual:result.graph.after.backgroundRmsByKeyframe) {
            const auto initial=result.graph.before.backgroundRmsByKeyframe.find(residual.first);
            if(residual.second>options.optimizer.maximumBackgroundRmsPx ||
               initial==result.graph.before.backgroundRmsByKeyframe.end() ||
               residual.second>initial->second+
                   options.optimizer.maximumBackgroundRmsIncreasePx)
                staged.excludedBackgroundKeyframes.insert(residual.first);
        }
        std::cout << "MARKER_MERGE_BACKGROUND_RETRY retry=" << retry+1
                  << " excluded=" << staged.excludedBackgroundKeyframes.size()
                  << " total=" << total << " cap=" << maximumExcluded << std::endl;
        if(staged.excludedBackgroundKeyframes.size()==before ||
           staged.excludedBackgroundKeyframes.size()>maximumExcluded ||
           total<=staged.excludedBackgroundKeyframes.size()+2) break;
        result.graph=MarkerGraphOptimizer::RefineAndValidate(staged,options.optimizer);
    }
    if(!staged.excludedBackgroundKeyframes.empty()) {
        for(KeyFrame* keyframe:staged.excludedBackgroundKeyframes)
            result.graph.excludedBackgroundKeyFrameIds.push_back(keyframe->mnId);
        std::sort(result.graph.excludedBackgroundKeyFrameIds.begin(),
                  result.graph.excludedBackgroundKeyFrameIds.end());
    }
    if(!result.graph.accepted) return reject("joint_ba_"+result.graph.reason);

    QuadMap refinedQuads=mergedQuads;
    for(const auto& tag:result.graph.staticTags) {
        if(tag.second.size()!=4) return reject("joint_ba_invalid_static_marker_geometry");
        for(std::size_t corner=0;corner<4;++corner)
            refinedQuads[tag.first][corner]=tag.second[corner];
    }
    // A low global mean is not sufficient: no individual strong common marker
    // may be hidden by many good background points or by the robust loss.
    targetSamples.clear(); sourceSamples.clear();
    reason=collectEvidence(targetFrames,targetQuads,common,options,targetSamples,
                           &result.graph.keyframePoses,&refinedQuads);
    if(!reason.empty()) return reject("refined_target_"+reason);
    reason=collectEvidence(sourceFrames,sourceQuads,common,options,sourceSamples,
                           &result.graph.keyframePoses,&refinedQuads);
    if(!reason.empty()) return reject("refined_source_"+reason);
    if(target->mnRevision!=result.targetRevision || source->mnRevision!=result.sourceRevision)
        return reject("map_changed_during_proposal");
    result.accepted=true;
    result.reason="verified_common_marker_merge";
    return result;
}

MarkerMapMerge::Proposal MarkerMapMerge::Propose(Map* target, Map* source)
{
    return Propose(target,source,Options());
}

} // namespace ORB_SLAM3
