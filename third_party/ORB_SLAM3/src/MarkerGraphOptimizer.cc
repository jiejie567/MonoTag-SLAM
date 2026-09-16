#include "MarkerGraphOptimizer.h"
#include "BackgroundResidualGate.h"

#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "OptimizableTypes.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <iostream>
#include <limits>
#include <queue>
#include <utility>
#include <Eigen/SVD>

namespace ORB_SLAM3 {
namespace {

using MGO = MarkerGraphOptimizer;
template<class T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;
constexpr double kTagInformation = 64.0;
constexpr double kHuberPixels = 2.447651936;
// Address-keyed containers may serve lookup, never numerical graph ordering.
// Call only on validated, non-null members with their persistent SLAM IDs.
template<class Container>
std::vector<typename Container::value_type> orderedById(const Container& values)
{
    std::vector<typename Container::value_type> ordered(values.begin(),values.end());
    std::sort(ordered.begin(),ordered.end(),[](const auto* a,const auto* b) {
        return a->mnId<b->mnId;
    });
    return ordered;
}

bool finitePose(const Sophus::SE3f& pose)
{
    return pose.matrix().allFinite();
}

bool validOptions(const MGO::Options& options)
{
    return options.graphIterations > 0 && options.baIterations > 0 &&
        options.covisibleNeighbors >= 0 &&
        std::isfinite(options.maximumTagRmsPx) && options.maximumTagRmsPx > 0 &&
        std::isfinite(options.maximumBackgroundRmsPx) && options.maximumBackgroundRmsPx > 0 &&
        std::isfinite(options.maximumBackgroundRmsIncreasePx) && options.maximumBackgroundRmsIncreasePx >= 0 &&
        std::isfinite(options.minimumPositiveDepthFraction) && options.minimumPositiveDepthFraction > 0 &&
        options.minimumPositiveDepthFraction <= 1 &&
        std::isfinite(options.maximumAnchorTranslationM) && options.maximumAnchorTranslationM >= 0 &&
        std::isfinite(options.maximumAnchorRotationRad) && options.maximumAnchorRotationRad >= 0 &&
        std::isfinite(options.maximumScaleAnchorRelativeError) && options.maximumScaleAnchorRelativeError > 0 &&
        std::isfinite(options.maximumMarkerCornerDisplacementM) &&
        options.maximumMarkerCornerDisplacementM > 0 &&
        std::isfinite(options.minimumScale) && std::isfinite(options.maximumScale) &&
        options.minimumScale > 0 && options.minimumScale <= 1 && options.maximumScale >= 1;
}

struct TagObservation {
    KeyFrame* keyframe;
    int markerId;
    std::size_t index;
    Eigen::Vector3d point;
    Eigen::Vector2d pixel;
    double information;
};

struct BackgroundObservation {
    KeyFrame* keyframe;
    MapPoint* point;
    Eigen::Vector2d pixel;
    double information;
};

struct Observations {
    std::vector<TagObservation> tags;
    std::vector<BackgroundObservation> background;
    std::map<MapPoint*, std::size_t> pointCounts;
    std::map<int, std::set<KeyFrame*>> markerKeyframes;
};

const std::vector<Eigen::Vector3f>& tagCorners(KeyFrame* keyframe, const MGO::StagedBAInput& input)
{
    const auto replacement = input.tagWorldCorners.find(keyframe);
    return replacement == input.tagWorldCorners.end() ? keyframe->mvTagWorldPoints : replacement->second;
}

bool sameOrderedMarkerShape(const std::vector<Eigen::Vector3f>& corners, std::size_t start,
                            const std::vector<Eigen::Vector3f>& canonical)
{
    if(canonical.size()!=4 || start+4>corners.size()) return false;
    const auto a=canonical[1]-canonical[0], b=canonical[3]-canonical[0];
    if(!a.allFinite() || !b.allFinite() || a.cross(b).norm()<1e-6f) return false;
    // A decoded four-corner group's order is its physical corner identity.
    // Check the same ordered shape, not proximity in a superseded map gauge.
    for(std::size_t i=0;i<4;++i) for(std::size_t j=i+1;j<4;++j) {
        if(!corners[start+i].allFinite() || !corners[start+j].allFinite()) return false;
        const float expected=(canonical[i]-canonical[j]).norm();
        const float measured=(corners[start+i]-corners[start+j]).norm();
        if(!std::isfinite(expected) || expected<=1e-6f ||
           std::abs(measured-expected)>std::max(1e-5f,1e-3f*expected)) return false;
    }
    return true;
}

bool reliableTag(KeyFrame* keyframe, bool includeInactive = false)
{
    if(!keyframe || !keyframe->mbHasTagObservation ||
       (!includeInactive && !keyframe->mbTagObservationActive) ||
       !std::isfinite(keyframe->mTagObservationConfidence) || keyframe->mTagObservationConfidence < .35f ||
       keyframe->mvTagWorldPoints.size() < 4 ||
       keyframe->mvTagWorldPoints.size() != keyframe->mvTagImagePoints.size() ||
       keyframe->mvTagIds.size() != keyframe->mvTagWorldPoints.size() ||
       (!keyframe->mvTagPointWeights.empty() &&
        keyframe->mvTagPointWeights.size() != keyframe->mvTagWorldPoints.size()))
        return false;
    std::map<int, std::size_t> strong;
    for(std::size_t i = 0; i < keyframe->mvTagWorldPoints.size(); ++i) {
        if(!keyframe->mvTagWorldPoints[i].allFinite() ||
           !std::isfinite(keyframe->mvTagImagePoints[i].x) ||
           !std::isfinite(keyframe->mvTagImagePoints[i].y)) return false;
        const float weight = keyframe->mvTagPointWeights.empty() ? 1.0f : keyframe->mvTagPointWeights[i];
        if(!std::isfinite(weight) || weight <= 0 || weight > 1) return false;
        if(weight >= .99f) ++strong[keyframe->mvTagIds[i]];
    }
    for(const auto& value : strong) if(value.second >= 4) return true;
    return false;
}

bool registeredTagGeometry(KeyFrame* keyframe, Map* map)
{
    if(!reliableTag(keyframe)) return false;
    for(std::size_t i = 0; i < keyframe->mvTagWorldPoints.size(); ++i) {
        const auto tag = map->mStaticTags.find(keyframe->mvTagIds[i]);
        if(tag == map->mStaticTags.end() || tag->second.size() != 12) return false;
        bool found = false;
        for(std::size_t corner = 0; corner < 4; ++corner) {
            const Eigen::Vector3f registered(tag->second[3*corner], tag->second[3*corner+1], tag->second[3*corner+2]);
            if((registered-keyframe->mvTagWorldPoints[i]).norm() <= 1e-4f) found = true;
        }
        if(!found) return false;
    }
    return true;
}

std::map<int,KeyFrame*> provisionalMarkerReferences(
        Map* map, const std::vector<KeyFrame*>& keyframes, KeyFrame* acceptedAnchor=nullptr)
{
    std::map<int,KeyFrame*> provisional;
    if(!map || map->mbRigidMarkerLayout) return provisional;
    const auto allKeyframes=map->GetAllKeyFrames();
    if(!acceptedAnchor) {
        const long anchorId=map->GetMarkerScaleAnchorKFId();
        if(anchorId<0) acceptedAnchor=map->GetOriginKF();
        else for(KeyFrame* k:allKeyframes)
            if(k && !k->isBad() && long(k->mnId)==anchorId) {acceptedAnchor=k;break;}
    }
    // Missing accepted-anchor metadata is not permission to release old poses.
    if(!acceptedAnchor || acceptedAnchor->isBad() || acceptedAnchor->GetMap()!=map)
        return provisional;
    for(KeyFrame* k:keyframes) {
        if(!reliableTag(k) || k->isBad() || k->GetMap()!=map) continue;
        for(std::size_t i=0;i+3<k->mvTagIds.size();++i) {
            const int id=k->mvTagIds[i];
            bool strong=true;
            for(std::size_t j=i;j<i+4;++j)
                strong=strong && k->mvTagIds[j]==id &&
                    (k->mvTagPointWeights.empty() || k->mvTagPointWeights[j]>=.99f);
            if(!strong) continue;
            auto first=provisional.find(id);
            if(first==provisional.end() || k->mnFrameId<first->second->mnFrameId)
                provisional[id]=k;
            i+=3;
        }
    }
    // Registration supplies an initial pose, not a surveyed world position.
    // Only an accepted metric interval advances this boundary, consistently
    // for reanchor, visual-loop and final-map proposals. Check old observers
    // outside the selected path too; revisited markers remain protected.
    for(KeyFrame* k:allKeyframes) {
        if(provisional.empty()) break;
        if(!k || k->isBad() || k->GetMap()!=map ||
           k->mnFrameId>acceptedAnchor->mnFrameId || !reliableTag(k)) continue;
        for(int id:k->mvTagIds) provisional.erase(id);
    }
    return provisional;
}

bool prepare(const MGO::StagedBAInput& input, const MGO::Options& options,
             MGO::Proposal& proposal, Observations& observations)
{
    if(!validOptions(options)) {proposal.reason = "invalid_options"; return false;}
    if(input.keyframes.empty() || input.fixedKeyframes.empty()) {
        proposal.reason = "missing_fixed_gauge"; return false;
    }
    std::set<KeyFrame*> keyframes;
    for(KeyFrame* keyframe : input.keyframes) {
        if(!keyframe || keyframe->isBad() || !keyframe->mpCamera ||
           keyframe->mpCamera2 || keyframe->NLeft != -1 || !keyframes.insert(keyframe).second) {
            proposal.reason = "invalid_or_nonmonocular_keyframe"; return false;
        }
        const auto estimate = input.keyframePoses.find(keyframe);
        const Sophus::SE3f pose = estimate == input.keyframePoses.end() ? keyframe->GetPose() : estimate->second;
        if(!finitePose(pose)) {proposal.reason = "nonfinite_pose"; return false;}
        proposal.keyframePoses.emplace(keyframe, pose);
        proposal.replayScaleMultipliers[keyframe] = 1.0f;
    }
    for(KeyFrame* fixed : input.fixedKeyframes) {
        if(!keyframes.count(fixed)) {proposal.reason = "fixed_keyframe_not_in_graph"; return false;}
    }
    for(const auto& pose : input.keyframePoses) {
        if(!keyframes.count(pose.first)) {proposal.reason = "pose_not_in_graph"; return false;}
    }
    for(KeyFrame* excluded : input.excludedTagKeyframes) {
        if(!keyframes.count(excluded)) {
            proposal.reason = "excluded_tag_keyframe_not_in_graph"; return false;
        }
    }
    for(KeyFrame* excluded : input.excludedBackgroundKeyframes) {
        if(!keyframes.count(excluded)) {
            proposal.reason = "excluded_background_keyframe_not_in_graph"; return false;
        }
    }
    for(const auto& excluded:input.excludedBackgroundObservations)
        if(!keyframes.count(excluded.first) || !input.pointPositions.count(excluded.second)) {
            proposal.reason="excluded_background_pixel_not_in_graph"; return false;
        }
    for(const auto& scale : input.replayScaleMultipliers) {
        if(!keyframes.count(scale.first) || !std::isfinite(scale.second) ||
           scale.second < options.minimumScale || scale.second > options.maximumScale) {
            proposal.reason = "invalid_replay_scale"; return false;
        }
        proposal.replayScaleMultipliers[scale.first] = scale.second;
    }
    for(const auto& replacement : input.tagWorldCorners) {
        if(!keyframes.count(replacement.first) ||
           replacement.second.size() != replacement.first->mvTagWorldPoints.size()) {
            proposal.reason = "invalid_tag_corner_override"; return false;
        }
        for(const auto& point : replacement.second) if(!point.allFinite()) {
            proposal.reason = "nonfinite_tag_corner"; return false;
        }
    }
    const auto orderedKeyframes=orderedById(input.keyframes);
    std::map<KeyFrame*, std::set<int>> completeByKeyframe;
    std::map<KeyFrame*, std::vector<std::size_t>> completeWeakGroups;
    for(KeyFrame* keyframe : orderedKeyframes) {
        if(input.excludedTagKeyframes.count(keyframe)) continue;
        if(!keyframe->mbHasTagObservation ||
           (!keyframe->mbTagObservationActive && !input.includeInactiveTagObservations)) continue;
        const auto& corners = tagCorners(keyframe, input);
        if(corners.size() != keyframe->mvTagImagePoints.size() ||
           corners.size() != keyframe->mvTagIds.size() ||
           (!keyframe->mvTagPointWeights.empty() && keyframe->mvTagPointWeights.size() != corners.size()) ||
           !std::isfinite(keyframe->mTagObservationConfidence)) {
            proposal.reason = "invalid_tag_observation"; return false;
        }
        proposal.tagWorldCorners[keyframe] = corners;
        auto& completeStrongMarkers = completeByKeyframe[keyframe];
        for(std::size_t i = 0; i+3 < corners.size(); ++i) {
            const int id = keyframe->mvTagIds[i];
            if(keyframe->mvTagIds[i+1] != id || keyframe->mvTagIds[i+2] != id ||
               keyframe->mvTagIds[i+3] != id) continue;
            // More than four occurrences cannot identify one ordered group.
            if(std::count(keyframe->mvTagIds.begin(),keyframe->mvTagIds.end(),id)!=4) continue;
            if(input.excludedTagGroups.count({keyframe,id})) {i+=3; continue;}
            bool strong = keyframe->mTagObservationConfidence >= .35f;
            for(std::size_t corner = 0; corner < 4; ++corner) {
                const float weight = keyframe->mvTagPointWeights.empty() ? 1.0f :
                    keyframe->mvTagPointWeights[i+corner];
                strong = strong && std::isfinite(weight) && weight >= .99f && weight <= 1.f;
                if(!corners[i+corner].allFinite()) {
                    proposal.reason="invalid_tag_observation"; return false;
                }
            }
            if(!strong) {
                completeWeakGroups[keyframe].push_back(i);
                i+=3;
                continue;
            }
        std::vector<Eigen::Vector3f> geometry(
                corners.begin()+i, corners.begin()+i+4);
            const auto existing = proposal.staticTags.find(id);
            if(existing == proposal.staticTags.end()) proposal.staticTags.emplace(id, geometry);
            else for(std::size_t corner = 0; corner < 4; ++corner) {
                const float disagreement=(existing->second[corner]-geometry[corner]).norm();
                // A validated cross-map merge admits at most 6 mm of layout
                // disagreement. Build one rigid marker factor from the first
                // accepted geometry and canonicalize later observations only
                // inside this staged proposal. A larger disagreement remains
                // a hard conflict; live keyframe data is changed only if the
                // complete BA passes and is atomically committed.
                if(disagreement > 6.1e-3f) {
                    std::cout << "MARKER_GEOMETRY_CONFLICT marker=" << id
                              << " keyframe=" << keyframe->mnId
                              << " corner=" << corner
                              << " disagreement_m=" << disagreement << std::endl;
                    proposal.reason = "inconsistent_static_marker_geometry"; return false;
                }
                proposal.tagWorldCorners.at(keyframe)[i+corner]=existing->second[corner];
            }
            completeStrongMarkers.insert(id);
            i += 3;
        }
    }
    // Only strong observations establish geometry, in the supplied staged
    // gauge (tagCorners honors Sim3/merge overrides). Never substitute a live
    // map's old mStaticTags for a transformed proposal. Complete weak decoded
    // groups already identify all four physical corners; resolve by ID/order,
    // not nearest-point snapping or a relaxed strong-geometry distance gate.
    for(const auto& entry:completeWeakGroups) for(std::size_t start:entry.second) {
        auto& corners=proposal.tagWorldCorners.at(entry.first);
        const auto canonical=proposal.staticTags.find(entry.first->mvTagIds[start]);
        if(canonical==proposal.staticTags.end() || !options.canonicalizeWeakMarkerCorners) continue;
        if(!sameOrderedMarkerShape(corners,start,canonical->second)) {
            proposal.reason="inconsistent_weak_marker_shape"; return false;
        }
        std::copy(canonical->second.begin(),canonical->second.end(),corners.begin()+start);
    }
    // Partial groups do not establish corner order. Their existing bounded
    // nearest-corner rule remains unchanged, after collecting strong geometry.
    for(KeyFrame* keyframe : orderedKeyframes) {
        const auto found = proposal.tagWorldCorners.find(keyframe);
        if(found == proposal.tagWorldCorners.end()) continue;
        auto& canonicalCorners=found->second;
        const auto& completeStrongMarkers=completeByKeyframe.at(keyframe);
        for(std::size_t i = 0; i < canonicalCorners.size(); ++i) {
            if(input.excludedTagGroups.count({keyframe,keyframe->mvTagIds[i]})) continue;
            const auto& pixel = keyframe->mvTagImagePoints[i];
            const float weight = keyframe->mvTagPointWeights.empty() ? 1.0f : keyframe->mvTagPointWeights[i];
            if(!canonicalCorners[i].allFinite() || !std::isfinite(pixel.x) || !std::isfinite(pixel.y) ||
               !std::isfinite(weight) || weight <= 0 || weight > 1) {
                proposal.reason = "invalid_tag_observation"; return false;
            }
            // Partial observations used to keep cached-layout XYZ while full
            // decoded corners used the optimized rigid marker. Resolve only a
            // unique nearby physical corner, never one chosen by image error.
            const auto marker=proposal.staticTags.find(keyframe->mvTagIds[i]);
            // No strong observation in this staged gauge means no landmark.
            // A weak/partial group alone cannot create absolute pose or scale.
            if(marker==proposal.staticTags.end()) continue;
            if(options.canonicalizeWeakMarkerCorners && weight < .99f &&
               marker!=proposal.staticTags.end() && marker->second.size()==4) {
                float best=std::numeric_limits<float>::infinity(), second=best;
                int index=-1;
                float side=std::numeric_limits<float>::infinity();
                for(int j=0;j<4;++j) {
                    side=std::min(side,(marker->second[j]-marker->second[(j+1)%4]).norm());
                    const float distance=(canonicalCorners[i]-marker->second[j]).norm();
                    if(distance<best) {second=best; best=distance; index=j;}
                    else second=std::min(second,distance);
                }
                if(index>=0 && best<=std::min(6.1e-3f,.2f*side) && second-best>1e-4f) {
                    if(best>1e-5f)
                        std::cout << "MARKER_WEAK_CORNER_CANONICAL keyframe=" << keyframe->mnId
                                  << " marker=" << marker->first << " offset_m=" << best << std::endl;
                    canonicalCorners[i]=marker->second[index];
                }
            }
            observations.tags.push_back({keyframe, keyframe->mvTagIds[i], i,
                canonicalCorners[i].cast<double>(),
                Eigen::Vector2d(pixel.x, pixel.y), kTagInformation *
                std::max(.35f, keyframe->mTagObservationConfidence) * weight});
            // A weak/border-damaged corner remains a useful robust projection
            // factor for an already registered marker.  It must not create a
            // free marker-pose variable or a rigid relation by itself.
            if(completeStrongMarkers.count(keyframe->mvTagIds[i]))
                observations.markerKeyframes[keyframe->mvTagIds[i]].insert(keyframe);
        }
    }
    if(observations.tags.size() < 8) {proposal.reason = "insufficient_tag_constraints"; return false;}
    if(!input.rigidMarkerLayout && !options.legacyIndependentMarkerWorldPrior &&
       !options.fixAllObservedGaugeMarkers) {
        // A different unknown marker in each frame does NOT constrain the
        // background scale. Require shared rigid geometry and translation,
        // or an already fixed metric camera baseline at the window boundary.
        const auto hasBaseline=[&](const std::set<KeyFrame*>& views) {
            if(views.size()<2) return false;
            const auto center=proposal.keyframePoses.at(*views.begin()).inverse().translation();
            for(KeyFrame* k:views)
                if((proposal.keyframePoses.at(k).inverse().translation()-center).norm()>1e-4f)
                    return true;
            return false;
        };
        bool observable=hasBaseline(input.fixedKeyframes);
        for(const auto& marker:observations.markerKeyframes)
            observable=observable || hasBaseline(marker.second);
        if(!observable) {proposal.reason="insufficient_shared_marker_baseline"; return false;}
    }
    for(int id:input.provisionalMarkerIds) {
        const auto marker=observations.markerKeyframes.find(id);
        if(marker==observations.markerKeyframes.end()) {
            proposal.reason="unobserved_provisional_marker";return false;
        }
        for(KeyFrame* k:marker->second) if(input.fixedKeyframes.count(k)) {
            proposal.reason="fixed_gauge_marked_provisional";return false;
        }
    }

    for(const auto& point : input.pointPositions) {
        if(!point.first || point.first->isBad() || !point.second.allFinite()) {
            proposal.reason = "invalid_map_point"; return false;
        }
        proposal.pointPositions.emplace(point.first, point.second);
        for(const auto& observation : point.first->GetObservations()) {
            KeyFrame* keyframe = observation.first;
            if(!keyframes.count(keyframe)) continue;
            if(input.excludedBackgroundKeyframes.count(keyframe)) continue;
            if(input.excludedBackgroundObservations.count({keyframe,point.first})) continue;
            const int index = std::get<0>(observation.second);
            if(index < 0 || std::size_t(index) >= keyframe->mvKeysUn.size()) {
                proposal.reason = "invalid_background_observation"; return false;
            }
            const auto& keypoint = keyframe->mvKeysUn[index];
            if(keypoint.octave < 0 || std::size_t(keypoint.octave) >= keyframe->mvInvLevelSigma2.size()) {
                proposal.reason = "invalid_pyramid_observation"; return false;
            }
            const double information = keyframe->mvInvLevelSigma2[keypoint.octave];
            if(!std::isfinite(keypoint.pt.x) || !std::isfinite(keypoint.pt.y) ||
               !std::isfinite(information) || information <= 0) {
                proposal.reason = "invalid_background_observation"; return false;
            }
            if(input.filterInitialBackgroundOutliers ||
               (input.filterFixedBackgroundOutliers && input.fixedKeyframes.count(keyframe))) {
                const Eigen::Vector3d cameraPoint = input.useCommittedAdmission
                    ? (keyframe->GetPose().cast<double>() * point.first->GetWorldPos().cast<double>()).eval()
                    : (proposal.keyframePoses.at(keyframe).cast<double>() * point.second.cast<double>()).eval();
                if(!cameraPoint.allFinite() || cameraPoint.z() <= 1e-6) continue;
                const Eigen::Vector2d projected = keyframe->mpCamera->project(cameraPoint);
                const Eigen::Vector2d measured(keypoint.pt.x, keypoint.pt.y);
                // Same normalized 2-D 99% gate used for robust feature
                // admission; pixel tolerance naturally grows by octave.
                if(!projected.allFinite() ||
                   (projected-measured).squaredNorm()*information > 9.21034) continue;
            }
            const auto alias=input.pointAliases.find(point.first);
            MapPoint* target=alias==input.pointAliases.end()?point.first:alias->second;
            if(!input.pointPositions.count(target)) {proposal.reason="invalid_point_alias";return false;}
            observations.background.push_back({keyframe, target,
                Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y), information});
            ++observations.pointCounts[target];
        }
    }
    std::sort(observations.tags.begin(),observations.tags.end(),[](const auto& a,const auto& b) {
        return std::tie(a.keyframe->mnId,a.markerId,a.index)<
               std::tie(b.keyframe->mnId,b.markerId,b.index);
    });
    // Aliases can put several measured pixels on one point in a loop trial;
    // include pixel/information tie-breakers without dropping any observation.
    std::sort(observations.background.begin(),observations.background.end(),[](const auto& a,const auto& b) {
        return std::make_tuple(a.keyframe->mnId,a.point->mnId,a.pixel.x(),a.pixel.y(),a.information)<
               std::make_tuple(b.keyframe->mnId,b.point->mnId,b.pixel.x(),b.pixel.y(),b.information);
    });
    return true;
}

MGO::ResidualSummary residuals(const MGO::Proposal& proposal, const Observations& observations)
{
    MGO::ResidualSummary result;
    double tagSquared = 0, backgroundSquared = 0;
    std::map<KeyFrame*, std::pair<double, std::size_t>> perKeyframe;
    std::map<int, std::pair<double, std::size_t>> perMarker;
    std::map<std::pair<KeyFrame*, int>, std::pair<double, std::size_t>> perKeyframeMarker;
    std::map<KeyFrame*, std::pair<double, std::size_t>> backgroundPerKeyframe;
    std::map<KeyFrame*, double> backgroundNormalizedSquared;
    const auto error = [&proposal, &result](KeyFrame* keyframe, const Eigen::Vector3d& point,
                                          const Eigen::Vector2d& pixel) {
        const Eigen::Vector3d cameraPoint = proposal.keyframePoses.at(keyframe).cast<double>() * point;
        if(cameraPoint.allFinite() && cameraPoint.z() > 1e-6) ++result.positiveDepth;
        if(!cameraPoint.allFinite() || std::abs(cameraPoint.z()) <= 1e-9)
            return std::numeric_limits<double>::infinity();
        const auto projected = keyframe->mpCamera->project(cameraPoint);
        return projected.allFinite() ? (projected-pixel).squaredNorm() : std::numeric_limits<double>::infinity();
    };
    for(const auto& observation : observations.tags) {
        const double squared = error(
            observation.keyframe,
            proposal.tagWorldCorners.at(observation.keyframe).at(observation.index).cast<double>(),
            observation.pixel);
        tagSquared += squared;
        auto& keyframe = perKeyframe[observation.keyframe];
        keyframe.first += squared; ++keyframe.second;
        auto& marker = perMarker[observation.markerId];
        marker.first += squared; ++marker.second;
        auto& keyframeMarker = perKeyframeMarker[{observation.keyframe, observation.markerId}];
        keyframeMarker.first += squared; ++keyframeMarker.second;
        ++result.tagCorners;
    }
    for(const auto& observation : observations.background) {
        const double squared = error(observation.keyframe, proposal.pointPositions.at(observation.point).cast<double>(), observation.pixel);
        backgroundSquared += squared;
        auto& keyframe = backgroundPerKeyframe[observation.keyframe];
        keyframe.first += squared; ++keyframe.second;
        backgroundNormalizedSquared[observation.keyframe]+=squared*observation.information;
        ++result.backgroundObservations;
    }
    result.tagRmsPx = result.tagCorners ? std::sqrt(tagSquared/result.tagCorners) : 0;
    result.backgroundRmsPx = result.backgroundObservations ? std::sqrt(backgroundSquared/result.backgroundObservations) : 0;
    const std::size_t count = result.tagCorners + result.backgroundObservations;
    result.positiveDepthFraction = count ? double(result.positiveDepth)/count : 0;
    for(const auto& keyframe : perKeyframe)
        result.tagRmsByKeyframe[keyframe.first] = std::sqrt(keyframe.second.first/keyframe.second.second);
    for(const auto& marker : perMarker)
        result.tagRmsByMarker[marker.first] = std::sqrt(marker.second.first/marker.second.second);
    for(const auto& marker : perKeyframeMarker)
        result.tagRmsByKeyframeMarker[marker.first] =
            std::sqrt(marker.second.first/marker.second.second);
    for(const auto& keyframe : backgroundPerKeyframe) {
        result.backgroundRmsByKeyframe[keyframe.first] = std::sqrt(keyframe.second.first/keyframe.second.second);
        result.backgroundNormalizedRmsByKeyframe[keyframe.first]=
            std::sqrt(backgroundNormalizedSquared.at(keyframe.first)/keyframe.second.second);
    }
    return result;
}

// Read-only frozen-loop diagnostics; never changes observations or gates.
void traceLoopStage(const char* stage, const MGO::Proposal& proposal,
                    const Observations& observations, KeyFrame* origin, std::size_t aliases)
{
    const char* path=std::getenv("MARKER_LOOP_STAGE_DUMP");
    if(!path) return;
    std::ofstream out(path,std::ios::app);
    if(!out.good()) return;
    out << std::setprecision(12);
    const auto number=[&](double x) {if(std::isfinite(x)) out<<x; else out<<"null";};
    const auto summary=residuals(proposal,observations);
    std::map<std::pair<KeyFrame*,int>,double> minDepth;
    std::size_t fixedTags=0,fixedBackground=0;
    for(const auto& o:observations.tags) {
        const double z=(proposal.keyframePoses.at(o.keyframe).cast<double>()*
                       proposal.tagWorldCorners.at(o.keyframe).at(o.index).cast<double>()).z();
        const auto key=std::make_pair(o.keyframe,o.markerId);
        if(!minDepth.count(key)) minDepth[key]=z;else minDepth[key]=std::min(minDepth[key],z);
        fixedTags+=o.keyframe==origin;
    }
    for(const auto& o:observations.background) fixedBackground+=o.keyframe==origin;
    out << "{\"type\":\"stage\",\"stage\":\"" << stage << "\",\"tag_count\":" << summary.tagCorners
        << ",\"background_count\":" << summary.backgroundObservations << ",\"aliases\":" << aliases
        << ",\"fixed_tag_count\":" << fixedTags << ",\"fixed_background_count\":" << fixedBackground
        << ",\"tag_rms\":";number(summary.tagRmsPx);out << ",\"background_rms\":";number(summary.backgroundRmsPx);
    out << ",\"positive_depth_fraction\":";number(summary.positiveDepthFraction);
    out << ",\"gauge_matrix_delta\":";
    number((proposal.keyframePoses.at(origin).matrix()-origin->GetPose().matrix()).norm());out << "}\n";
    for(const auto& item:summary.tagRmsByKeyframeMarker) {
        out << "{\"type\":\"tag\",\"stage\":\"" << stage << "\",\"kf\":" << item.first.first->mnId
            << ",\"time\":" << item.first.first->mTimeStamp << ",\"marker\":" << item.first.second
            << ",\"rms\":";number(item.second);out << ",\"min_z\":";number(minDepth.at(item.first));
        const auto center=proposal.keyframePoses.at(item.first.first).inverse().translation();
        out << ",\"camera_center\":[" << center.x() << ',' << center.y() << ',' << center.z() << "]}\n";
    }
    for(const auto& item:summary.backgroundRmsByKeyframe) {
        out << "{\"type\":\"background\",\"stage\":\"" << stage << "\",\"kf\":" << item.first->mnId
            << ",\"rms\":";number(item.second);out << ",\"normalized_rms\":";
        number(summary.backgroundNormalizedRmsByKeyframe.at(item.first));out << "}\n";
    }
}

bool validate(MGO::Proposal& proposal, const Observations& observations,
              const MGO::Options& options, const MGO::PoseMap& fixedPoses,
              bool finalBackgroundPolicy = false)
{
    for(const auto& pose : proposal.keyframePoses) if(!finitePose(pose.second)) {
        proposal.reason = "nonfinite_optimized_pose"; return false;
    }
    for(const auto& point : proposal.pointPositions) if(!point.second.allFinite()) {
        proposal.reason = "nonfinite_optimized_point"; return false;
    }
    proposal.after = residuals(proposal, observations);
    if(!std::isfinite(proposal.after.tagRmsPx) || !std::isfinite(proposal.after.backgroundRmsPx) ||
       proposal.after.positiveDepthFraction < options.minimumPositiveDepthFraction) {
        proposal.reason = "positive_depth_validation_failed"; return false;
    }
    // All fixed marker corners must remain in front of their observing camera;
    // the small background outlier allowance must not hide a flipped tag.
    for(const auto& observation : observations.tags) {
        const Eigen::Vector3d point = proposal.tagWorldCorners.at(
            observation.keyframe).at(observation.index).cast<double>();
        if((proposal.keyframePoses.at(observation.keyframe).cast<double>() * point).z() <= 1e-6) {
            proposal.reason = "tag_behind_camera"; return false;
        }
    }
    for(const auto& value : proposal.after.tagRmsByKeyframe) {
        if(value.second > options.maximumTagRmsPx) {
            proposal.reason = "tag_reprojection_validation_failed"; return false;
        }
    }
    // A bad marker must not be hidden by several good markers in the same
    // keyframe. The caller may retry after excluding that whole observation
    // keyframe, preserving the atomic raw-corner group contract.
    for(const auto& value : proposal.after.tagRmsByKeyframeMarker) {
        if(value.second > options.maximumTagRmsPx) {
            proposal.reason = "tag_reprojection_validation_failed"; return false;
        }
    }
    if(proposal.after.backgroundRmsPx > options.maximumBackgroundRmsPx ||
       proposal.after.backgroundRmsPx > proposal.before.backgroundRmsPx + options.maximumBackgroundRmsIncreasePx) {
        std::cout << "MARKER_BACKGROUND_GATE scope=aggregate before=" << proposal.before.backgroundRmsPx
                  << " after=" << proposal.after.backgroundRmsPx << std::endl;
        proposal.reason = "background_reprojection_validation_failed"; return false;
    }
    // A large healthy map must not dilute one damaged source frame's residual.
    // A frame that was healthy before BA must remain under the absolute gate.
    // A pre-existing high-residual frame is grandfathered only when BA does
    // not materially worsen it; otherwise one stale native association makes
    // every later full-map refinement impossible even when aggregate marker
    // and background errors both improve.
    for(const auto& value : proposal.after.backgroundNormalizedRmsByKeyframe) {
        const auto before = proposal.before.backgroundNormalizedRmsByKeyframe.find(value.first);
        if(before == proposal.before.backgroundNormalizedRmsByKeyframe.end() ||
           !(finalBackgroundPolicy ? FinalBackgroundResidualFrameConsistent(before->second,value.second)
                                   : BackgroundResidualFrameConsistent(before->second,value.second))) {
            const auto rawBefore=proposal.before.backgroundRmsByKeyframe.find(value.first);
            const auto rawAfter=proposal.after.backgroundRmsByKeyframe.find(value.first);
            std::cout << "MARKER_BACKGROUND_GATE scope=keyframe id=" << value.first->mnId
                      << " timestamp=" << value.first->mTimeStamp
                      << " normalized_before=" << (before==proposal.before.backgroundNormalizedRmsByKeyframe.end()?-1:before->second)
                      << " normalized_after=" << value.second
                      << " raw_before=" << (rawBefore==proposal.before.backgroundRmsByKeyframe.end()?-1:rawBefore->second)
                      << " raw_after=" << (rawAfter==proposal.after.backgroundRmsByKeyframe.end()?-1:rawAfter->second) << std::endl;
            double normalizedSquared=0.; std::size_t count=0, outliers=0;
            for(const auto& observation:observations.background) {
                if(observation.keyframe!=value.first) continue;
                const Eigen::Vector3d point=proposal.keyframePoses.at(value.first).cast<double>()*
                    proposal.pointPositions.at(observation.point).cast<double>();
                const double squared=(value.first->mpCamera->project(point)-observation.pixel).squaredNorm();
                normalizedSquared+=squared*observation.information;
                ++count; if(squared*observation.information>9.21034) ++outliers;
            }
            std::cout << "MARKER_BACKGROUND_GATE_DETAIL id=" << value.first->mnId
                      << " observations=" << count << " normalized_rms="
                      << std::sqrt(normalizedSquared/std::max<std::size_t>(1,count))
                      << " chi2_outliers=" << outliers << std::endl;
            proposal.reason = "background_reprojection_validation_failed"; return false;
        }
    }
    for(const auto& fixed : fixedPoses) {
        const auto& candidate = proposal.keyframePoses.at(fixed.first);
        const double translation = (candidate.inverse().translation()-fixed.second.inverse().translation()).norm();
        const double rotation = (candidate.so3()*fixed.second.so3().inverse()).log().norm();
        if(translation > options.maximumAnchorTranslationM || rotation > options.maximumAnchorRotationRad) {
            proposal.reason = "fixed_anchor_changed"; return false;
        }
    }
    return true;
}

template<class Edge>
void robustify(Edge* edge, double delta)
{
    auto* kernel = new g2o::RobustKernelHuber;
    kernel->setDelta(delta);
    edge->setRobustKernel(kernel);
}

std::size_t repairBackgroundCheirality(
        const MGO::PoseMap& initialPoses, const Observations& observations,
        const std::map<KeyFrame*,g2o::VertexSE3Expmap*>& cameras,
        const std::map<MapPoint*,g2o::VertexSBAPointXYZ*>& points)
{
    std::map<MapPoint*,std::vector<const BackgroundObservation*>> groups;
    for(const auto& observation:observations.background)
        if(points.count(observation.point)) groups[observation.point].push_back(&observation);
    std::vector<MapPoint*> ordered;
    for(const auto& item:groups) ordered.push_back(item.first);
    std::size_t negative=0,repaired=0,unobservable=0,invalid=0,noGain=0;
    double repairedCostBefore=0,repairedCostAfter=0;
    g2o::RobustKernelHuber kernel;
    kernel.setDelta(kHuberPixels);
    for(MapPoint* point:orderedById(ordered)) {
        const auto& views=groups.at(point);
        const Eigen::Vector3d previous=points.at(point)->estimate();
        if(!previous.allFinite()) continue;
        bool behind=false;
        for(const auto* view:views)
            behind=behind || cameras.at(view->keyframe)->estimate().map(previous).z()<=0;
        // This is a cheirality repair, not a general replacement of positive
        // points or of difficult raw observations. Single-view points have no
        // optimized vertex and never enter this operation.
        if(!behind) continue;
        ++negative;
        AlignedVector<Eigen::Vector3d> rays,oldWorldRays,newWorldRays;
        bool finite=true;
        for(const auto* view:views) {
            const Eigen::Vector3d ray=view->keyframe->mpCamera->unprojectEig(
                cv::Point2f(view->pixel.x(),view->pixel.y())).cast<double>();
            if(!ray.allFinite() || ray.norm()<=1e-12 || std::abs(ray.z())<=1e-12) {
                finite=false;break;
            }
            rays.push_back(ray/ray.z());
            oldWorldRays.push_back(initialPoses.at(view->keyframe).cast<double>().so3().inverse()*ray.normalized());
            newWorldRays.push_back(cameras.at(view->keyframe)->estimate().rotation().inverse()*ray.normalized());
            finite=finite && oldWorldRays.back().allFinite() && newWorldRays.back().allFinite();
        }
        if(!finite) {++invalid;continue;}
        bool parallax=false;
        for(std::size_t a=0;a<views.size() && !parallax;++a)
            for(std::size_t b=a+1;b<views.size();++b) {
                if(views[a]->keyframe->mnFrameId==views[b]->keyframe->mnFrameId) continue;
                const double oldCos=oldWorldRays[a].dot(oldWorldRays[b]);
                const double newCos=newWorldRays[a].dot(newWorldRays[b]);
                if(oldCos>0 && oldCos<.9998 && newCos>0 && newCos<.9998) {parallax=true;break;}
            }
        if(!parallax) {++unobservable;continue;}
        const Eigen::Vector3d centre=cameras.at(views.front()->keyframe)->estimate().inverse().translation();
        Eigen::MatrixXd design(2*views.size(),4);
        for(std::size_t i=0;i<views.size();++i) {
            const auto& pose=cameras.at(views[i]->keyframe)->estimate();
            Eigen::Matrix<double,3,4> projection;
            projection.leftCols<3>()=pose.rotation().toRotationMatrix();
            projection.col(3)=pose.translation()+projection.leftCols<3>()*centre;
            const double weight=std::sqrt(views[i]->information);
            design.row(2*i)=weight*(rays[i].x()*projection.row(2)-projection.row(0));
            design.row(2*i+1)=weight*(rays[i].y()*projection.row(2)-projection.row(1));
        }
        if(!design.allFinite()) {++invalid;continue;}
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(design,Eigen::ComputeFullV);
        const Eigen::Vector4d h=svd.matrixV().col(3);
        if(!h.allFinite() || std::abs(h.w())<=1e-12 || svd.singularValues()(2)<=1e-12) {
            ++invalid;continue;
        }
        const Eigen::Vector3d candidate=h.head<3>()/h.w()+centre;
        if(!candidate.allFinite()) {++invalid;continue;}
        const auto cost=[&](const Eigen::Vector3d& world,bool requirePositive) {
            double total=0;
            for(const auto* view:views) {
                const Eigen::Vector3d camera=cameras.at(view->keyframe)->estimate().map(world);
                if(!camera.allFinite() || std::abs(camera.z())<=1e-9 ||
                   (requirePositive && camera.z()<=1e-6)) return std::numeric_limits<double>::infinity();
                const Eigen::Vector2d error=view->keyframe->mpCamera->project(camera)-view->pixel;
                const double chi2=view->information*error.squaredNorm();
                if(!std::isfinite(chi2)) return std::numeric_limits<double>::infinity();
                Eigen::Vector3d rho; kernel.robustify(chi2,rho);
                total+=rho[0];
            }
            return total;
        };
        const double before=cost(previous,false),after=cost(candidate,true);
        if(!std::isfinite(before) || !std::isfinite(after)) {++invalid;continue;}
        if(!(after<before)) {++noGain;continue;}
        points.at(point)->setEstimate(candidate);
        ++repaired;repairedCostBefore+=before;repairedCostAfter+=after;
    }
    if(negative)
        std::cout << "MARKER_BA_CHEIRALITY_REPAIR negative=" << negative << " repaired=" << repaired
                  << " unobservable=" << unobservable << " invalid=" << invalid << " no_gain=" << noGain
                  << " point_objective_before=" << repairedCostBefore << " point_objective_after=" << repairedCostAfter
                  << " retained_observations=" << observations.background.size() << std::endl;
    return repaired;
}


class MarkerPoseProjectionEdge : public g2o::BaseBinaryEdge<
        2, Eigen::Vector2d, g2o::VertexSE3Expmap, g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d initialWorldPoint;
    GeometricCamera* camera = nullptr;
    void computeError() override {
        const auto* cameraPose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const auto* markerDelta = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
        const Eigen::Vector3d point = cameraPose->estimate().map(
            markerDelta->estimate().map(initialWorldPoint));
        if(!point.allFinite() || std::abs(point.z()) < 1e-9) {
            _error.setConstant(1e6); return;
        }
        _error = _measurement-camera->project(point);
    }
    bool read(std::istream&) override {return false;}
    bool write(std::ostream&) const override {return false;}
};

class FixedTagSim3Edge : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSim3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d worldPoint;
    GeometricCamera* camera = nullptr;
    void computeError() override {
        const auto* vertex = static_cast<const g2o::VertexSim3Expmap*>(_vertices[0]);
        const Eigen::Vector3d point = vertex->estimate().map(worldPoint);
        if(!point.allFinite() || std::abs(point.z()) < 1e-9) {_error.setConstant(1e6); return;}
        _error = _measurement-camera->project(point);
    }
    bool read(std::istream&) override {return false;}
    bool write(std::ostream&) const override {return false;}
};

class LogScaleEdge : public g2o::BaseUnaryEdge<1, double, g2o::VertexSim3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const auto* vertex = static_cast<const g2o::VertexSim3Expmap*>(_vertices[0]);
        _error[0] = std::log(vertex->estimate().scale())-_measurement;
    }
    void linearizeOplus() override {
        _jacobianOplusXi.setZero();
        _jacobianOplusXi(0, 6) = 1.0;
    }
    bool read(std::istream&) override {return false;}
    bool write(std::ostream&) const override {return false;}
};

bool addTreePath(KeyFrame* first, KeyFrame* second, Map* map, std::set<KeyFrame*>& selected)
{
    std::vector<KeyFrame*> firstChain;
    std::set<KeyFrame*> firstAncestors;
    for(KeyFrame* keyframe = first; keyframe; keyframe = keyframe->GetParent()) {
        if(keyframe->isBad() || keyframe->GetMap() != map || !firstAncestors.insert(keyframe).second) return false;
        firstChain.push_back(keyframe);
    }
    std::vector<KeyFrame*> secondChain;
    std::set<KeyFrame*> seen;
    KeyFrame* common = second;
    while(common && !firstAncestors.count(common)) {
        if(common->isBad() || common->GetMap() != map || !seen.insert(common).second) return false;
        secondChain.push_back(common);
        common = common->GetParent();
    }
    if(!common) return false;
    for(KeyFrame* keyframe : firstChain) {
        selected.insert(keyframe);
        if(keyframe == common) break;
    }
    selected.insert(secondChain.begin(), secondChain.end());
    return true;
}

} // namespace

static MarkerGraphOptimizer::CornerScaleEvidence EstimateCornerScaleImpl(
        const std::vector<KeyFrame*>& input, bool initialization)
{
    struct View {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        KeyFrame* k; Eigen::Matrix<double,3,4> shape;
        Eigen::Matrix<double,2,4> pixels;
    };
    using ViewVector = AlignedVector<View>;
    std::map<int,ViewVector> groups;
    std::set<unsigned long> frames;
    for(KeyFrame* k:input) {
        if(!reliableTag(k,initialization) || k->isBad() || !k->mpCamera || !finitePose(k->GetPose()) ||
           !frames.insert(k->mnFrameId).second) continue;
        for(std::size_t i=0;i+3<k->mvTagIds.size();i+=4) {
            bool strong=true; View v; v.k=k;
            for(int j=0;j<4;++j) {
                strong=strong && k->mvTagIds[i+j]==k->mvTagIds[i] &&
                    (k->mvTagPointWeights.empty() || k->mvTagPointWeights[i+j]>=.99f);
                v.shape.col(j)=k->mvTagWorldPoints[i+j].cast<double>();
                v.pixels.col(j)<<k->mvTagImagePoints[i+j].x,k->mvTagImagePoints[i+j].y;
            }
            if(strong) groups[k->mvTagIds[i]].push_back(v);
        }
    }
    MarkerGraphOptimizer::CornerScaleEvidence result;
    std::vector<double> estimates,uncertainties;
    for(auto& item:groups) {
        auto& views=item.second;
        std::sort(views.begin(),views.end(),[](const View& a,const View& b){return a.k->mnFrameId<b.k->mnFrameId;});
        if(views.size()<3) continue;
        // Bounded fitting cost: retain spread views, not just a dense burst.
        if(views.size()>12) {
            ViewVector spread;
            for(int i=0;i<12;++i) spread.push_back(views[i*(views.size()-1)/11]);
            views=std::move(spread);
        }
        const Eigen::Vector3d origin=views.front().k->GetCameraCenter().cast<double>();
        const auto fit=[&](int excluded, Eigen::Matrix4d& transform) {
            Eigen::Matrix<double,3,4> triangulated;
            for(int corner=0;corner<4;++corner) {
                Eigen::MatrixXd A(2*(views.size()-(excluded>=0?1:0)),4);
                int row=0;
                for(std::size_t i=0;i<views.size();++i) {
                    if(int(i)==excluded) continue;
                    const auto& v=views[i];
                    const Eigen::Vector3d ray=v.k->mpCamera->unprojectEig(cv::Point2f(
                        v.pixels(0,corner),v.pixels(1,corner))).cast<double>();
                    Eigen::Matrix<double,3,4> P=v.k->GetPose().cast<double>().matrix3x4();
                    P.col(3)+=P.leftCols<3>()*origin;
                    A.row(row++)=ray.x()/ray.z()*P.row(2)-P.row(0);
                    A.row(row++)=ray.y()/ray.z()*P.row(2)-P.row(1);
                }
                const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeFullV);
                const Eigen::Vector4d h=svd.matrixV().col(3);
                if(!h.allFinite() || std::abs(h.w())<1e-9) return false;
                triangulated.col(corner)=h.head<3>()/h.w()+origin;
            }
            transform=Eigen::umeyama(views.front().shape,triangulated,true);
            const double s=std::cbrt(transform.topLeftCorner<3,3>().determinant());
            if(!transform.allFinite() || s<=0 || !std::isfinite(s)) return false;
            // The four reconstructed points must support one rigid square,
            // not four unrelated depths that happen to average to a size.
            const Eigen::Matrix<double,3,4> rigid=
                (transform.topLeftCorner<3,3>()*views.front().shape).colwise()+transform.topRightCorner<3,1>();
            const double edge=(triangulated.col(0)-triangulated.col(1)).norm();
            return edge>1e-6 && (rigid-triangulated).norm()/2<.05*edge;
        };
        const auto rms=[&](const Eigen::Matrix4d& transform,int only) {
            double squared=0;int count=0;
            for(std::size_t i=0;i<views.size();++i) {
                if(only>=0 && int(i)!=only) continue;
                for(int j=0;j<4;++j) {
                    const Eigen::Vector3d p=views[i].k->GetPose().cast<double>()*
                        (transform.topLeftCorner<3,3>()*views.front().shape.col(j)+transform.topRightCorner<3,1>());
                    if(!p.allFinite() || p.z()<=1e-6) return std::numeric_limits<double>::infinity();
                    squared+=(views[i].k->mpCamera->project(p)-views[i].pixels.col(j)).squaredNorm();++count;
                }
            }
            return std::sqrt(squared/count);
        };
        Eigen::Matrix4d fitted;
        if(!fit(-1,fitted)) continue;
        const double scale=1/std::cbrt(fitted.topLeftCorner<3,3>().determinant());
        double baseline=0;
        for(const auto& a:views) for(const auto& b:views)
            baseline=std::max(baseline,double((a.k->GetCameraCenter()-b.k->GetCameraCenter()).norm())*scale);
        if(scale<(initialization?1e-4:.05) || scale>(initialization?1000.:20.) ||
           baseline<.04 || rms(fitted,-1)>2.5) continue;
        if(initialization || scale<.5 || scale>2.) {
            // A large repair needs observed depth, not just a translated
            // camera and a numerically stable fit of a distant tiny square.
            // Use the same actual world-ray parallax bound as the final
            // background depth check. Every physical corner must support it.
            bool observable=true;
            for(int corner=0;corner<4 && observable;++corner) {
                AlignedVector<Eigen::Vector3d> rays;
                for(const auto& view:views) {
                    const Eigen::Vector3d ray=view.k->mpCamera->unprojectEig(cv::Point2f(
                        view.pixels(0,corner),view.pixels(1,corner))).cast<double>();
                    if(!ray.allFinite() || ray.norm()<=1e-9) {observable=false;break;}
                    rays.push_back(view.k->GetPose().cast<double>().so3().inverse()*ray.normalized());
                }
                bool parallax=false;
                for(std::size_t i=0;i<rays.size();++i) for(std::size_t j=i+1;j<rays.size();++j) {
                    const double cosine=rays[i].dot(rays[j]);
                    if(cosine>0 && cosine<.9998) parallax=true;
                }
                observable=observable && parallax;
            }
            if(!observable) continue;
        }
        double sigma=.005;bool valid=true;
        for(int withheld:{0,int(views.size()-1)}) {
            Eigen::Matrix4d split;
            if(!fit(withheld,split) || rms(split,withheld)>2.5) {valid=false;break;}
            const double splitScale=1/std::cbrt(split.topLeftCorner<3,3>().determinant());
            sigma=std::max(sigma,std::abs(std::log(splitScale/scale)));
        }
        if(!valid || sigma>.10) continue;
        estimates.push_back(std::log(scale));uncertainties.push_back(sigma);
        result.rms=std::max(result.rms,rms(fitted,-1));
    }
    if(estimates.empty()) return result;
    std::sort(estimates.begin(),estimates.end());
    const double middle=estimates[estimates.size()/2];
    result.scale=std::exp(middle);result.sigma=.005;result.markers=estimates.size();
    for(double value:estimates) result.sigma=std::max(result.sigma,std::abs(value-middle));
    for(double value:uncertainties) result.sigma=std::max(result.sigma,value);
    // Extreme repairs require two independent physical markers; a single
    // small planar target never bypasses the ordinary correction envelope.
    result.valid=result.sigma<=.10 &&
        (initialization || (result.scale>=.5 && result.scale<=2) || result.markers>=2);
    return result;
}

MarkerGraphOptimizer::CornerScaleEvidence MarkerGraphOptimizer::EstimateCornerScale(
        const std::vector<KeyFrame*>& input)
{
    return EstimateCornerScaleImpl(input,false);
}

MarkerGraphOptimizer::CornerScaleEvidence MarkerGraphOptimizer::EstimateInitialCornerScale(
        const std::vector<KeyFrame*>& input)
{
    return EstimateCornerScaleImpl(input,true);
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::RefineAndValidate(const StagedBAInput& input)
{
    return RefineAndValidate(input, Options());
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::RefineAndValidate(const StagedBAInput& input,
                                                                      const Options& options,
                                                                      const ResidualSummary* committedBaseline)
{
    return RefineAndValidate(input, options, committedBaseline, false);
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::RefineAndValidate(const StagedBAInput& input,
                                                                      const Options& options,
                                                                      const ResidualSummary* committedBaseline,
                                                                      bool observerRelativeMarkerGate)
{
    return RefineAndValidate(input, options, committedBaseline, observerRelativeMarkerGate, false);
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::RefineAndValidate(const StagedBAInput& input,
                                                                      const Options& options,
                                                                      const ResidualSummary* committedBaseline,
                                                                      bool observerRelativeMarkerGate,
                                                                      bool finalBackgroundPolicy)
{
    Proposal proposal;
    Observations observations;
    if(!prepare(input, options, proposal, observations)) return proposal;
    // A Sim3 seed is an optimizer initialization, not the committed map.
    // Acceptance and reported before-errors must use the same baseline.
    proposal.before = committedBaseline ? *committedBaseline : residuals(proposal, observations);
    const PoseMap initialPoses = proposal.keyframePoses;
    PoseMap fixedPoses;
    for(KeyFrame* fixed : input.fixedKeyframes) fixedPoses.emplace(fixed, proposal.keyframePoses.at(fixed));

    g2o::SparseOptimizer optimizer;
    auto* linear = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(new g2o::BlockSolver_6_3(linear)));
    optimizer.setVerbose(false);
    std::map<KeyFrame*, g2o::VertexSE3Expmap*> vertices;
    int nextId = 0;
    for(KeyFrame* keyframe : orderedById(input.keyframes)) {
        auto* vertex = new g2o::VertexSE3Expmap;
        const auto pose = proposal.keyframePoses.at(keyframe).cast<double>();
        vertex->setEstimate(g2o::SE3Quat(pose.unit_quaternion(), pose.translation()));
        vertex->setId(nextId++);
        vertex->setFixed(input.fixedKeyframes.count(keyframe));
        optimizer.addVertex(vertex);
        vertices[keyframe] = vertex;
    }
    std::map<int, g2o::VertexSE3Expmap*> markerVertices;
    g2o::VertexSE3Expmap* rigidMarkerVertex = nullptr;
    // Legacy A/B selection only. The camera boundary already fixes SE(3)
    // gauge; also locking its first unsurveyed marker freezes a noisy pose
    // measurement rather than merely choosing the coordinate system.
    std::pair<unsigned long,int> gaugeMarker(std::numeric_limits<unsigned long>::max(),-1);
    for(const auto& marker:observations.markerKeyframes)
        for(KeyFrame* observer:marker.second)
            if(input.fixedKeyframes.count(observer))
                gaugeMarker=std::min(gaugeMarker,std::make_pair(observer->mnId,marker.first));
    for(const auto& marker : proposal.staticTags) {
        const auto observers = observations.markerKeyframes.find(marker.first);
        if(marker.second.size() != 4 || observers == observations.markerKeyframes.end() ||
           observers->second.empty()) continue;
        // A newly registered one-view marker has an uncertain world pose,
        // not a surveyed absolute position. Give it a rigid pose variable;
        // this preserves its physical size without pinning the camera to a
        // provisional layout. Even a marker seen at the origin is unsurveyed.
        auto* vertex = input.rigidMarkerLayout ? rigidMarkerVertex : nullptr;
        if(!vertex) {
            vertex = new g2o::VertexSE3Expmap;
            vertex->setEstimate(g2o::SE3Quat());
            vertex->setId(nextId++);
            optimizer.addVertex(vertex);
            if(input.rigidMarkerLayout) rigidMarkerVertex = vertex;
        }
        // Preserve the existing explicitly configured rigid-board mode and
        // diagnostic legacy policies. Independent marker poses remain free.
        for(KeyFrame* observer : observers->second)
            if(input.fixedKeyframes.count(observer) &&
               (input.rigidMarkerLayout || options.fixAllObservedGaugeMarkers ||
                (options.legacyIndependentMarkerWorldPrior && marker.first==gaugeMarker.second))) {
                vertex->setFixed(true);
                break;
            }
        markerVertices[marker.first] = vertex;
    }
    // In independent-marker mode, co-visibility is evidence through the same
    // camera pose and raw corner projections; it is not a rigid prior between
    // different IDs.  Calibrated boards use one shared marker-pose vertex via
    // rigidMarkerLayout above, which is the only mode that fixes inter-marker
    // layout.
    for(const auto& observation : observations.tags) {
        const auto marker = markerVertices.find(observation.markerId);
        if(marker == markerVertices.end()) {
            auto* edge = new EdgeSE3ProjectXYZOnlyPose;
            edge->setVertex(0, vertices.at(observation.keyframe));
            edge->setMeasurement(observation.pixel);
            edge->setInformation(Eigen::Matrix2d::Identity()*observation.information);
            edge->Xw = observation.point;
            edge->pCamera = observation.keyframe->mpCamera;
            robustify(edge, kHuberPixels*std::sqrt(observation.information));
            optimizer.addEdge(edge);
        } else {
            auto* edge = new MarkerPoseProjectionEdge;
            edge->setVertex(0, vertices.at(observation.keyframe));
            edge->setVertex(1, marker->second);
            edge->setMeasurement(observation.pixel);
            edge->setInformation(Eigen::Matrix2d::Identity()*observation.information);
            edge->initialWorldPoint = observation.point;
            edge->camera = observation.keyframe->mpCamera;
            robustify(edge, kHuberPixels*std::sqrt(observation.information));
            optimizer.addEdge(edge);
        }
    }
    std::map<MapPoint*, g2o::VertexSBAPointXYZ*> pointVertices;
    std::vector<MapPoint*> orderedPoints;
    for(const auto& count:observations.pointCounts) orderedPoints.push_back(count.first);
    for(MapPoint* point : orderedById(orderedPoints)) {
        // A monocular point seen once has no independent depth constraint.
        // Keep its already staged reference-frame propagation, not a fake BA.
        if(observations.pointCounts.at(point) < 2) continue;
        auto* vertex = new g2o::VertexSBAPointXYZ;
        vertex->setEstimate(proposal.pointPositions.at(point).cast<double>());
        vertex->setId(nextId++);
        vertex->setMarginalized(true);
        optimizer.addVertex(vertex);
        pointVertices[point] = vertex;
    }
    for(const auto& observation : observations.background) {
        const auto point = pointVertices.find(observation.point);
        if(point == pointVertices.end()) continue;
        auto* edge = new EdgeSE3ProjectXYZ;
        edge->setVertex(0, point->second);
        edge->setVertex(1, vertices.at(observation.keyframe));
        edge->setMeasurement(observation.pixel);
        edge->setInformation(Eigen::Matrix2d::Identity()*observation.information);
        edge->pCamera = observation.keyframe->mpCamera;
        robustify(edge, kHuberPixels);
        optimizer.addEdge(edge);
    }
    if(!optimizer.initializeOptimization()) {proposal.reason = "ba_initialization_failed"; return proposal;}
    optimizer.computeActiveErrors();
    const double initialObjective=optimizer.activeRobustChi2();
    double previousObjective=initialObjective;
    int totalIterations=0, stableBatches=0;
    // Optional diagnostic of one unchanged graph, so iteration comparisons
    // do not confound optimization with a different tracking run.
    std::ofstream iterationTrace;
    const char* tracePath=std::getenv("MARKER_BA_ITERATION_TRACE");
    if(input.convergeOffline && tracePath) iterationTrace.open(tracePath,std::ios::app);
    // Only a validated new-station scale seed gets the larger budget. Its
    // initial geometry can be far from the BA basin; ordinary/local BA and
    // final-map refinement retain their existing budgets and early stopping.
    const int convergenceBudget=input.provisionalMarkerIds.empty()?90:240;
    const int maximumIterations=input.convergeOffline
        ?std::max(options.baIterations,convergenceBudget):options.baIterations;
    const auto optimizePhase=[&]() {
      const int endIteration=totalIterations+maximumIterations;
      stableBatches=0;
      while(totalIterations<endIteration) {
        const int remaining=endIteration-totalIterations;
        // g2o restarts Levenberg damping at iteration zero on EACH optimize()
        // call. Repeated 15-step restarts stall a new-station scale solve even
        // when its raw pixels are exactly consistent. Keep this solve's LM
        // sequence continuous, with the same bounded budget/early termination.
        const int batch=input.convergeOffline && !input.provisionalMarkerIds.empty()
            ?remaining:std::min(options.baIterations,remaining);
        const int count=optimizer.optimize(batch);
        if(count<=0) {proposal.reason = "ba_solver_failed"; return false;}
        totalIterations+=count;
        optimizer.computeActiveErrors();
        const double objective=optimizer.activeRobustChi2();
        if(!std::isfinite(objective) || objective>previousObjective+1e-6*std::max(1.0,previousObjective)) {
            proposal.reason="ba_objective_increased"; return false;
        }
        const double improvement=(previousObjective-objective)/std::max(1.0,previousObjective);
        stableBatches=improvement<1e-5?stableBatches+1:0;
        previousObjective=objective;
        if(iterationTrace.is_open())
            for(const auto& item:vertices) {
                const auto center=item.second->estimate().inverse().translation();
                iterationTrace << std::setprecision(12) << totalIterations << " "
                    << item.first->mnId << " " << item.first->mTimeStamp << " "
                    << center.x() << " " << center.y() << " " << center.z()
                    << " " << objective << "\n";
            }
        if(!input.convergeOffline || stableBatches>=2) break;
      }
      return true;
    };
    if(!optimizePhase()) return proposal;
    // Projection error alone has a second basin behind the cameras. Rescue
    // only those invalid-depth vertices, using every unchanged measured edge
    // and the current solved cameras, then jointly optimize the SAME graph.
    // At most one rescue/extra bounded phase is permitted; newly invalid
    // points after it remain subject to all original depth/pixel gates.
    if(options.repairBackgroundCheirality &&
       repairBackgroundCheirality(proposal.keyframePoses,observations,vertices,pointVertices)) {
        optimizer.computeActiveErrors();
        const double repairedObjective=optimizer.activeRobustChi2();
        if(!std::isfinite(repairedObjective) ||
           repairedObjective>previousObjective+1e-6*std::max(1.0,previousObjective)) {
            proposal.reason="ba_objective_increased";return proposal;
        }
        previousObjective=repairedObjective;
        if(!optimizePhase()) return proposal;
    }
    if(input.convergeOffline)
        std::cout << "MARKER_OFFLINE_BA iterations=" << totalIterations
                  << " objective_before=" << initialObjective
                  << " objective_after=" << previousObjective
                  << " converged=" << (stableBatches>=2?1:0) << std::endl;
    if(const char* path=std::getenv("MARKER_LOOP_STAGE_DUMP")) {
        std::ofstream out(path,std::ios::app);
        out << std::setprecision(12) << "{\"type\":\"solver\",\"iterations\":" << totalIterations
            << ",\"objective_before\":" << initialObjective << ",\"objective_after\":" << previousObjective
            << ",\"converge_offline\":" << (input.convergeOffline?"true":"false") << "}\n";
    }
    if(input.convergeOffline && !input.provisionalMarkerIds.empty() && stableBatches<2) {
        proposal.reason="corner_scale_ba_not_converged";return proposal;
    }
    for(const auto& vertex : vertices) {
        const auto& pose = vertex.second->estimate();
        proposal.keyframePoses.at(vertex.first) = Sophus::SE3f(pose.rotation().cast<float>(), pose.translation().cast<float>());
    }
    // g2o marks the gauge vertex fixed, but converting its double-precision
    // estimate back to Sophus::SE3f can still leave a tiny round-trip drift.
    // Restore the exact staged gauge in the proposal before validation and
    // commit; this keeps fixed-world coordinates deterministic without
    // changing any optimized free pose or residual.
    for(const auto& fixed : fixedPoses)
        proposal.keyframePoses.at(fixed.first) = fixed.second;
    const TagCornerMap initialTagCorners = proposal.tagWorldCorners;
    bool markerDisplacementRejected = false;
    for(const auto& vertex : markerVertices) {
        const auto& estimate = vertex.second->estimate();
        const Sophus::SE3f delta(
            estimate.rotation().cast<float>(), estimate.translation().cast<float>());
        if(!finitePose(delta)) {proposal.reason = "nonfinite_optimized_marker_pose"; return proposal;}
        double worldChange = 0;
        for(const auto& point : proposal.staticTags.at(vertex.first))
            worldChange = std::max(worldChange, double((delta*point-point).norm()));
        // An unsurveyed marker's world displacement is not measurement error.
        // Production independent landmarks are validated against raw pixels,
        // rigid dimensions, positive depth and the background, below. Retain
        // the old displacement policy only for board/explicit legacy tests.
        if((input.rigidMarkerLayout || options.legacyIndependentMarkerWorldPrior ||
            options.fixAllObservedGaugeMarkers) && !input.provisionalMarkerIds.count(vertex.first) &&
           worldChange > options.maximumMarkerCornerDisplacementM) {
            // A validated local scale does not survey a distant marker's
            // world pose. During final joint BA, a free marker and its cameras
            // may coherently correct accumulated drift. Test their observable
            // relative geometry, not just an arbitrary world displacement.
            // Never extend this exception to rigid boards, a fixed gauge,
            // weak-only evidence or ordinary loop/reanchor admission.
            const auto& observers = observations.markerKeyframes.at(vertex.first);
            bool coherent = observerRelativeMarkerGate && !input.rigidMarkerLayout &&
                !vertex.second->fixed() && observers.size() >= 3;
            double relativeChange = 0, relativeFraction = 0;
            if(coherent) for(KeyFrame* observer : observers)
                for(const auto& point : proposal.staticTags.at(vertex.first)) {
                    const Eigen::Vector3d before = initialPoses.at(observer).cast<double>() * point.cast<double>();
                    const Eigen::Vector3d after = proposal.keyframePoses.at(observer).cast<double>() *
                        (delta * point).cast<double>();
                    if(!before.allFinite() || !after.allFinite() || before.z() <= 1e-6 || after.z() <= 1e-6) {
                        coherent = false;
                        continue;
                    }
                    const double change = (after-before).norm();
                    relativeChange = std::max(relativeChange, change);
                    relativeFraction = std::max(relativeFraction, change/before.norm());
                }
            coherent = coherent && relativeChange <= options.maximumMarkerCornerDisplacementM &&
                relativeFraction <= options.maximumScaleAnchorRelativeError;
            if(observerRelativeMarkerGate)
                std::cout << "MARKER_OBSERVER_DISPLACEMENT_GATE marker=" << vertex.first
                          << " world_m=" << worldChange << " relative_m=" << relativeChange
                          << " relative_fraction=" << relativeFraction
                          << " strong_views=" << observers.size() << " coherent=" << coherent << std::endl;
            if(!coherent) {
                std::cout << "MARKER_DISPLACEMENT_GATE marker=" << vertex.first
                          << " displacement_m=" << worldChange
                          << " limit_m=" << options.maximumMarkerCornerDisplacementM
                          << " strong_views=" << observations.markerKeyframes.at(vertex.first).size()
                          << " fixed=" << vertex.second->fixed() << std::endl;
                markerDisplacementRejected = true;
            }
        }
        for(auto& point : proposal.staticTags.at(vertex.first)) point = delta*point;
        for(const auto& observation : observations.tags)
            if(observation.markerId == vertex.first)
                proposal.tagWorldCorners.at(observation.keyframe).at(observation.index) =
                    delta*initialTagCorners.at(observation.keyframe).at(observation.index);
        proposal.optimizedMarkerIds.push_back(vertex.first);
    }
    for(const auto& vertex : pointVertices)
        proposal.pointPositions.at(vertex.first) = vertex.second->estimate().cast<float>();
    // A point without multi-view depth was propagated by the graph, not fitted
    // by BA. Keep that measured ray/depth in its reference camera when BA moves
    // the camera; leaving it at the pre-BA pose would create a stale point.
    std::map<MapPoint*,KeyFrame*> singleObservers;
    for(const auto& observation:observations.background)
        if(observations.pointCounts.at(observation.point)==1)
            singleObservers[observation.point]=observation.keyframe;
    for(auto& point : proposal.pointPositions) {
        if(input.pointAliases.count(point.first)) continue;
        if(pointVertices.count(point.first)) continue;
        // Admission/retries can leave a different sole observer than the
        // MapPoint's original reference. Preserve the retained observation's
        // ray/depth; propagating via a removed observer creates fake residuals.
        const auto single=singleObservers.find(point.first);
        KeyFrame* reference = single==singleObservers.end()
            ? point.first->GetReferenceKeyFrame() : single->second;
        if(!reference || !proposal.keyframePoses.count(reference)) continue;
        const auto initial = input.keyframePoses.find(reference);
        const Sophus::SE3f initialPose = initial == input.keyframePoses.end() ? reference->GetPose() : initial->second;
        const Eigen::Vector3f referenceRay = initialPose * input.pointPositions.at(point.first);
        const Sophus::SE3f finalWorldFromReference = proposal.keyframePoses.at(reference).inverse();
        point.second = finalWorldFromReference * referenceRay;
        const Eigen::Vector3f recoveredRay = proposal.keyframePoses.at(reference) * point.second;
        // These values pass through Sophus::SE3f twice.  The round trip is a
        // bookkeeping sanity check, not a geometric residual, so its tolerance
        // must reflect single-precision accumulation on long metric maps.
        if(!referenceRay.allFinite() || !recoveredRay.allFinite() ||
           (recoveredRay-referenceRay).norm() > 1e-3f*std::max(1.0f, referenceRay.norm())) {
            proposal.reason = "single_view_point_propagation_failed";
            return proposal;
        }
    }
    for(const auto& alias:input.pointAliases)
        proposal.pointPositions.at(alias.first)=proposal.pointPositions.at(alias.second);
    // Opt-in residual audit only; does not change edges or acceptance.
    const char* audit=std::getenv("MARKER_BA_RESIDUAL_AUDIT");
    if(audit && std::string(audit)=="1") {
        struct Group { int n=0, foreignReference=0; double before=0, after=0, normalized=0; };
        std::map<std::pair<unsigned long,int>,Group> groups;
        for(const auto& o:observations.background) {
            const auto initial=input.keyframePoses.find(o.keyframe);
            const Sophus::SE3d pose=(initial==input.keyframePoses.end()?o.keyframe->GetPose():initial->second).cast<double>();
            const Eigen::Vector3d a=pose*input.pointPositions.at(o.point).cast<double>();
            const Eigen::Vector3d b=proposal.keyframePoses.at(o.keyframe).cast<double>()*proposal.pointPositions.at(o.point).cast<double>();
            const double before=(o.keyframe->mpCamera->project(a)-o.pixel).squaredNorm();
            const double after=(o.keyframe->mpCamera->project(b)-o.pixel).squaredNorm();
            const bool single=!pointVertices.count(o.point);
            const int category=(single?2:0)+(after*o.information>5.991?1:0);
            auto& g=groups[{o.keyframe->mnId,category}];
            ++g.n;g.before+=before;g.after+=after;g.normalized+=after*o.information;
            if(single && o.point->GetReferenceKeyFrame()!=o.keyframe) ++g.foreignReference;
        }
        for(const auto& item:groups) {
            const auto& g=item.second;
            std::cout << "BA_RESIDUAL_GROUP kf=" << item.first.first << " category=" << item.first.second
                      << " n=" << g.n << " before=" << std::sqrt(g.before/g.n)
                      << " after=" << std::sqrt(g.after/g.n) << " normalized=" << std::sqrt(g.normalized/g.n)
                      << " foreign_reference=" << g.foreignReference << std::endl;
        }
    }
    // Keep the complete, uncommitted candidate and actual after-residuals for
    // diagnostics even on rejection. A passed relative gate is NOT acceptance:
    // all raw tag/background/depth/fixed-gauge checks still must pass. Preserve
    // rejection priority so legacy callers do not start new exclusion retries.
    if(markerDisplacementRejected) {
        proposal.after = residuals(proposal, observations);
        proposal.reason = "marker_pose_displacement_too_large";
        return proposal;
    }
    if(!validate(proposal, observations, options, fixedPoses, finalBackgroundPolicy)) return proposal;
    for(KeyFrame* keyframe : input.keyframes) if(!input.fixedKeyframes.count(keyframe))
        proposal.affectedKeyFrameIds.push_back(keyframe->mnId);
    std::sort(proposal.affectedKeyFrameIds.begin(), proposal.affectedKeyFrameIds.end());
    proposal.accepted = true;
    proposal.reason = "accepted";
    return proposal;
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::InitializeMetric(
        Map* map, const Sophus::SE3f& metricWorldFromVisualWorld,
        double metricPerVisual)
{
    return InitializeMetric(map, metricWorldFromVisualWorld, metricPerVisual, Options());
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::RefineMetricMap(Map* map)
{
    return RefineMetricMap(map, Options());
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::RefineMetricMap(
        Map* map, const Options& options)
{
    return RefineMetricMap(map, options, true);
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::RefineMetricMap(
        Map* map, const Options& options, bool observerRelativeMarkerGate)
{
    return RefineMetricMap(map, options, observerRelativeMarkerGate, true);
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::RefineMetricMap(
        Map* map, const Options& options, bool observerRelativeMarkerGate, bool finalBackgroundPolicy)
{
    Proposal rejected;
    if(!map || map->IsBad() || !map->mbMetric || map->IsInertial() ||
       !map->mbBackgroundReady) {
        rejected.reason = "invalid_metric_map_refinement_input";
        return rejected;
    }
    StagedBAInput staged;
    staged.rigidMarkerLayout = map->mbRigidMarkerLayout;
    // Experimental: the long corridor test showed little benefit and higher
    // finalization cost. Keep normal processing at its validated budget.
    const char* converge = std::getenv("MARKER_BA_CONVERGE_OFFLINE");
    staged.convergeOffline = converge && std::string(converge)=="1";
    // Native ORB intentionally keeps some stale associations until later
    // culling. The final joint BA uses only observations that are already
    // chi-square inliers in the committed pre-BA map; otherwise one obsolete
    // per-keyframe match can veto a globally improving marker/point solve.
    staged.filterInitialBackgroundOutliers = true;
    for(KeyFrame* keyframe : map->GetAllKeyFrames())
        if(keyframe && !keyframe->isBad() && keyframe->GetMap() == map)
            staged.keyframes.push_back(keyframe);
    std::sort(staged.keyframes.begin(), staged.keyframes.end(), KeyFrame::lId);
    KeyFrame* origin = map->GetOriginKF();
    if(staged.keyframes.size() < 2 || !origin || origin->isBad() ||
       origin->GetMap() != map) {
        rejected.reason = "missing_metric_map_refinement_gauge";
        return rejected;
    }
    staged.fixedKeyframes.insert(origin);
    for(const auto& marker:provisionalMarkerReferences(map,staged.keyframes))
        staged.provisionalMarkerIds.insert(marker.first);
    for(MapPoint* point : map->GetAllMapPoints())
        if(point && !point->isBad() && point->GetMap() == map)
            staged.pointPositions.emplace(point, point->GetWorldPos());
    Proposal proposal = RefineAndValidate(staged, options, nullptr, observerRelativeMarkerGate, finalBackgroundPolicy);
    std::vector<KeyFrame*> reliable;
    for(KeyFrame* keyframe : staged.keyframes)
        if(reliableTag(keyframe)) reliable.push_back(keyframe);
    for(int retry=0; retry<3 && !proposal.accepted &&
        proposal.reason == "tag_reprojection_validation_failed"; ++retry) {
        if(options.retryIsolatedMarkerGroups) {
            const auto healthy=[&](const std::pair<KeyFrame*,int>& group) {
                const auto before=proposal.before.tagRmsByKeyframeMarker.find(group);
                const auto after=proposal.after.tagRmsByKeyframeMarker.find(group);
                if(before==proposal.before.tagRmsByKeyframeMarker.end() ||
                   after==proposal.after.tagRmsByKeyframeMarker.end() ||
                   before->second>options.maximumTagRmsPx || after->second>options.maximumTagRmsPx ||
                   staged.excludedTagGroups.count(group)) return false;
                int strong=0;
                for(std::size_t i=0;i<group.first->mvTagIds.size();++i)
                    if(group.first->mvTagIds[i]==group.second &&
                       (group.first->mvTagPointWeights.empty() || group.first->mvTagPointWeights[i]>=.99f)) ++strong;
                return strong==4 && group.first->mTagObservationConfidence>=.35f;
            };
            bool added=false;
            const std::size_t limit=std::max<std::size_t>(1,proposal.before.tagRmsByKeyframeMarker.size()/4);
            for(const auto& residual:proposal.after.tagRmsByKeyframeMarker) {
                const auto initial=proposal.before.tagRmsByKeyframeMarker.find(residual.first);
                if(residual.second<=options.maximumTagRmsPx ||
                   initial==proposal.before.tagRmsByKeyframeMarker.end() ||
                   initial->second<=options.maximumTagRmsPx ||
                   staged.fixedKeyframes.count(residual.first.first) ||
                   staged.excludedTagGroups.size()>=limit) continue;
                int otherMarkers=0, otherViews=0;
                for(const auto& other:proposal.after.tagRmsByKeyframeMarker) {
                    if(!healthy(other.first)) continue;
                    if(other.first.first==residual.first.first && other.first.second!=residual.first.second) ++otherMarkers;
                    if(other.first.first!=residual.first.first && other.first.second==residual.first.second) ++otherViews;
                }
                if(otherMarkers>=2 && otherViews>=2 && staged.excludedTagGroups.insert(residual.first).second) {
                    std::cout << "MARKER_GROUP_RETRY keyframe=" << residual.first.first->mnId
                              << " marker=" << residual.first.second << std::endl;
                    added=true;
                    break; // one group per bounded retry, revalidate all remaining evidence
                }
            }
            if(added) {
                proposal=RefineAndValidate(staged,options,nullptr,observerRelativeMarkerGate,finalBackgroundPolicy);
                continue;
            }
            // A high residual alone cannot identify a bad decoded frame.
            // In particular, dropping the sole returning view of a known
            // marker removes the interval evidence we are meant to validate.
            // Default retries require the independent group evidence above;
            // whole-keyframe exclusion remains explicit legacy A/B only.
            break;
        }
        const std::size_t before=staged.excludedTagKeyframes.size();
        for(const auto& residual : proposal.after.tagRmsByKeyframe)
            if(residual.second > options.maximumTagRmsPx)
                staged.excludedTagKeyframes.insert(residual.first);
        for(const auto& residual : proposal.after.tagRmsByKeyframeMarker)
            if(residual.second > options.maximumTagRmsPx)
                staged.excludedTagKeyframes.insert(residual.first.first);
        std::size_t retained = 0;
        for(KeyFrame* keyframe : reliable)
            if(!staged.excludedTagKeyframes.count(keyframe)) ++retained;
        if(staged.excludedTagKeyframes.size()==before || retained<3 ||
           retained*2<reliable.size()) break;
        proposal = RefineAndValidate(staged, options, nullptr, observerRelativeMarkerGate, finalBackgroundPolicy);
    }
    // Legacy frozen-graph A/B only. Default final BA no longer drops an
    // entire frame's background evidence to get past a residual gate.
    // Do not let one stale keyframe-wide ORB association group veto a solve
    // that improves both aggregate marker and background error. Keep the
    // camera and tag factors, remove only that frame's background pixels, and
    // rerun the same unchanged validation. This is deliberately bounded and
    // may discard at most five percent of background-observing keyframes.
    for(int retry=0; !finalBackgroundPolicy && retry<3 && !proposal.accepted &&
        proposal.reason == "background_reprojection_validation_failed"; ++retry) {
        const std::size_t total=proposal.after.backgroundRmsByKeyframe.size();
        const std::size_t maximumExcluded=std::max<std::size_t>(1,total/20);
        const std::size_t before=staged.excludedBackgroundKeyframes.size();
        for(const auto& residual:proposal.after.backgroundNormalizedRmsByKeyframe) {
            const auto initial=proposal.before.backgroundNormalizedRmsByKeyframe.find(residual.first);
            if(initial==proposal.before.backgroundNormalizedRmsByKeyframe.end() ||
               !BackgroundResidualFrameConsistent(initial->second,residual.second))
                staged.excludedBackgroundKeyframes.insert(residual.first);
        }
        if(staged.excludedBackgroundKeyframes.size()==before ||
           staged.excludedBackgroundKeyframes.size()>maximumExcluded ||
           total<=staged.excludedBackgroundKeyframes.size()+2) break;
        proposal=RefineAndValidate(staged,options,nullptr,observerRelativeMarkerGate,finalBackgroundPolicy);
    }
    if(finalBackgroundPolicy && !proposal.accepted &&
       proposal.reason=="background_reprojection_validation_failed") {
        Proposal initial; Observations retained;
        if(!prepare(staged,options,initial,retained)) return initial;
        std::map<KeyFrame*,std::vector<const BackgroundObservation*>> frameViews;
        std::map<MapPoint*,std::vector<const BackgroundObservation*>> pointViews;
        for(const auto& o:retained.background) {
            frameViews[o.keyframe].push_back(&o); pointViews[o.point].push_back(&o);
        }
        const auto strong=[](KeyFrame* k,int id) {
            int n=0;
            for(std::size_t i=0;i<k->mvTagIds.size();++i)
                if(k->mvTagIds[i]==id && (k->mvTagPointWeights.empty() || k->mvTagPointWeights[i]>=.99f)) ++n;
            return n==4 && k->mTagObservationConfidence>=.35f;
        };
        const auto chi2=[&](const BackgroundObservation& o) {
            const Eigen::Vector3d p=proposal.keyframePoses.at(o.keyframe).cast<double>()*
                proposal.pointPositions.at(o.point).cast<double>();
            return p.allFinite() && p.z()>1e-6
                ? (o.keyframe->mpCamera->project(p)-o.pixel).squaredNorm()*o.information
                : std::numeric_limits<double>::infinity();
        };
        std::set<std::pair<KeyFrame*,MapPoint*>> rejected;
        for(const auto& frame:frameViews) {
            KeyFrame* k=frame.first;
            if(staged.fixedKeyframes.count(k)) continue;
            const auto old=proposal.before.backgroundNormalizedRmsByKeyframe.find(k);
            const auto now=proposal.after.backgroundNormalizedRmsByKeyframe.find(k);
            if(old==proposal.before.backgroundNormalizedRmsByKeyframe.end() ||
               now==proposal.after.backgroundNormalizedRmsByKeyframe.end() ||
               FinalBackgroundResidualFrameConsistent(old->second,now->second)) continue;
            // Independent support for the camera: two strong markers, each
            // also healthy in two other views. A single planar tag or a lower
            // aggregate objective alone cannot authorize deleting ORB pixels.
            int supportedMarkers=0;
            for(const auto& group:proposal.after.tagRmsByKeyframeMarker) {
                const int id=group.first.second;
                if(group.first.first!=k || !strong(k,id) || group.second>options.maximumTagRmsPx) continue;
                int otherViews=0;
                for(const auto& other:proposal.after.tagRmsByKeyframeMarker) {
                    const auto before=proposal.before.tagRmsByKeyframeMarker.find(other.first);
                    if(other.first.first!=k && other.first.second==id && strong(other.first.first,id) &&
                       before!=proposal.before.tagRmsByKeyframeMarker.end() &&
                       before->second<=options.maximumTagRmsPx && other.second<=options.maximumTagRmsPx) ++otherViews;
                }
                if(otherViews>=2) ++supportedMarkers;
            }
            if(supportedMarkers<2) continue;
            std::set<std::pair<KeyFrame*,MapPoint*>> local;
            for(const auto* pixel:frame.second) {
                const double error=chi2(*pixel);
                if(!std::isfinite(error) || error<=9.21034) continue;
                std::set<KeyFrame*> healthy;
                for(const auto* other:pointViews.at(pixel->point))
                    if(other->keyframe!=k && chi2(*other)<=9.21034) healthy.insert(other->keyframe);
                bool parallax=false;
                const Eigen::Vector3d point=proposal.pointPositions.at(pixel->point).cast<double>();
                for(KeyFrame* a:healthy) for(KeyFrame* b:healthy) {
                    if(a==b) continue;
                    const Eigen::Vector3d ra=point-proposal.keyframePoses.at(a).inverse().translation().cast<double>();
                    const Eigen::Vector3d rb=point-proposal.keyframePoses.at(b).inverse().translation().cast<double>();
                    if(ra.norm()>1e-6 && rb.norm()>1e-6 && ra.normalized().dot(rb.normalized())<.9998) parallax=true;
                }
                if(parallax) local.emplace(k,pixel->point);
            }
            // Same bounded minority budgets as the interval pixel retry.
            // Never trim the top N errors until a bad frame happens to pass.
            if(local.size()*4<=frame.second.size()) rejected.insert(local.begin(),local.end());
        }
        if(!rejected.empty() && rejected.size()*20<=retained.background.size()) {
            staged.excludedBackgroundObservations.insert(rejected.begin(),rejected.end());
            for(const auto& pixel:rejected)
                std::cout << "MARKER_FINAL_FEATURE_RETRY keyframe=" << pixel.first->mnId
                          << " point=" << pixel.second->mnId << std::endl;
            // One retry from the unchanged initial graph. Both before/after
            // errors use the same retained observations; the live map and its
            // raw feature associations are not erased by this trial.
            proposal=RefineAndValidate(staged,options,nullptr,observerRelativeMarkerGate,true);
        }
    }
    if(!staged.excludedTagKeyframes.empty()) {
        for(KeyFrame* keyframe : staged.excludedTagKeyframes)
            proposal.excludedTagKeyFrameIds.push_back(keyframe->mnId);
        std::sort(proposal.excludedTagKeyFrameIds.begin(),
                  proposal.excludedTagKeyFrameIds.end());
    }
    if(!staged.excludedBackgroundKeyframes.empty()) {
        for(KeyFrame* keyframe:staged.excludedBackgroundKeyframes)
            proposal.excludedBackgroundKeyFrameIds.push_back(keyframe->mnId);
        std::sort(proposal.excludedBackgroundKeyFrameIds.begin(),
                  proposal.excludedBackgroundKeyFrameIds.end());
    }
    if(proposal.accepted && proposal.optimizedMarkerIds.empty()) {
        proposal.accepted = false;
        proposal.reason = "no_multiview_marker_pose";
    }
    return proposal;
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::InitializeMetric(
        Map* map, const Sophus::SE3f& metricWorldFromVisualWorld,
        double metricPerVisual, const Options& options)
{
    Proposal rejected;
    if(!map || map->IsBad() || map->mbMetric || map->IsInertial() ||
       !finitePose(metricWorldFromVisualWorld) || !std::isfinite(metricPerVisual) ||
       metricPerVisual < 1e-4 || metricPerVisual > 1000.0) {
        rejected.reason = "invalid_metric_initialization_input";
        return rejected;
    }

    StagedBAInput staged;
    staged.rigidMarkerLayout = map->mbRigidMarkerLayout;
    staged.includeInactiveTagObservations = true;
    staged.filterInitialBackgroundOutliers = true;
    for(KeyFrame* keyframe : map->GetAllKeyFrames())
        if(keyframe && !keyframe->isBad() && keyframe->GetMap() == map)
            staged.keyframes.push_back(keyframe);
    std::sort(staged.keyframes.begin(), staged.keyframes.end(), KeyFrame::lId);
    KeyFrame* origin = map->GetOriginKF();
    if(staged.keyframes.size() < 2 || !origin || origin->isBad() || origin->GetMap() != map) {
        rejected.reason = "missing_metric_initialization_gauge";
        return rejected;
    }
    staged.fixedKeyframes.insert(origin);

    const Eigen::Matrix3f rotation = metricWorldFromVisualWorld.rotationMatrix();
    const Eigen::Vector3f translation = metricWorldFromVisualWorld.translation();
    for(KeyFrame* keyframe : staged.keyframes) {
        Sophus::SE3f visualTwc = keyframe->GetPoseInverse();
        visualTwc.translation() *= static_cast<float>(metricPerVisual);
        staged.keyframePoses.emplace(
            keyframe, (metricWorldFromVisualWorld * visualTwc).inverse());
    }
    for(MapPoint* point : map->GetAllMapPoints()) {
        if(!point || point->isBad() || point->GetMap() != map) continue;
        staged.pointPositions.emplace(
            point, static_cast<float>(metricPerVisual) * rotation * point->GetWorldPos() + translation);
    }

    // The fixed visual origin preserves the robust Sim(3) prior. At least one
    // separate decoded-tag keyframe must then pull the visual graph in metric
    // units; a tag attached only to the fixed origin cannot determine scale.
    const Eigen::Vector3f originCenter = staged.keyframePoses.at(origin).inverse().translation();
    bool tagBaseline = false;
    for(KeyFrame* keyframe : staged.keyframes) {
        if(keyframe == origin || !reliableTag(keyframe, true)) continue;
        const Eigen::Vector3f center = staged.keyframePoses.at(keyframe).inverse().translation();
        if((center-originCenter).norm() >= .01f) tagBaseline = true;
    }
    if(!tagBaseline) {
        rejected.reason = "insufficient_tag_keyframe_baseline";
        return rejected;
    }

    std::size_t multiViewPoints = 0;
    const std::set<KeyFrame*> selected(staged.keyframes.begin(), staged.keyframes.end());
    for(const auto& item : staged.pointPositions) {
        std::size_t observations = 0;
        for(const auto& observation : item.first->GetObservations())
            if(selected.count(observation.first)) ++observations;
        if(observations >= 2) ++multiViewPoints;
    }
    if(multiViewPoints < 3) {
        rejected.reason = "insufficient_background_geometry";
        return rejected;
    }

    Proposal proposal = RefineAndValidate(staged, options);
    std::vector<KeyFrame*> reliable;
    for(KeyFrame* keyframe : staged.keyframes)
        if(reliableTag(keyframe, true)) reliable.push_back(keyframe);
    for(int retry=0; retry<3 && !proposal.accepted &&
        proposal.reason == "tag_reprojection_validation_failed"; ++retry) {
        const std::size_t before=staged.excludedTagKeyframes.size();
        for(const auto& residual : proposal.after.tagRmsByKeyframe)
            if(residual.second > options.maximumTagRmsPx)
                staged.excludedTagKeyframes.insert(residual.first);
        for(const auto& residual : proposal.after.tagRmsByKeyframeMarker)
            if(residual.second > options.maximumTagRmsPx)
                staged.excludedTagKeyframes.insert(residual.first.first);
        std::size_t retained = 0;
        bool retainedBaseline = false;
        for(KeyFrame* keyframe : reliable) {
            if(staged.excludedTagKeyframes.count(keyframe)) continue;
            ++retained;
            const Eigen::Vector3f center = staged.keyframePoses.at(keyframe).inverse().translation();
            retainedBaseline = retainedBaseline || (center-originCenter).norm() >= .01f;
        }
        // Reject a contradictory marker set rather than cherry-picking a tiny
        // consistent subset. Retries are only for isolated damaged groups.
        if(staged.excludedTagKeyframes.size()==before || retained<3 ||
           retained*2<reliable.size() || !retainedBaseline) break;
        if(proposal.keyframePoses.size() == staged.keyframePoses.size() &&
           proposal.pointPositions.size() == staged.pointPositions.size()) {
            staged.keyframePoses = proposal.keyframePoses;
            staged.pointPositions = proposal.pointPositions;
        }
        proposal = RefineAndValidate(staged, options);
    }
    // Metric initialization uses the same complete marker/camera/point graph
    // as final refinement.  A single stale keyframe-wide ORB association
    // group must therefore be handled in the same bounded way: retain its
    // camera and marker factors, remove only that keyframe's background pixel
    // group, and rerun the unchanged validation.  Without this, an otherwise
    // sound arbitrary-scale submap cannot become metric and consequently
    // cannot be merged through a common marker at offline finalization.
    for(int retry=0; retry<8 && !proposal.accepted &&
        proposal.reason == "background_reprojection_validation_failed"; ++retry) {
        const std::size_t total=proposal.after.backgroundRmsByKeyframe.size();
        // Sequential BA retries can expose the next stale group only after the
        // previous one is gone. Keep the cap conservative but give that
        // cascade enough iterations to settle.
        const std::size_t maximumExcluded=std::max<std::size_t>(1,total/10);
        const std::size_t before=staged.excludedBackgroundKeyframes.size();
        for(const auto& residual:proposal.after.backgroundNormalizedRmsByKeyframe) {
            const auto initial=proposal.before.backgroundNormalizedRmsByKeyframe.find(residual.first);
            if(initial==proposal.before.backgroundNormalizedRmsByKeyframe.end() ||
               !BackgroundResidualFrameConsistent(initial->second,residual.second))
                staged.excludedBackgroundKeyframes.insert(residual.first);
        }
        std::cout << "METRIC_INITIALIZATION_BACKGROUND_RETRY retry=" << retry+1
                  << " excluded=" << staged.excludedBackgroundKeyframes.size()
                  << " total=" << total << " cap=" << maximumExcluded << std::endl;
        if(staged.excludedBackgroundKeyframes.size()==before ||
           staged.excludedBackgroundKeyframes.size()>maximumExcluded ||
           total<=staged.excludedBackgroundKeyframes.size()+2) break;
        proposal=RefineAndValidate(staged,options);
    }
    if(!staged.excludedTagKeyframes.empty()) {
        for(KeyFrame* keyframe : staged.excludedTagKeyframes)
            proposal.excludedTagKeyFrameIds.push_back(keyframe->mnId);
        std::sort(proposal.excludedTagKeyFrameIds.begin(),
                  proposal.excludedTagKeyFrameIds.end());
    }
    if(!staged.excludedBackgroundKeyframes.empty()) {
        for(KeyFrame* keyframe:staged.excludedBackgroundKeyframes)
            proposal.excludedBackgroundKeyFrameIds.push_back(keyframe->mnId);
        std::sort(proposal.excludedBackgroundKeyFrameIds.begin(),
                  proposal.excludedBackgroundKeyFrameIds.end());
    }
    if(!proposal.accepted) return proposal;
    if(proposal.after.backgroundObservations < 30) {
        proposal.accepted = false;
        proposal.reason = "insufficient_inlier_background_observations";
        return proposal;
    }

    // The corner BA is allowed to improve the closed-form scale. Recover the
    // scale actually present in final multi-view depth for every reference KF;
    // replay/history must not keep the superseded closed-form value.
    std::map<KeyFrame*, std::vector<double>> depthRatios;
    for(const auto& item : staged.pointPositions) {
        MapPoint* point = item.first;
        if(point->GetObservations().size() < 2 || !proposal.pointPositions.count(point)) continue;
        for(const auto& observation : point->GetObservations()) {
            KeyFrame* keyframe = observation.first;
            if(!proposal.keyframePoses.count(keyframe)) continue;
            const Eigen::Vector3f visualRay = keyframe->GetPose()*point->GetWorldPos();
            const Eigen::Vector3f metricRay = proposal.keyframePoses.at(keyframe)*proposal.pointPositions.at(point);
            if(visualRay.z() > 1e-6f && metricRay.z() > 1e-6f)
                depthRatios[keyframe].push_back(metricRay.z()/visualRay.z());
        }
    }
    for(KeyFrame* keyframe : staged.keyframes) {
        auto& ratios = depthRatios[keyframe];
        float recovered = static_cast<float>(metricPerVisual);
        if(ratios.size() >= 3) {
            std::sort(ratios.begin(), ratios.end());
            recovered = static_cast<float>(.5*(ratios[(ratios.size()-1)/2]+ratios[ratios.size()/2]));
        }
        if(!std::isfinite(recovered) || recovered <= 0 ||
           recovered/metricPerVisual < options.minimumScale ||
           recovered/metricPerVisual > options.maximumScale) {
            proposal.accepted = false;
            proposal.reason = "invalid_post_ba_initial_scale";
            return proposal;
        }
        proposal.replayScaleMultipliers[keyframe] = recovered;
    }

    // A singleton has no BA depth. Keep its measured visual ray, but express
    // that ray in the final metric unit of its optimized reference keyframe.
    for(auto& item : proposal.pointPositions) {
        MapPoint* point = item.first;
        if(point->GetObservations().size() >= 2) continue;
        KeyFrame* reference = point->GetReferenceKeyFrame();
        if(!reference || !proposal.keyframePoses.count(reference)) continue;
        const Eigen::Vector3f visualRay = reference->GetPose()*point->GetWorldPos();
        item.second = proposal.keyframePoses.at(reference).inverse() *
            (visualRay*proposal.replayScaleMultipliers.at(reference));
    }
    Proposal raw;
    Observations observations;
    if(!prepare(staged, options, raw, observations)) return raw;
    PoseMap fixedPoses;
    for(KeyFrame* fixed : staged.fixedKeyframes)
        fixedPoses.emplace(fixed, staged.keyframePoses.at(fixed));
    if(!validate(proposal, observations, options, fixedPoses)) return proposal;
    return proposal;
}

MarkerGraphOptimizer::KnownMarkerLoopEvidence MarkerGraphOptimizer::ValidateKnownMarkerLoopScale(
        Map* map, KeyFrame* current, const g2o::Sim3& seed,
        const std::vector<MapPoint*>& matches, const Options& options)
{
    KnownMarkerLoopEvidence rejected;
    if(!validOptions(options) || !map || map->IsBad() || !map->mbMetric || map->IsInertial() ||
       !current || current->isBad() || current->GetMap()!=map || !current->mpCamera || !reliableTag(current) ||
       !std::isfinite(seed.scale()) || seed.scale()<.05 || seed.scale()>20 ||
       !seed.translation().allFinite() || !seed.rotation().coeffs().allFinite()) return rejected;
    KeyFrame* origin=map->GetOriginKF();
    if(!origin || origin==current || origin->isBad() || !origin->mpCamera || !reliableTag(origin)) return rejected;
    const auto provisional=provisionalMarkerReferences(map,map->GetAllKeyFrames());
    const double limit=std::min(2.5,options.maximumTagRmsPx);
    struct Group {
        AlignedVector<Eigen::Vector3d> world;
        AlignedVector<Eigen::Vector2d> pixel;
    };
    const auto groups=[&](KeyFrame* k) {
        std::map<int,Group> result;
        if(!reliableTag(k)) return result;
        for(const auto& tag:map->mStaticTags) {
            if(tag.second.size()!=12 || provisional.count(tag.first)) continue;
            Group group;std::set<int> corners;
            for(std::size_t i=0;i<k->mvTagIds.size();++i) {
                if(k->mvTagIds[i]!=tag.first ||
                   (!k->mvTagPointWeights.empty() && k->mvTagPointWeights[i]<.99f)) continue;
                int found=-1;
                for(int j=0;j<4;++j) {
                    const Eigen::Vector3f point(tag.second[3*j],tag.second[3*j+1],tag.second[3*j+2]);
                    if((point-k->mvTagWorldPoints[i]).norm()<=1e-4f) found=j;
                }
                if(found<0 || !corners.insert(found).second) {group.world.clear();break;}
                group.world.emplace_back(tag.second[3*found],tag.second[3*found+1],tag.second[3*found+2]);
                group.pixel.emplace_back(k->mvTagImagePoints[i].x,k->mvTagImagePoints[i].y);
            }
            if(group.world.size()==4) result.emplace(tag.first,group);
        }
        return result;
    };
    const auto currentGroups=groups(current), originGroups=groups(origin);
    if(currentGroups.empty() || originGroups.empty()) return rejected;
    const auto rms=[](KeyFrame* k,const Group& group,const g2o::Sim3& transform) {
        double squared=0;
        for(std::size_t i=0;i<group.world.size();++i) {
            const Eigen::Vector3d camera=transform.rotation()*group.world[i]+transform.translation()/transform.scale();
            if(!camera.allFinite() || camera.z()<=1e-6) return std::numeric_limits<double>::infinity();
            const double error=(k->mpCamera->project(camera)-group.pixel[i]).squaredNorm();
            if(!std::isfinite(error)) return std::numeric_limits<double>::infinity();
            squared+=error;
        }
        return std::sqrt(squared/group.world.size());
    };
    // Do not let printed-marker features supply the supposed independent
    // natural-feature support. Use unique pairs and the existing 4x4 coverage
    // criterion; neither corners nor a marker PnP is used to fit this seed.
    std::set<MapPoint*> sources,destinations;
    std::set<std::pair<int,int>> cells;
    const auto currentPoints=current->GetMapPointMatches();
    const double width=current->mnMaxX-current->mnMinX, height=current->mnMaxY-current->mnMinY;
    if(!std::isfinite(width) || !std::isfinite(height) || width<=0 || height<=0) return rejected;
    for(std::size_t i=0;i<matches.size() && i<currentPoints.size() && i<current->mvKeysUn.size();++i) {
        MapPoint* a=currentPoints[i];MapPoint* b=matches[i];
        if(!a || !b || a==b || a->isBad() || b->isBad() || a->GetMap()!=map || b->GetMap()!=map) continue;
        const auto& key=current->mvKeysUn[i];bool nearMarker=false;
        for(const auto& group:currentGroups) {
            Eigen::Vector2d low=group.second.pixel.front(),high=low;
            for(const auto& p:group.second.pixel) {low=low.cwiseMin(p);high=high.cwiseMax(p);}
            if(key.pt.x>=low.x()-20 && key.pt.x<=high.x()+20 &&
               key.pt.y>=low.y()-20 && key.pt.y<=high.y()+20) nearMarker=true;
        }
        if(nearMarker || !std::isfinite(key.pt.x) || !std::isfinite(key.pt.y) || key.octave<0 ||
           std::size_t(key.octave)>=current->mvInvLevelSigma2.size() ||
           !std::isfinite(current->mvInvLevelSigma2[key.octave]) || current->mvInvLevelSigma2[key.octave]<=0) continue;
        const Eigen::Vector3d camera=seed.rotation()*b->GetWorldPos().cast<double>()+seed.translation()/seed.scale();
        const Eigen::Vector2d pixel(key.pt.x,key.pt.y);
        if(camera.z()<=1e-6 || !camera.allFinite() ||
           (current->mpCamera->project(camera)-pixel).squaredNorm()*current->mvInvLevelSigma2[key.octave]>9.21034) continue;
        if(sources.count(a) || destinations.count(b)) continue;
        sources.insert(a);destinations.insert(b);
        const int x=int(4*(key.pt.x-current->mnMinX)/width),y=int(4*(key.pt.y-current->mnMinY)/height);
        if(x>=0 && x<4 && y>=0 && y<4) cells.emplace(x,y);
    }
    if(sources.size()<20 || cells.size()<4) return rejected;
    const auto originPose=origin->GetPose().cast<double>();
    const g2o::Sim3 originTransform(originPose.unit_quaternion(),originPose.translation(),1);
    for(KeyFrame* hold:orderedById(current->GetConnectedKeyFrames())) {
        // A future frame in a frozen Atlas is never online evidence. The
        // fixed origin is not a second local observation of the drifted unit.
        if(!hold || hold==origin || hold->isBad() || hold->GetMap()!=map || !hold->mpCamera ||
           hold->mnFrameId>=current->mnFrameId || !reliableTag(hold)) continue;
        const auto relative=(hold->GetPose()*current->GetPoseInverse()).cast<double>();
        const double baseline=relative.translation().norm()/seed.scale();
        if(!std::isfinite(baseline) || baseline<.04) continue;
        const g2o::Sim3 local(relative.unit_quaternion(),relative.translation(),1);
        const auto holdGroups=groups(hold);
        for(const auto& group:currentGroups) {
            const int id=group.first;
            if(!originGroups.count(id) || !holdGroups.count(id) ||
               rms(origin,originGroups.at(id),originTransform)>limit) continue;
            const double self=rms(current,group.second,seed), cross=rms(hold,holdGroups.at(id),local*seed);
            if(self>limit || cross>limit) continue;
            double separation=std::numeric_limits<double>::infinity();
            for(double ratio:{.9,1.1}) {
                // Preserve this view's normalized SE3 while changing scale.
                // Only translated, held-out pixels can reject these seeds.
                const g2o::Sim3 alternative(seed.rotation(),seed.translation()*ratio,seed.scale()*ratio);
                separation=std::min(separation,rms(hold,holdGroups.at(id),local*alternative)-cross);
            }
            if(!std::isfinite(separation) || separation<2*limit) continue;
            KnownMarkerLoopEvidence evidence;
            evidence.valid=true;evidence.markerId=id;evidence.holdoutKeyframeId=hold->mnId;
            evidence.selfRms=self;evidence.holdoutRms=cross;evidence.baselineM=baseline;
            evidence.logScaleTolerance=std::min(.1,std::max(.01,2*limit*std::log(1.1)/separation));
            return evidence;
        }
    }
    return rejected;
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::ProposeVisualLoop(Map* map,
        KeyFrame* current, KeyFrame* matched, const g2o::Sim3& seed,
        const std::vector<MapPoint*>& matches)
{
    return ProposeVisualLoop(map,current,matched,seed,matches,Options());
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::ProposeVisualLoop(Map* map,
        KeyFrame* current, KeyFrame* matched, const g2o::Sim3& seed,
        const std::vector<MapPoint*>& matches, const Options& requestedOptions)
{
    Options options=requestedOptions;
    Proposal rejected;
    if(!validOptions(options) || !map || map->IsBad() || !map->mbMetric || map->IsInertial() || !current || !matched ||
       current->isBad() || matched->isBad() ||
       current==matched || current->GetMap()!=map || matched->GetMap()!=map) {
        rejected.reason="invalid_metric_loop_seed";return rejected;
    }
    if(!std::isfinite(seed.scale()) || !seed.translation().allFinite() ||
       !seed.rotation().coeffs().allFinite()) {
        rejected.reason="nonfinite_metric_loop_seed";return rejected;
    }
    KnownMarkerLoopEvidence scaleEvidence;
    double minimumUnit=options.minimumScale,maximumUnit=options.maximumScale;
    if(seed.scale()<options.minimumScale || seed.scale()>options.maximumScale) {
        if(options.allowLargeKnownMarkerLoopRepair)
            scaleEvidence=ValidateKnownMarkerLoopScale(map,current,seed,matches,options);
        if(!scaleEvidence.valid) {rejected.reason="metric_loop_seed_scale_out_of_bounds";return rejected;}
        const double margin=std::exp(scaleEvidence.logScaleTolerance),unit=1/seed.scale();
        // Retain the ordinary interior correction range. Endpoint evidence
        // only extends its supported direction; it does not imply monotonic
        // scale along the path or authorize the inverse extreme correction.
        minimumUnit=std::min(requestedOptions.minimumScale,std::min(1.,unit)/margin);
        maximumUnit=std::max(requestedOptions.maximumScale,std::max(1.,unit)*margin);
        options.minimumScale=std::min({options.minimumScale,seed.scale()/margin,minimumUnit});
        options.maximumScale=std::max({options.maximumScale,seed.scale()*margin,maximumUnit});
        std::cout << "MARKER_KNOWN_LOOP_SCALE current=" << current->mnId
                  << " holdout=" << scaleEvidence.holdoutKeyframeId << " marker=" << scaleEvidence.markerId
                  << " seed=" << seed.scale() << " self_rms=" << scaleEvidence.selfRms
                  << " holdout_rms=" << scaleEvidence.holdoutRms << " baseline_m=" << scaleEvidence.baselineM
                  << " log_tolerance=" << scaleEvidence.logScaleTolerance << " staged_only=1" << std::endl;
    }
    StagedBAInput raw;
    raw.rigidMarkerLayout=map->mbRigidMarkerLayout;
    raw.filterInitialBackgroundOutliers=true;
    raw.useCommittedAdmission=true;
    KeyFrame* origin=map->GetOriginKF();
    if(!origin || origin->isBad()) {rejected.reason="missing_loop_gauge";return rejected;}
    raw.fixedKeyframes.insert(origin);
    for(KeyFrame* k:map->GetAllKeyFrames())
        if(k && !k->isBad() && k->GetMap()==map) {
            raw.keyframes.push_back(k);raw.keyframePoses[k]=k->GetPose();
        }
    std::sort(raw.keyframes.begin(),raw.keyframes.end(),KeyFrame::lId);
    for(const auto& marker:provisionalMarkerReferences(map,raw.keyframes))
        raw.provisionalMarkerIds.insert(marker.first);
    for(MapPoint* p:map->GetAllMapPoints())
        if(p && !p->isBad() && p->GetMap()==map) raw.pointPositions[p]=p->GetWorldPos();
    Proposal baseline; Observations originalObservations;
    if(!prepare(raw,options,baseline,originalObservations)) return baseline;
    baseline.before=residuals(baseline,originalObservations);
    traceLoopStage("committed",baseline,originalObservations,origin,0);
    StagedBAInput staged=raw;
    const auto currentPoints=current->GetMapPointMatches();
    std::set<MapPoint*> destinations;
    for(std::size_t i=0;i<matches.size() && i<currentPoints.size();++i) {
        MapPoint* a=currentPoints[i];MapPoint* b=matches[i];
        if(!a || !b || a==b || !raw.pointPositions.count(a) || !raw.pointPositions.count(b)) continue;
        if(staged.pointAliases.count(a) || staged.pointAliases.count(b) || destinations.count(a) ||
           !destinations.insert(b).second) continue;
        staged.pointAliases[a]=b;
    }
    if(staged.pointAliases.size()<20) {rejected.reason="insufficient_loop_point_pairs";return rejected;}
    g2o::SparseOptimizer graph;
    auto* linear=new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    graph.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(new g2o::BlockSolver_7_3(linear)));
    std::map<KeyFrame*,g2o::VertexSim3Expmap*> vertices;
    // g2o::Sim3 contains fixed-size Eigen members.  A C++14 std::map node
    // does not guarantee the alignment required by Eigen on all allocators;
    // use Eigen's aligned allocator to avoid corrupting the loop graph when
    // a marker loop is staged.
    using Sim3Entry=std::pair<KeyFrame* const,g2o::Sim3>;
    std::map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
             Eigen::aligned_allocator<Sim3Entry>> old;
    for(KeyFrame* k:raw.keyframes) {
        const auto p=k->GetPose().cast<double>();
        old.emplace(k,g2o::Sim3(p.unit_quaternion(),p.translation(),1.0));
        auto* v=new g2o::VertexSim3Expmap;
        v->setId(vertices.size());v->setEstimate(old.at(k));v->setFixed(k==origin);v->_fix_scale=false;
        vertices[k]=v;graph.addVertex(v);
    }
    std::set<std::pair<unsigned long,unsigned long>> edges;
    g2o::EdgeSim3* loopEdge=nullptr;
    const auto connect=[&](KeyFrame* a,KeyFrame* b,const g2o::Sim3& measurement) {
        if(!a || !b || a==b || !vertices.count(a) || !vertices.count(b))return;
        const auto key=std::minmax(a->mnId,b->mnId);
        if(!edges.emplace(key.first,key.second).second)return;
        auto* e=new g2o::EdgeSim3;e->setVertex(0,vertices.at(a));e->setVertex(1,vertices.at(b));
        e->setMeasurement(measurement);e->setInformation(Eigen::Matrix<double,7,7>::Identity());graph.addEdge(e);
        if(!loopEdge) loopEdge=e;
    };
    connect(matched,current,seed*old.at(matched).inverse());
    for(KeyFrame* k:raw.keyframes) {
        std::set<KeyFrame*> neighbors;
        if(options.useEssentialGraphCovisibility) {
            const auto strong=k->GetCovisiblesByWeight(100);
            neighbors.insert(strong.begin(),strong.end());
        } else neighbors=k->GetConnectedKeyFrames();
        if(k->GetParent())neighbors.insert(k->GetParent());
        const auto loops=k->GetLoopEdges();neighbors.insert(loops.begin(),loops.end());
        for(KeyFrame* n:orderedById(neighbors))if(old.count(n))connect(k,n,old.at(n)*old.at(k).inverse());
    }
    const auto traceGraph=[&](const char* stage) {
        const char* path=std::getenv("MARKER_LOOP_STAGE_DUMP");
        if(!path || !loopEdge) return;
        loopEdge->computeError();
        const auto& currentSeed=vertices.at(current)->estimate();
        const auto center=currentSeed.inverse().translation();
        const auto target=seed.inverse().translation();
        std::ofstream out(path,std::ios::app);out << std::setprecision(12)
            << "{\"type\":\"loop_graph\",\"stage\":\"" << stage << "\",\"current\":" << current->mnId
            << ",\"matched\":" << matched->mnId << ",\"sim3_edges\":" << edges.size()
            << ",\"essential_covisibility\":" << (options.useEssentialGraphCovisibility?"true":"false")
            << ",\"error7\":[";
        for(int i=0;i<7;++i) out << (i?",":"") << loopEdge->error()[i];
        out << "],\"current_center\":[" << center.x() << ',' << center.y() << ',' << center.z()
            << "],\"measurement_center\":[" << target.x() << ',' << target.y() << ',' << target.z()
            << "],\"center_distance\":" << (center-target).norm() << ",\"current_scale\":" << currentSeed.scale() << "}\n";
    };
    traceGraph("before");
    if(!graph.initializeOptimization() || graph.optimize(options.graphIterations)<=0) {
        rejected.reason="loop_graph_solver_failed";return rejected;
    }
    traceGraph("after");
    for(const auto& item:vertices) {
        const auto& s=item.second->estimate();const double unit=1.0/s.scale();
        if(!std::isfinite(unit) || unit<minimumUnit || unit>maximumUnit) {
            rejected.reason="loop_graph_scale_out_of_bounds";return rejected;
        }
        staged.keyframePoses[item.first]=Sophus::SE3f(s.rotation().cast<float>(),(s.translation()/s.scale()).cast<float>());
    }
    for(auto& p:staged.pointPositions) {
        KeyFrame* ref=p.first->GetReferenceKeyFrame();
        if(vertices.count(ref))p.second=vertices.at(ref)->estimate().inverse().map(old.at(ref).map(p.second.cast<double>())).cast<float>();
    }
    // Admission is frozen in the original geometry. Never remove failed
    // frames or re-filter using the proposed loop to make it pass.
    Proposal traceSeed;Observations traceObservations;
    const bool tracing=std::getenv("MARKER_LOOP_STAGE_DUMP") && prepare(staged,options,traceSeed,traceObservations);
    if(tracing) traceLoopStage("sim3_seed",traceSeed,traceObservations,origin,staged.pointAliases.size());
    Proposal result=RefineAndValidate(staged,options,&baseline.before);
    if(tracing && result.keyframePoses.size()==staged.keyframePoses.size() &&
       result.pointPositions.size()==staged.pointPositions.size())
        traceLoopStage("ba_final",result,traceObservations,origin,staged.pointAliases.size());
    // Reject inconsistent NEW fusion constraints, never remove original
    // background frames to make a candidate pass. Recompute from the same
    // seed with the same admission population on each bounded retry.
    const auto backgroundFailed=[&](const Proposal& p) {
        if(p.after.backgroundRmsPx>options.maximumBackgroundRmsPx ||
           p.after.backgroundRmsPx>p.before.backgroundRmsPx+options.maximumBackgroundRmsIncreasePx) return true;
        for(const auto& frame:p.after.backgroundNormalizedRmsByKeyframe) {
            const auto before=p.before.backgroundNormalizedRmsByKeyframe.find(frame.first);
            if(before==p.before.backgroundNormalizedRmsByKeyframe.end() ||
               !BackgroundResidualFrameConsistent(before->second,frame.second)) return true;
        }
        return false;
    };
    for(int retry=0;retry<2 && !result.accepted &&
        (result.reason=="background_reprojection_validation_failed" ||
         (options.retryLoopAliasesAfterTagFailure && result.reason=="tag_reprojection_validation_failed" &&
          backgroundFailed(result)));++retry) {
        std::set<MapPoint*> bad;
        for(const auto& pair:staged.pointAliases) {
            for(const auto& o:originalObservations.background) {
                if(o.point!=pair.first && o.point!=pair.second)continue;
                const Eigen::Vector3d p=result.keyframePoses.at(o.keyframe).cast<double>()*result.pointPositions.at(pair.second).cast<double>();
                const Eigen::Vector3d q=raw.keyframePoses.at(o.keyframe).cast<double>()*raw.pointPositions.at(o.point).cast<double>();
                const double after=(o.keyframe->mpCamera->project(p)-o.pixel).norm();
                const double before=(o.keyframe->mpCamera->project(q)-o.pixel).norm();
                if(p.z()<=0 || !std::isfinite(after) || (after*after*o.information>9.21034 && after>before+.5)) {
                    bad.insert(pair.first);break;
                }
            }
        }
        if(bad.empty())break;
        for(MapPoint* p:bad)staged.pointAliases.erase(p);
        std::cout << "MARKER_LOOP_ALIAS_RETRY iteration=" << retry << " trigger=" << result.reason
                  << " removed=" << bad.size() << " remaining=" << staged.pointAliases.size()
                  << " original_background_observations=" << originalObservations.background.size() << std::endl;
        if(staged.pointAliases.size()<20) {result.reason="insufficient_consistent_loop_pairs";return result;}
        result=RefineAndValidate(staged,options,&baseline.before);
        if(tracing) {
            traceSeed=Proposal();traceObservations=Observations();
            if(prepare(staged,options,traceSeed,traceObservations) &&
               result.keyframePoses.size()==staged.keyframePoses.size() &&
               result.pointPositions.size()==staged.pointPositions.size())
                traceLoopStage("ba_retry",result,traceObservations,origin,staged.pointAliases.size());
        }
    }
    // A single partial marker decode can poison an otherwise valid visual
    // loop.  Retry at most twice after excluding only that weak
    // keyframe/marker group, and only when independent complete marker views
    // support the same geometry.  Endpoint and gauge keyframes, complete
    // groups, and all ORB loop-pair admission stay untouched.
    Observations validationObservations=originalObservations;
    if(!result.accepted && result.reason=="tag_reprojection_validation_failed" &&
       options.retryIsolatedMarkerGroups) {
        const auto completeStrong=[](KeyFrame* k, int id) {
            if(!k || !std::isfinite(k->mTagObservationConfidence) ||
               k->mTagObservationConfidence<.35f) return false;
            int strong=0;
            for(std::size_t i=0;i<k->mvTagIds.size();++i)
                if(k->mvTagIds[i]==id && (k->mvTagPointWeights.empty() ||
                   (i<k->mvTagPointWeights.size() &&
                    std::isfinite(k->mvTagPointWeights[i]) &&
                    k->mvTagPointWeights[i]>=.99f))) ++strong;
            return strong==4;
        };
        const std::set<KeyFrame*> protectedKeyframes{origin,current,matched};
        const std::size_t limit=std::max<std::size_t>(1,
            baseline.before.tagRmsByKeyframeMarker.size()/4);
        for(int retry=0; retry<2 && !result.accepted; ++retry) {
            std::vector<std::pair<std::pair<KeyFrame*,int>,double>> ordered(
                result.after.tagRmsByKeyframeMarker.begin(),
                result.after.tagRmsByKeyframeMarker.end());
            std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b) {
                return std::tie(a.first.first->mnId,a.first.second) <
                       std::tie(b.first.first->mnId,b.first.second);
            });
            bool added=false;
            for(const auto& residual:ordered) {
                const auto group=residual.first;
                KeyFrame* k=group.first;
                const auto before=baseline.before.tagRmsByKeyframeMarker.find(group);
                if(residual.second<=options.maximumTagRmsPx ||
                   before==baseline.before.tagRmsByKeyframeMarker.end() ||
                   before->second>options.maximumTagRmsPx ||
                   protectedKeyframes.count(k) || staged.excludedTagGroups.count(group) ||
                   staged.excludedTagGroups.size()>=limit || completeStrong(k,group.second))
                    continue;
                int count=0;
                for(int id:k->mvTagIds) if(id==group.second) ++count;
                if(count==0 || count>=4) continue;
                int otherViews=0,otherMarkers=0;
                for(const auto& other:result.after.tagRmsByKeyframeMarker) {
                    const auto previous=baseline.before.tagRmsByKeyframeMarker.find(other.first);
                    if(previous==baseline.before.tagRmsByKeyframeMarker.end() ||
                       previous->second>options.maximumTagRmsPx ||
                       other.second>options.maximumTagRmsPx ||
                       !completeStrong(other.first.first,other.first.second)) continue;
                    if(other.first.first==k && other.first.second!=group.second) ++otherMarkers;
                    if(other.first.first!=k && other.first.second==group.second) ++otherViews;
                }
                if(otherMarkers<1 || otherViews<2) continue;
                staged.excludedTagGroups.insert(group);
                std::cout << "MARKER_LOOP_WEAK_GROUP_RETRY keyframe=" << k->mnId
                          << " marker=" << group.second << " rms_px=" << residual.second
                          << " other_views=" << otherViews << " other_markers=" << otherMarkers
                          << std::endl;
                added=true;
                break;
            }
            if(!added) break;
            Proposal retryBaseline; Observations retryObservations;
            if(!prepare(staged,options,retryBaseline,retryObservations)) break;
            retryBaseline.before=residuals(retryBaseline,retryObservations);
            result=RefineAndValidate(staged,options,&retryBaseline.before);
            if(result.accepted) validationObservations=std::move(retryObservations);
        }
        for(const auto& group:staged.excludedTagGroups)
            result.excludedTagGroupIds.emplace_back(group.first->mnId,group.second);
        if(!result.accepted && !staged.excludedTagGroups.empty()) {
            Proposal ignored; Observations retained;
            if(prepare(staged,options,ignored,retained))
                validationObservations=std::move(retained);
        }
    }
    if(!result.accepted)return result;
    result.accepted=false;
    // Validate all original admitted observations, including aliased sources,
    // against the same pre-loop population and the fixed original gauge.
    PoseMap fixed{{origin,origin->GetPose()}};
    if(!validate(result,originalObservations,options,fixed))return result;
    Proposal trial;Observations trialObservations;
    if(!prepare(staged,options,trial,trialObservations))return trial;
    std::set<KeyFrame*> baObserved;
    for(const auto& o:trialObservations.tags)baObserved.insert(o.keyframe);
    for(const auto& o:trialObservations.background)
        if(trialObservations.pointCounts.at(o.point)>=2)baObserved.insert(o.keyframe);
    std::map<KeyFrame*,std::vector<double>> ratios;
    // This is bookkeeping of already optimized, shared landmark depths,
    // not a new triangulation/metric scale observation. Near-stationary
    // keyframes may share well-established points with no new parallax.
    for(const auto& o:originalObservations.background) {
        if(originalObservations.pointCounts.at(o.point)<2)continue;
        const Eigen::Vector3f a=raw.keyframePoses.at(o.keyframe)*raw.pointPositions.at(o.point);
        const Eigen::Vector3f b=result.keyframePoses.at(o.keyframe)*result.pointPositions.at(o.point);
        if(a.z()<=1e-6 || b.z()<=1e-6)continue;
        const Eigen::Vector3d cameraPoint=b.cast<double>();
        const Eigen::Vector2d projected=o.keyframe->mpCamera->project(cameraPoint);
        if((projected-o.pixel).squaredNorm()*o.information<=9.21034)
            ratios[o.keyframe].push_back(b.z()/a.z());
    }
    for(KeyFrame* k:raw.keyframes) {
        auto& values=ratios[k];
        if(k==origin){result.replayScaleMultipliers[k]=1;continue;}
        if(values.size()<3){
            if(registeredTagGeometry(k,map)){result.replayScaleMultipliers[k]=1;continue;}
            // A culled/isolated historical KF can have no BA edges. If BA
            // leaves its graph-propagated pose exactly unchanged, retain the
            // corresponding graph unit as well (not a new depth estimate).
            const auto delta=result.keyframePoses.at(k)*staged.keyframePoses.at(k).inverse();
            if(!baObserved.count(k) && delta.translation().norm()<1e-5 && delta.so3().log().norm()<1e-5) {
                result.replayScaleMultipliers[k]=1.0/vertices.at(k)->estimate().scale();
                continue;
            }
            std::cout << "STAGED_LOOP_DEPTH kf=" << k->mnId << " count=" << values.size() << std::endl;
            result.reason="loop_depth_scale_unobservable";return result;
        }
        std::sort(values.begin(),values.end());const double s=values[values.size()/2];
        if(!std::isfinite(s) || s<minimumUnit || s>maximumUnit) {
            result.reason="loop_final_scale_out_of_bounds";return result;
        }
        result.replayScaleMultipliers[k]=s;
    }
    for(const auto& o:trialObservations.background) {
        if(trialObservations.pointCounts.at(o.point)!=1)continue;
        const Eigen::Vector3f ray=raw.keyframePoses.at(o.keyframe)*raw.pointPositions.at(o.point);
        result.pointPositions.at(o.point)=result.keyframePoses.at(o.keyframe).inverse()*
            (ray*result.replayScaleMultipliers.at(o.keyframe));
    }
    for(const auto& pair:staged.pointAliases)result.pointPositions.at(pair.first)=result.pointPositions.at(pair.second);
    if(!validate(result,originalObservations,options,fixed))return result;
    if(scaleEvidence.valid && std::abs(std::log(result.replayScaleMultipliers.at(current)*seed.scale()))>
       scaleEvidence.logScaleTolerance) {
        result.reason="loop_final_scale_disagrees_with_known_marker";return result;
    }
    // Only pairs trial-fused in BA may be fused on commit.
    result.pointAliases=staged.pointAliases;
    result.accepted=true;result.reason="staged_visual_loop_validated";return result;
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::Reanchor(Map* map, KeyFrame* anchorA,
        const std::vector<KeyFrame*>& anchorB, double metricPerVisual, double sigma)
{
    return Reanchor(map, anchorA, anchorB, metricPerVisual, sigma, Options());
}

MarkerGraphOptimizer::Proposal MarkerGraphOptimizer::Reanchor(Map* map, KeyFrame* anchorA,
        const std::vector<KeyFrame*>& anchorB, double metricPerVisual, double sigma, const Options& requestedOptions)
{
    Options options=requestedOptions;
    Proposal rejected;
    if(!validOptions(options) || !map || map->IsBad() || !map->mbMetric || map->IsInertial() ||
       !anchorA || anchorA->isBad() || anchorA->GetMap() != map ||
       !std::isfinite(metricPerVisual) || metricPerVisual < options.minimumScale ||
       metricPerVisual > options.maximumScale || !std::isfinite(sigma) || sigma <= 0 || sigma > .1) {
        rejected.reason = "invalid_reanchor_input"; return rejected;
    }
    // The motion-ratio cue is a noisy log-scale measurement, not ground
    // truth. Use its uncertainty consistently with LogScaleEdge below.
    // Raw corner/background/depth validation remains mandatory.
    if(!registeredTagGeometry(anchorA, map)) {
        rejected.reason = "unreliable_or_unregistered_anchor_a"; return rejected;
    }
    std::set<KeyFrame*> anchorsB;
    std::set<unsigned long> frameIds;
    std::set<KeyFrame*> selected;
    for(KeyFrame* keyframe : anchorB) {
        if(!keyframe || keyframe == anchorA || keyframe->isBad() || keyframe->GetMap() != map ||
           !registeredTagGeometry(keyframe, map) || !anchorsB.insert(keyframe).second ||
           !frameIds.insert(keyframe->mnFrameId).second) {
            rejected.reason = "unreliable_or_duplicate_anchor_b"; return rejected;
        }
        if(!addTreePath(anchorA, keyframe, map, selected)) {
            rejected.reason = "disconnected_spanning_tree"; return rejected;
        }
    }
    if(anchorsB.size() < 2 || selected.size() < 3) {
        rejected.reason = "insufficient_anchor_window"; return rejected;
    }

    CornerScaleEvidence cornerScale;
    if(std::abs(metricPerVisual-1.)<1e-9 && sigma>=.099) {
        cornerScale=EstimateCornerScale(anchorB);
        if(cornerScale.valid) {
            metricPerVisual=cornerScale.scale; sigma=cornerScale.sigma;
            std::cout << "MARKER_CORNER_SCALE scale=" << metricPerVisual << " sigma=" << sigma
                      << " markers=" << cornerScale.markers << " rms=" << cornerScale.rms << std::endl;
            // A visual/PnP proposal alone never widens the normal envelope.
            // Independent physical squares, each checked on held-out views,
            // can instead establish that the interval really lost its unit.
            // Authorize only the measured range for this staged transaction;
            // no live scale is changed until raw tag/background/depth and
            // fixed-A validation all pass.
            if(metricPerVisual<options.minimumScale || metricPerVisual>options.maximumScale) {
                if(!options.allowLargeCornerScaleRepair || cornerScale.markers<2) {
                    rejected.cornerScale=metricPerVisual; rejected.cornerScaleSigma=sigma;
                    rejected.reason="corner_scale_outside_safe_range";return rejected;
                }
                const double uncertainty=std::exp(std::max(
                    std::log1p(options.maximumScaleAnchorRelativeError),3.*sigma));
                options.minimumScale=std::min(options.minimumScale,metricPerVisual/uncertainty);
                options.maximumScale=std::max(options.maximumScale,metricPerVisual*uncertainty);
                std::cout << "MARKER_CORNER_SCALE_REPAIR scale=" << metricPerVisual
                          << " sigma=" << sigma << " markers=" << cornerScale.markers
                          << " staged_only=1" << std::endl;
            }
        }
    }
    const double scaleCueTolerance = std::max(std::log1p(options.maximumScaleAnchorRelativeError), 3.0*sigma);
    double baseline = 0;
    for(KeyFrame* first : anchorsB) for(KeyFrame* second : anchorsB)
        baseline = std::max(baseline, double((first->GetCameraCenter()-second->GetCameraCenter()).norm()));
    if(baseline < .01) {rejected.reason = "degenerate_anchor_window"; return rejected;}

    // A spanning-tree path is not the travelled interval: a revisit may
    // attach B near A and bypass the entire excursion. Include persisted
    // frames in the A->B interval, then validate actual graph connectivity
    // below. This selects variables only; it adds no time-interpolation or
    // synthetic motion factors and never crosses map ownership.
    unsigned long lastFrame = anchorA->mnFrameId;
    for(KeyFrame* keyframe : anchorsB) lastFrame = std::max(lastFrame, keyframe->mnFrameId);
    for(KeyFrame* keyframe : map->GetAllKeyFrames())
        if(keyframe && !keyframe->isBad() && keyframe->GetMap() == map &&
           keyframe->mnFrameId >= anchorA->mnFrameId && keyframe->mnFrameId <= lastFrame)
            selected.insert(keyframe);

    const auto path=orderedById(selected);
    for(KeyFrame* keyframe : path) {
        if(options.covisibleNeighbors == 0) continue;
        for(KeyFrame* neighbor : keyframe->GetBestCovisibilityKeyFrames(options.covisibleNeighbors)) {
            if(neighbor && !neighbor->isBad() && neighbor->GetMap() == map)
                selected.insert(neighbor);
        }
    }
    StagedBAInput raw;
    raw.keyframes=orderedById(selected);
    raw.fixedKeyframes.insert(anchorA);
    for(KeyFrame* keyframe : selected) {
        if(!keyframe->mpCamera || keyframe->mpCamera2 || keyframe->NLeft != -1 || !finitePose(keyframe->GetPose())) {
            rejected.reason = "invalid_or_nonmonocular_keyframe"; return rejected;
        }
        raw.keyframePoses.emplace(keyframe, keyframe->GetPose());
    }

    g2o::SparseOptimizer graph;
    auto* linear = new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    graph.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(new g2o::BlockSolver_7_3(linear)));
    graph.setVerbose(false);
    std::map<KeyFrame*, g2o::VertexSim3Expmap*> vertices;
    // See the corresponding loop-graph map above: Sim3 must be stored in
    // Eigen-aligned map nodes before g2o starts the staged re-anchor solve.
    using Sim3Entry=std::pair<KeyFrame* const,g2o::Sim3>;
    std::map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
             Eigen::aligned_allocator<Sim3Entry>> original;
    int nextId = 0;
    for(KeyFrame* keyframe : raw.keyframes) {
        const auto pose = raw.keyframePoses.at(keyframe).cast<double>();
        const g2o::Sim3 estimate(pose.unit_quaternion(), pose.translation(), 1.0);
        original.emplace(keyframe, estimate);
        auto* vertex = new g2o::VertexSim3Expmap;
        vertex->setId(nextId++);
        vertex->setEstimate(estimate);
        vertex->setFixed(keyframe == anchorA);
        vertex->_fix_scale = false;
        graph.addVertex(vertex);
        vertices[keyframe] = vertex;
    }
    std::map<std::pair<unsigned long,unsigned long>,std::pair<KeyFrame*,KeyFrame*>> links;
    std::map<KeyFrame*, std::set<KeyFrame*>> adjacency;
    const auto connect = [&](KeyFrame* first, KeyFrame* second) {
        if(!second || first == second || !selected.count(second)) return;
        if(first->mnId>second->mnId) std::swap(first,second);
        links.emplace(std::make_pair(first->mnId,second->mnId),std::make_pair(first,second));
    };
    for(KeyFrame* keyframe : raw.keyframes) {
        connect(keyframe, keyframe->GetParent());
        for(KeyFrame* neighbor : keyframe->GetConnectedKeyFrames()) connect(keyframe, neighbor);
        for(KeyFrame* loop : keyframe->GetLoopEdges()) connect(keyframe, loop);
    }
    // g2o sorts factors by their internal insertion ID. Canonicalize both
    // orientation and insertion order; reverse Sim3 residuals with unchanged
    // information are not generally the same numerical constraint.
    for(const auto& link:links) {
        KeyFrame* first=link.second.first; KeyFrame* second=link.second.second;
        auto* edge = new g2o::EdgeSim3;
        edge->setVertex(0, vertices.at(first));
        edge->setVertex(1, vertices.at(second));
        edge->setMeasurement(original.at(second)*original.at(first).inverse());
        // The old graph is a soft spatial relation, not a new marker length
        // measurement. Only B's measured scale prior carries that information.
        Eigen::Matrix<double, 7, 7> information = Eigen::Matrix<double, 7, 7>::Identity();
        information.block<3,3>(0,0) *= 100.0;
        information.block<3,3>(3,3) *= 100.0;
        edge->setInformation(information);
        graph.addEdge(edge);
        adjacency[first].insert(second); adjacency[second].insert(first);
    }
    std::set<KeyFrame*> reached{anchorA};
    std::queue<KeyFrame*> queue;
    queue.push(anchorA);
    while(!queue.empty()) {
        KeyFrame* keyframe = queue.front(); queue.pop();
        for(KeyFrame* neighbor : adjacency[keyframe]) if(reached.insert(neighbor).second) queue.push(neighbor);
    }
    if(reached.size() != selected.size()) {rejected.reason = "disconnected_pose_graph"; return rejected;}
    for(KeyFrame* keyframe : orderedById(anchorsB)) {
        auto* edge = new LogScaleEdge;
        edge->setVertex(0, vertices.at(keyframe));
        edge->setMeasurement(-std::log(metricPerVisual));
        edge->setInformation(Eigen::Matrix<double,1,1>::Constant(1.0/(sigma*sigma)));
        graph.addEdge(edge);
    }
    Proposal rawGraph;
    Observations graphObservations;
    if(!prepare(raw, options, rawGraph, graphObservations)) return rawGraph;
    // A new station's world placement comes from the possibly drifted visual
    // trajectory. Its physical dimensions are evidence, its provisional
    // absolute location is not. Old/revisited anchors retain their factors.
    const auto provisional=provisionalMarkerReferences(map,raw.keyframes,anchorA);
    for(const auto& observation : graphObservations.tags) {
        if(provisional.count(observation.markerId)) continue;
        auto* edge = new FixedTagSim3Edge;
        edge->setVertex(0, vertices.at(observation.keyframe));
        edge->setMeasurement(observation.pixel);
        edge->setInformation(Eigen::Matrix2d::Identity()*observation.information);
        edge->worldPoint = observation.point;
        edge->camera = observation.keyframe->mpCamera;
        robustify(edge, kHuberPixels*std::sqrt(observation.information));
        graph.addEdge(edge);
    }
    if(!graph.initializeOptimization()) {rejected.reason = "graph_initialization_failed"; return rejected;}
    if(graph.optimize(options.graphIterations) < 0) {rejected.reason = "graph_solver_failed"; return rejected;}
    StagedBAInput staged;
    staged.rigidMarkerLayout = map->mbRigidMarkerLayout;
    staged.fixedKeyframes.insert(anchorA);
    for(const auto& marker:provisional) staged.provisionalMarkerIds.insert(marker.first);
    staged.convergeOffline=cornerScale.valid || !provisional.empty();
    for(KeyFrame* keyframe : selected) {
        const auto& estimate = vertices.at(keyframe)->estimate();
        const double multiplier = 1.0/estimate.scale();
        if(!std::isfinite(multiplier) || multiplier < options.minimumScale || multiplier > options.maximumScale ||
           !estimate.translation().allFinite() || !estimate.rotation().coeffs().allFinite()) {
            std::cout << "MARKER_REANCHOR_GRAPH_GATE keyframe=" << keyframe->mnId
                      << " time=" << keyframe->mTimeStamp << " multiplier=" << multiplier
                      << " minimum=" << options.minimumScale << " maximum=" << options.maximumScale
                      << " translation_finite=" << estimate.translation().allFinite()
                      << " rotation_finite=" << estimate.rotation().coeffs().allFinite() << std::endl;
            rejected.reason = "invalid_graph_sim3"; return rejected;
        }
        if(anchorsB.count(keyframe) && std::abs(std::log(multiplier/metricPerVisual)) > scaleCueTolerance) {
            std::cout << "MARKER_SCALE_GATE stage=graph keyframe=" << keyframe->mnId
                      << " measured=" << metricPerVisual << " optimized=" << multiplier
                      << " sigma=" << sigma << " log_tolerance=" << scaleCueTolerance << std::endl;
            rejected.reason = "scale_anchor_validation_failed"; return rejected;
        }
        staged.keyframePoses.emplace(keyframe, Sophus::SE3f(estimate.rotation().cast<float>(),
                                              (estimate.translation()/estimate.scale()).cast<float>()));
        staged.replayScaleMultipliers[keyframe] = multiplier;
    }
    std::set<MapPoint*> points;
    for(KeyFrame* keyframe : selected) {
        const auto matches = keyframe->GetMapPoints();
        for(MapPoint* point : matches) if(point && !point->isBad() && point->GetMap() == map) points.insert(point);
    }
    std::set<KeyFrame*> bundleKeyframes = selected;
    std::size_t multiViewPoints = 0;
    for(MapPoint* point : points) {
        const Eigen::Vector3f oldPoint = point->GetWorldPos();
        raw.pointPositions.emplace(point, oldPoint);
        KeyFrame* reference = point->GetReferenceKeyFrame();
        if(reference && selected.count(reference)) {
            const Eigen::Vector3d cameraPoint = original.at(reference).map(oldPoint.cast<double>());
            staged.pointPositions.emplace(point, vertices.at(reference)->estimate().inverse().map(cameraPoint).cast<float>());
        } else staged.pointPositions.emplace(point, oldPoint);
        std::size_t observers = 0;
        for(const auto& observation : point->GetObservations()) {
            KeyFrame* observer = observation.first;
            if(!observer || observer->isBad() || observer->GetMap() != map) continue;
            ++observers;
            if(bundleKeyframes.insert(observer).second) {
                // Observe the changed point from the untouched world as well;
                // never silently sever measurements outside the selected path.
                staged.fixedKeyframes.insert(observer);
                raw.fixedKeyframes.insert(observer);
                staged.keyframePoses.emplace(observer, observer->GetPose());
                raw.keyframePoses.emplace(observer, observer->GetPose());
            }
        }
        if(observers >= 2) ++multiViewPoints;
    }
    if(multiViewPoints < 3) {rejected.reason = "insufficient_background_geometry"; return rejected;}
    staged.keyframes=orderedById(bundleKeyframes);
    raw.keyframes = staged.keyframes;
    // Freeze boundary admission in the committed geometry, BEFORE the Sim3
    // seed moves points. Otherwise baseline and candidate RMS can refer to
    // different pixels, and a proposed correction can admit stale matches or
    // drop healthy ones merely by changing its initialization. Variable A->B
    // observations face the full gate; only independently verified outlier
    // pixels may enter the bounded, matched-population retry below.
    staged.filterFixedBackgroundOutliers = true;
    raw.filterFixedBackgroundOutliers = true;
    staged.useCommittedAdmission = true;
    raw.useCommittedAdmission = true;
    Proposal before;
    Observations observations;
    if(!prepare(raw, options, before, observations)) return before;
    before.before = residuals(before, observations);
    // A unit/.1 prior is the coordinator's corner-only initializer, not a
    // measured scale correction. Retain the committed basin for joint
    // camera/marker/point BA instead of letting
    // a provisional fixed-tag Sim3 graph deform a free marker layout first.
    // The same interval, fixed A, raw factors, depth-derived scales and final
    // validation below still apply. This seed choice removes no observation.
    // A large current marker residual is precisely what free-layout BA must
    // resolve; it does not make the provisional fixed-layout seed reliable.
    // Only the initializer changes here, never the post-solve acceptance gates.
    if(std::abs(metricPerVisual-1.)<1e-9 && sigma>=.099) {
        staged.keyframePoses=raw.keyframePoses;
        staged.pointPositions=raw.pointPositions;
        for(auto& scale:staged.replayScaleMultipliers) scale.second=1.f;
        std::cout << "MARKER_REANCHOR_SEED committed_geometry weak_unit_prior=1" << std::endl;
    }
    for(KeyFrame* k:staged.keyframes) {
        auto corners=k->mvTagWorldPoints;
        bool changed=false;
        for(std::size_t i=0;i<corners.size();++i) {
            const auto marker=provisional.find(k->mvTagIds[i]);
            if(marker==provisional.end()) continue;
            KeyFrame* reference=marker->second;
            // Marker-to-camera distance is already metric. Carry it rigidly
            // with the corrected camera, NEVER scale a physical marker.
            corners[i]=staged.keyframePoses.at(reference).inverse()*
                (raw.keyframePoses.at(reference)*corners[i]);
            changed=true;
        }
        if(changed) staged.tagWorldCorners.emplace(k,std::move(corners));
    }
    // With only one retained pixel there is no BA depth variable. Stage the
    // point along that observer's COMMITTED ray before entering BA as well;
    // a stale reference would otherwise introduce a false residual in the
    // very first validation, before final unit propagation can repair it.
    for(const auto& observation:observations.background) {
        if(observations.pointCounts.at(observation.point)!=1) continue;
        KeyFrame* observer=observation.keyframe;
        const auto unit=staged.replayScaleMultipliers.find(observer);
        const float scale=unit==staged.replayScaleMultipliers.end()?1.f:unit->second;
        const Eigen::Vector3f ray=before.keyframePoses.at(observer)*
            before.pointPositions.at(observation.point);
        staged.pointPositions.at(observation.point)=staged.keyframePoses.at(observer).inverse()*(ray*scale);
    }
    Proposal proposal = RefineAndValidate(staged, options, &before.before);
    // A partial low-information observation must not veto a whole interval
    // when independent full-marker evidence identifies an isolated conflict.
    // Keep A/B, all strong groups and all background pixels; retry from the
    // same staged seed, at most three times, with a matched baseline subset.
    const auto completeStrong=[](KeyFrame* k, int id) {
        int count=0;
        for(std::size_t i=0;i<k->mvTagIds.size();++i)
            if(k->mvTagIds[i]==id && (k->mvTagPointWeights.empty() || k->mvTagPointWeights[i]>=.99f)) ++count;
        return count==4 && k->mTagObservationConfidence>=.35f;
    };
    const std::set<KeyFrame*> protectedB(anchorsB.begin(),anchorsB.end());
    const std::size_t exclusionLimit=std::max<std::size_t>(1,before.before.tagRmsByKeyframeMarker.size()/4);
    for(int retry=0; options.retryIsolatedMarkerGroups && retry<3 && !proposal.accepted &&
        proposal.reason=="tag_reprojection_validation_failed"; ++retry) {
        bool added=false;
        std::vector<std::pair<std::pair<KeyFrame*,int>,double>> ordered(
            proposal.after.tagRmsByKeyframeMarker.begin(),proposal.after.tagRmsByKeyframeMarker.end());
        std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b) {
            return std::tie(a.first.first->mnId,a.first.second)<std::tie(b.first.first->mnId,b.first.second);
        });
        for(const auto& residual:ordered) {
            const auto group=residual.first;
            KeyFrame* k=group.first;
            if(residual.second<=options.maximumTagRmsPx || k==anchorA || protectedB.count(k) ||
               staged.provisionalMarkerIds.count(group.second) || staged.excludedTagGroups.count(group) ||
               staged.excludedTagGroups.size()>=exclusionLimit) continue;
            int count=0; bool weakOnly=true;
            for(std::size_t i=0;i<k->mvTagIds.size();++i) if(k->mvTagIds[i]==group.second) {
                ++count;
                weakOnly=weakOnly && !k->mvTagPointWeights.empty() && k->mvTagPointWeights[i]<.99f;
            }
            if(!weakOnly || count==0 || count>=4) continue;
            int otherViews=0, otherMarkers=0;
            for(const auto& other:proposal.after.tagRmsByKeyframeMarker) {
                const auto initial=proposal.before.tagRmsByKeyframeMarker.find(other.first);
                if(other.second>options.maximumTagRmsPx || initial==proposal.before.tagRmsByKeyframeMarker.end() ||
                   initial->second>options.maximumTagRmsPx || !completeStrong(other.first.first,other.first.second)) continue;
                if(other.first.first==k && other.first.second!=group.second) ++otherMarkers;
                if(other.first.first!=k && other.first.second==group.second) ++otherViews;
            }
            if(otherMarkers<1 || otherViews<2) continue;
            staged.excludedTagGroups.insert(group);
            std::cout << "MARKER_REANCHOR_WEAK_RETRY keyframe=" << k->mnId
                      << " marker=" << group.second << " rms_px=" << residual.second << std::endl;
            added=true; break;
        }
        if(!added) break;
        raw.excludedTagGroups=staged.excludedTagGroups;
        before=Proposal(); observations=Observations();
        if(!prepare(raw,options,before,observations)) return before;
        before.before=residuals(before,observations);
        proposal=RefineAndValidate(staged,options,&before.before);
    }
    for(const auto& group:staged.excludedTagGroups)
        proposal.excludedTagGroupIds.emplace_back(group.first->mnId,group.second);
    // A brief visual mismatch can leave a historical KF with good decoded
    // markers but a few wrong natural-feature correspondences. Robust BA
    // moves its pose back into marker consensus; those stale pixels must not
    // veto every later interval. One bounded retry, with independent support:
    // two complete markers, each healthy in two other views, and each rejected
    // point healthy in two other views. Never discard a gauge/B observation,
    // a whole frame, or pixels merely because the aggregate cost improves.
    if(!proposal.accepted && proposal.reason=="background_reprojection_validation_failed") {
        std::map<MapPoint*,std::vector<const BackgroundObservation*>> pointViews;
        std::map<KeyFrame*,std::size_t> frameCounts;
        for(const auto& o:observations.background) {
            pointViews[o.point].push_back(&o); ++frameCounts[o.keyframe];
        }
        const auto error=[&](const Proposal& p,const BackgroundObservation& o) {
            const Eigen::Vector3d q=p.keyframePoses.at(o.keyframe).cast<double>()*
                p.pointPositions.at(o.point).cast<double>();
            return q.allFinite() && q.z()>1e-6
                ? (o.keyframe->mpCamera->project(q)-o.pixel).norm()
                : std::numeric_limits<double>::infinity();
        };
        std::set<std::pair<KeyFrame*,MapPoint*>> rejectedPixels;
        for(const auto& f:proposal.after.backgroundRmsByKeyframe) {
            KeyFrame* k=f.first;
            if(raw.fixedKeyframes.count(k) || protectedB.count(k) ||
               f.second<=options.maximumBackgroundRmsPx) continue;
            int supportedMarkers=0;
            for(const auto& group:proposal.after.tagRmsByKeyframeMarker) {
                const int id=group.first.second;
                if(group.first.first!=k || !completeStrong(k,id) ||
                   group.second>options.maximumTagRmsPx) continue;
                int healthyViews=0;
                for(const auto& other:proposal.after.tagRmsByKeyframeMarker) {
                    const auto old=proposal.before.tagRmsByKeyframeMarker.find(other.first);
                    if(other.first.first!=k && other.first.second==id &&
                       completeStrong(other.first.first,id) && old!=proposal.before.tagRmsByKeyframeMarker.end() &&
                       old->second<=options.maximumTagRmsPx && other.second<=options.maximumTagRmsPx)
                        ++healthyViews;
                }
                if(healthyViews>=2) ++supportedMarkers;
            }
            if(supportedMarkers<2) continue;
            std::set<std::pair<KeyFrame*,MapPoint*>> local;
            for(const auto& o:observations.background) {
                if(o.keyframe!=k) continue;
                const double after=error(proposal,o), initial=error(before,o);
                if(!std::isfinite(after) || after<=options.maximumBackgroundRmsPx ||
                   after<=initial+options.maximumBackgroundRmsIncreasePx ||
                   after*after*o.information<=9.21034) continue;
                std::set<KeyFrame*> healthyViews;
                for(const auto* other:pointViews.at(o.point)) {
                    if(other->keyframe==k) continue;
                    const double e=error(proposal,*other), old=error(before,*other);
                    if(std::isfinite(e) && e<=options.maximumBackgroundRmsPx &&
                       e<=old+options.maximumBackgroundRmsIncreasePx && e*e*other->information<=9.21034)
                        healthyViews.insert(other->keyframe);
                }
                bool parallax=false;
                const Eigen::Vector3d point=proposal.pointPositions.at(o.point).cast<double>();
                for(KeyFrame* a:healthyViews) for(KeyFrame* b:healthyViews) {
                    if(a==b) continue;
                    const Eigen::Vector3d ra=point-proposal.keyframePoses.at(a).inverse().translation().cast<double>();
                    const Eigen::Vector3d rb=point-proposal.keyframePoses.at(b).inverse().translation().cast<double>();
                    if(ra.norm()>1e-6 && rb.norm()>1e-6 && ra.normalized().dot(rb.normalized())<.9998)
                        parallax=true;
                }
                if(parallax) local.emplace(k,o.point);
            }
            if(local.size()*4<=frameCounts.at(k)) rejectedPixels.insert(local.begin(),local.end());
        }
        if(!rejectedPixels.empty() && rejectedPixels.size()*20<=observations.background.size()) {
            raw.excludedBackgroundObservations=rejectedPixels;
            staged.excludedBackgroundObservations=rejectedPixels;
            // Both RMS values use exactly the same retained raw pixels.
            before=Proposal(); observations=Observations();
            if(!prepare(raw,options,before,observations)) return before;
            before.before=residuals(before,observations);
            for(const auto& pixel:rejectedPixels)
                std::cout << "MARKER_REANCHOR_FEATURE_RETRY keyframe=" << pixel.first->mnId
                          << " point=" << pixel.second->mnId << std::endl;
            proposal=RefineAndValidate(staged,options,&before.before);
            for(const auto& group:staged.excludedTagGroups)
                proposal.excludedTagGroupIds.emplace_back(group.first->mnId,group.second);
        }
    }
    if(cornerScale.valid) {
        proposal.cornerScale=cornerScale.scale;
        proposal.cornerScaleSigma=cornerScale.sigma;
    }
    if(!proposal.accepted) return proposal;
    proposal.accepted = false;

    // A pure SE3/XYZ BA can undo the graph's scale change while still fitting
    // every raw pixel. Replay units must describe the final geometry, not an
    // earlier Sim3 prior which BA was free to discard. Only independently
    // triangulated background points can supply this check; propagated
    // singleton depths would merely repeat the graph prior.
    struct DepthEvidence {
        KeyFrame* keyframe;
        Eigen::Vector3d beforeRay, afterRay;
        double ratio;
    };
    std::map<MapPoint*, std::vector<const BackgroundObservation*>> pointObservations;
    for(const auto& observation : observations.background)
        pointObservations[observation.point].push_back(&observation);
    std::map<KeyFrame*, std::vector<double>> depthRatios;
    const double maximumSquaredError = options.maximumBackgroundRmsPx*options.maximumBackgroundRmsPx;
    for(const auto& point : pointObservations) {
        if(point.second.size() < 2) continue;
        std::vector<DepthEvidence> evidence;
        for(const BackgroundObservation* observation : point.second) {
            KeyFrame* keyframe = observation->keyframe;
            const auto oldPose = before.keyframePoses.at(keyframe).cast<double>();
            const auto newPose = proposal.keyframePoses.at(keyframe).cast<double>();
            const Eigen::Vector3d oldPoint = oldPose*before.pointPositions.at(point.first).cast<double>();
            const Eigen::Vector3d newPoint = newPose*proposal.pointPositions.at(point.first).cast<double>();
            if(oldPoint.z() <= 1e-6 || newPoint.z() <= 1e-6 ||
               !oldPoint.allFinite() || !newPoint.allFinite()) continue;
            const Eigen::Vector2d oldPixel = keyframe->mpCamera->project(oldPoint);
            const Eigen::Vector2d newPixel = keyframe->mpCamera->project(newPoint);
            if(!oldPixel.allFinite() || !newPixel.allFinite() ||
               (oldPixel-observation->pixel).squaredNorm() > maximumSquaredError ||
               (newPixel-observation->pixel).squaredNorm() > maximumSquaredError) continue;
            const Eigen::Vector3d ray = keyframe->mpCamera->unprojectEig(
                cv::Point2f(observation->pixel.x(), observation->pixel.y())).cast<double>();
            if(!ray.allFinite() || ray.norm() <= 1e-9) continue;
            evidence.push_back({keyframe, oldPose.so3().inverse()*ray.normalized(),
                                newPose.so3().inverse()*ray.normalized(), newPoint.z()/oldPoint.z()});
        }
        for(const auto& view : evidence) {
            if(!selected.count(view.keyframe) || raw.fixedKeyframes.count(view.keyframe)) continue;
            for(const auto& other : evidence) {
                if(view.keyframe->mnFrameId == other.keyframe->mnFrameId) continue;
                const double oldCosine = view.beforeRay.dot(other.beforeRay);
                const double newCosine = view.afterRay.dot(other.afterRay);
                // Same angular observability threshold as non-inertial
                // monocular triangulation in LocalMapping. Check both states
                // so a rotation correction cannot create false old parallax.
                if(oldCosine > 0 && oldCosine < .9998 && newCosine > 0 && newCosine < .9998) {
                    depthRatios[view.keyframe].push_back(view.ratio);
                    break;
                }
            }
        }
    }
    ScaleMap finalReplayScales = proposal.replayScaleMultipliers;
    for(KeyFrame* fixed : raw.fixedKeyframes) finalReplayScales[fixed] = 1.0f;
    for(KeyFrame* keyframe : selected) {
        if(raw.fixedKeyframes.count(keyframe)) continue;
        auto& ratios = depthRatios[keyframe];
        if(ratios.size() < 3) {
            // A decoded fixed marker supplies the world pose, not a visual
            // length scale. Such a non-B camera keeps its original unit.
            if(!anchorsB.count(keyframe) && registeredTagGeometry(keyframe, map)) {
                finalReplayScales[keyframe] = 1.0f;
                continue;
            }
            proposal.reason = "insufficient_scale_geometry";
            std::cout << "MARKER_SCALE_GATE stage=observability keyframe=" << keyframe->mnId
                      << " timestamp=" << keyframe->mTimeStamp << " depth_points=" << ratios.size()
                      << " anchor_b=" << anchorsB.count(keyframe) << std::endl;
            return proposal;
        }
        std::sort(ratios.begin(), ratios.end());
        const double scale = .5*(ratios[(ratios.size()-1)/2]+ratios[ratios.size()/2]);
        if(!std::isfinite(scale) || scale < options.minimumScale || scale > options.maximumScale) {
            proposal.reason = "invalid_post_ba_scale";
            return proposal;
        }
        // BA may improve the provisional Sim3 scale. Check the final
        // triangulated geometry against the measured cue's uncertainty;
        // never require it to repeat an intermediate optimizer estimate.
        if(anchorsB.count(keyframe) &&
           std::abs(std::log(scale/metricPerVisual)) > scaleCueTolerance) {
            std::cout << "MARKER_SCALE_GATE stage=depth keyframe=" << keyframe->mnId
                      << " measured=" << metricPerVisual << " optimized=" << scale
                      << " sigma=" << sigma << " log_tolerance=" << scaleCueTolerance << std::endl;
            proposal.reason = "scale_geometry_consistency_failed";
            return proposal;
        }
        finalReplayScales[keyframe] = scale;
    }
    proposal.replayScaleMultipliers = finalReplayScales;
    // BA did not determine singleton depth. Reapply the final, verified local
    // unit to its original measured reference ray, keeping point propagation
    // and historical frame translations on the same scale contract.
    for(auto& point : proposal.pointPositions) {
        const auto count = observations.pointCounts.find(point.first);
        if(count != observations.pointCounts.end() && count->second >= 2) continue;
        KeyFrame* reference = point.first->GetReferenceKeyFrame();
        // Culling or boundary admission can leave a sole observer different
        // from the original MapPoint reference. Use the retained measured
        // ray AND that observer's final unit, as RefineAndValidate does.
        // Following an obsolete reference here undoes BA's correct singleton
        // propagation and can manufacture a background residual at the gauge.
        const auto retained=pointObservations.find(point.first);
        if(retained!=pointObservations.end() && retained->second.size()==1)
            reference=retained->second.front()->keyframe;
        if(!reference || !proposal.keyframePoses.count(reference)) continue;
        const Eigen::Vector3f originalRay = before.keyframePoses.at(reference)*before.pointPositions.at(point.first);
        const Eigen::Vector3f correctedRay = originalRay*proposal.replayScaleMultipliers.at(reference);
        point.second = proposal.keyframePoses.at(reference).inverse()*correctedRay;
    }
    PoseMap fixedPoses;
    for(KeyFrame* fixed : raw.fixedKeyframes) fixedPoses.emplace(fixed, raw.keyframePoses.at(fixed));
    if(!validate(proposal, observations, options, fixedPoses)) return proposal;
    proposal.affectedKeyFrameIds.clear();
    for(KeyFrame* keyframe : selected) if(keyframe != anchorA)
        proposal.affectedKeyFrameIds.push_back(keyframe->mnId);
    std::sort(proposal.affectedKeyFrameIds.begin(), proposal.affectedKeyFrameIds.end());
    proposal.accepted = true;
    proposal.reason = "accepted";
    return proposal;
}

} // namespace ORB_SLAM3
