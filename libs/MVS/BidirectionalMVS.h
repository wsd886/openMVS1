/*
* BidirectionalMVS.h
*
* INNOVATION: Bidirectional Iterative Loop Architecture
*
* This file implements the core innovation: a bidirectional loop where
* depth estimation and superpixel segmentation mutually enhance each other.
*
* Architecture:
*   Loop:
*     1. Depth-Guided Segmentation: Use current depth to guide superpixel boundaries
*     2. Segment-Guided Depth: Use superpixel planes to guide depth estimation
*     3. Joint Optimization: Simultaneously refine depth and segmentation
*   Until Convergence
*/

#ifndef _MVS_BIDIRECTIONAL_MVS_H_
#define _MVS_BIDIRECTIONAL_MVS_H_

#include "Common.h"
#include "DepthMap.h"
#include "SuperpixelSegmenter.h"

namespace MVS {

// Bidirectional MVS configuration
struct BidirectionalConfig {
	// Iteration control
	unsigned max_iterations;           // Maximum bidirectional iterations
	float convergence_threshold;       // Convergence threshold for depth change

	// Superpixel parameters
	SegmentationConfig segmentation;

	// Depth estimation parameters
	float plane_weight;                // Weight for plane constraint
	float texture_threshold;           // Threshold for textureless detection
	float confidence_threshold;        // Minimum confidence for plane fitting

	// Joint optimization
	float lambda_photo;                // Photo-consistency weight
	float lambda_plane;                // Plane-consistency weight
	float lambda_smooth;               // Smoothness weight
	float lambda_boundary;             // Boundary alignment weight

	// Adaptive parameters
	bool adaptive_weights;             // Adjust weights based on iteration
	bool early_stopping;               // Stop if converged early

	inline BidirectionalConfig() :
		max_iterations(5),
		convergence_threshold(0.01f),
		plane_weight(0.5f),
		texture_threshold(0.02f),
		confidence_threshold(0.5f),
		lambda_photo(1.0f),
		lambda_plane(0.3f),
		lambda_smooth(0.1f),
		lambda_boundary(0.2f),
		adaptive_weights(true),
		early_stopping(true) {}
};

// Bidirectional MVS engine
class MVS_API BidirectionalMVS {
public:
	BidirectionalMVS(const BidirectionalConfig& config = BidirectionalConfig());
	~BidirectionalMVS();

	// Main entry point: Bidirectional iterative depth estimation
	bool EstimateDepthMap(DepthData& depthData);

	// Individual steps (can be called separately for analysis)
	bool DepthGuidedSegmentation(DepthData& depthData);
	bool SegmentGuidedDepthEstimation(DepthData& depthData);
	bool JointOptimization(DepthData& depthData);

	// Get current state
	const SuperpixelSegmenter& GetSegmenter() const { return *pSegmenter; }
	int GetCurrentIteration() const { return current_iteration; }
	float GetConvergenceMetric() const { return convergence_metric; }

	// Debugging/visualization
	void SaveDebugImages(const String& prefix, int iteration) const;
	void PrintIterationStats() const;

protected:
	// Convergence check
	bool CheckConvergence(const DepthMap& prev_depth, const DepthMap& curr_depth);

	// Adaptive weight adjustment
	void UpdateWeights(int iteration);

	// Energy computation
	float ComputeTotalEnergy(const DepthData& depthData) const;
	float ComputePhotometricEnergy(const DepthData& depthData) const;
	float ComputePlaneEnergy(const DepthData& depthData) const;
	float ComputeBoundaryEnergy(const DepthData& depthData) const;

protected:
	BidirectionalConfig config;

	SuperpixelSegmenter* pSegmenter;   // Superpixel segmentation engine

	// Iteration state
	int current_iteration;
	float convergence_metric;
	std::vector<float> energy_history;

	// Previous state for convergence check
	DepthMap prev_depthMap;
};

} // namespace MVS

#endif // _MVS_BIDIRECTIONAL_MVS_H_
