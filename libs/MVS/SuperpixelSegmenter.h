/*
* SuperpixelSegmenter.h
*
* Copyright (c) 2025 OpenMVS Contributors
*
* Bidirectional Iterative Superpixel MVS Architecture
*
* This file implements an adaptive superpixel segmentation system
* that can be guided by depth information in a bidirectional loop.
*/

#ifndef _MVS_SUPERPIXEL_SEGMENTER_H_
#define _MVS_SUPERPIXEL_SEGMENTER_H_

#include "Common.h"
#include "DepthMap.h"
#include <opencv2/ximgproc.hpp>

namespace MVS {

// Superpixel representation
struct MVS_API Superpixel {
	int id;                              // Superpixel ID
	std::vector<ImageRef> pixels;        // Pixel coordinates
	ImageRef center;                     // Centroid

	// Plane parameters
	bool has_plane;                      // Valid plane flag
	Planef plane;                        // Plane equation: n·X + d = 0
	Normal normal;                       // Plane normal in world space
	Depth depth_center;                  // Depth at centroid
	float plane_confidence;              // Plane fitting confidence [0,1]

	// Statistics
	int num_reliable_pixels;             // Number of high-confidence pixels
	float avg_confidence;                // Average depth confidence
	float avg_depth;                     // Average depth
	float depth_variance;                // Depth variance
	float texture_variance;              // Texture variance
	Pixel32F avg_color;                  // Average color

	// Boundary info
	bool is_boundary;                    // Is on depth discontinuity
	std::vector<int> neighbor_ids;       // Adjacent superpixel IDs

	inline Superpixel() : id(-1), has_plane(false), plane_confidence(0),
	                      num_reliable_pixels(0), avg_confidence(0),
	                      avg_depth(0), depth_variance(0), texture_variance(0),
	                      is_boundary(false) {}

	// Fit plane to superpixel using RANSAC
	void FitPlane(const DepthData& depthData, float conf_threshold = 0.5f);

	// Project pixel to plane to get depth
	Depth ProjectDepth(const ImageRef& x, const Camera& camera) const;

	// Compute statistics
	void ComputeStatistics(const DepthData& depthData);

	// Check if depth is consistent with plane
	bool IsDepthConsistentWithPlane(const ImageRef& x, Depth depth, const Camera& camera, float threshold = 0.05f) const;
};

// Superpixel segmentation configuration
struct SegmentationConfig {
	// Algorithm selection
	enum Algorithm {
		SLIC,           // Simple Linear Iterative Clustering
		SEEDS,          // Superpixels Extracted via Energy-Driven Sampling
		ADAPTIVE        // Adaptive algorithm selection based on image
	};
	Algorithm algorithm;

	// SLIC parameters
	int region_size;                 // Desired superpixel size
	float ruler;                     // Compactness vs boundary adherence
	int num_iterations;              // SLIC iterations

	// Depth-guided parameters
	bool use_depth_guidance;         // Enable depth-guided segmentation
	float depth_weight;              // Weight for depth similarity [0,1]
	float depth_discontinuity_threshold;  // Threshold for depth edges

	// Adaptive parameters
	bool adaptive_region_size;       // Adjust region size based on texture
	int min_region_size;             // Minimum superpixel size
	int max_region_size;             // Maximum superpixel size

	inline SegmentationConfig() :
		algorithm(SLIC),
		region_size(20),
		ruler(10.0f),
		num_iterations(10),
		use_depth_guidance(false),
		depth_weight(0.5f),
		depth_discontinuity_threshold(0.1f),
		adaptive_region_size(false),
		min_region_size(10),
		max_region_size(50) {}
};

// Superpixel segmentation manager
class MVS_API SuperpixelSegmenter {
public:
	SuperpixelSegmenter(const SegmentationConfig& config = SegmentationConfig());
	~SuperpixelSegmenter();

	// Segment image into superpixels
	// Returns number of superpixels
	int Segment(const Image32F& image, const BitMatrix* pMask = NULL);

	// Depth-guided segmentation (our innovation!)
	// Uses depth map to guide boundary placement
	int SegmentWithDepthGuidance(const Image32F& image, const DepthMap& depthMap,
	                              const NormalMap* pNormalMap = NULL, const BitMatrix* pMask = NULL);

	// Fit planes to all superpixels
	void FitPlanes(const DepthData& depthData, float conf_threshold = 0.5f);

	// Refine superpixel boundaries based on depth
	void RefineBoundaries(const DepthMap& depthMap, float threshold = 0.1f);

	// Adaptive merge/split based on depth variance
	void AdaptiveMergeSplit(const DepthMap& depthMap, float merge_threshold = 0.01f, float split_threshold = 0.1f);

	// Query methods
	int GetSuperpixelID(const ImageRef& x) const;
	const Superpixel& GetSuperpixel(int id) const { return superpixels[id]; }
	int GetNumSuperpixels() const { return (int)superpixels.size(); }
	const std::vector<Superpixel>& GetSuperpixels() const { return superpixels; }

	// Get superpixel label map
	const cv::Mat& GetLabelMap() const { return labels; }

	// Visualization
	Image8U3 VisualizeSuperpixels() const;
	Image8U3 VisualizePlanes() const;
	Image8U3 VisualizeDepthVariance() const;
	Image8U3 VisualizeBoundaries(const DepthMap* pDepthMap = NULL) const;

	// Save/Load
	bool Save(const String& fileName) const;
	bool Load(const String& fileName);

protected:
	// Internal segmentation methods
	void SegmentSLIC(const cv::Mat& image);
	void SegmentSLICWithDepth(const cv::Mat& image, const cv::Mat& depth);
	void BuildSuperpixelList();
	void ComputeNeighborhood();
	void DetectBoundarySuperpixels(const DepthMap& depthMap, float threshold);

	// Helper methods
	cv::Mat ComputeDepthEdges(const DepthMap& depthMap) const;
	int AdaptiveRegionSize(const Image32F& image, const ImageRef& x) const;

protected:
	SegmentationConfig config;

	cv::Mat labels;                          // Label map [H x W] (int)
	std::vector<Superpixel> superpixels;     // Superpixel list

	cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic;  // SLIC algorithm
	cv::Size image_size;
};

} // namespace MVS

#endif // _MVS_SUPERPIXEL_SEGMENTER_H_
