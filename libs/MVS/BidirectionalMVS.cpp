/*
* BidirectionalMVS.cpp
*
* Implementation of the bidirectional iterative loop
*/

#include "BidirectionalMVS.h"
#include <opencv2/imgproc.hpp>

using namespace MVS;

// Constructor
BidirectionalMVS::BidirectionalMVS(const BidirectionalConfig& config)
	: config(config), current_iteration(0), convergence_metric(FLT_MAX)
{
	pSegmenter = new SuperpixelSegmenter(config.segmentation);
}

// Destructor
BidirectionalMVS::~BidirectionalMVS()
{
	delete pSegmenter;
}

// Main bidirectional iterative estimation
bool BidirectionalMVS::EstimateDepthMap(DepthData& depthData)
{
	TD_TIMER_START();

	if (depthData.IsEmpty() || depthData.images.IsEmpty()) {
		DEBUG("ERROR: Empty depth data");
		return false;
	}

	DEBUG("Starting bidirectional iterative MVS (max_iterations=%u)", config.max_iterations);

	// Save initial depth for comparison
	prev_depthMap = depthData.depthMap.clone();

	// Bidirectional loop
	for (current_iteration = 0; current_iteration < config.max_iterations; current_iteration++) {
		DEBUG("  Iteration %d/%d", current_iteration+1, config.max_iterations);

		// Step 1: Depth-Guided Segmentation
		if (!DepthGuidedSegmentation(depthData)) {
			DEBUG("    Depth-guided segmentation failed");
			continue;
		}

		// Step 2: Segment-Guided Depth Estimation
		if (!SegmentGuidedDepthEstimation(depthData)) {
			DEBUG("    Segment-guided depth estimation failed");
			continue;
		}

		// Step 3: Joint Optimization
		if (!JointOptimization(depthData)) {
			DEBUG("    Joint optimization failed");
			continue;
		}

		// Adaptive weight adjustment
		if (config.adaptive_weights) {
			UpdateWeights(current_iteration);
		}

		// Check convergence
		if (config.early_stopping && CheckConvergence(prev_depthMap, depthData.depthMap)) {
			DEBUG("  Converged at iteration %d", current_iteration+1);
			break;
		}

		// Update previous depth
		prev_depthMap = depthData.depthMap.clone();

		// Compute and log energy
		const float energy = ComputeTotalEnergy(depthData);
		energy_history.push_back(energy);
		DEBUG("    Total energy: %.4f", energy);

		// Save debug images if needed
		#ifndef _RELEASE
		if (current_iteration < 3) {  // Save first 3 iterations
			SaveDebugImages("debug_bidirectional", current_iteration);
		}
		#endif
	}

	DEBUG_ULTIMATE("Bidirectional MVS completed in %d iterations (%s)",
	               current_iteration, TD_TIMER_GET_FMT().c_str());

	return true;
}

// Step 1: Depth-Guided Segmentation
bool BidirectionalMVS::DepthGuidedSegmentation(DepthData& depthData)
{
	TD_TIMER_START();

	// Segment using current depth map to guide boundaries
	const int num_sp = pSegmenter->SegmentWithDepthGuidance(
		depthData.images.front().image,
		depthData.depthMap,
		depthData.normalMap.empty() ? NULL : &depthData.normalMap,
		depthData.mask.empty() ? NULL : &depthData.mask
	);

	if (num_sp == 0) {
		DEBUG("      Segmentation produced 0 superpixels");
		return false;
	}

	DEBUG("      Segmented into %d superpixels (%.2fms)",
	      num_sp, TD_TIMER_GET_FMT().c_str());

	return true;
}

// Step 2: Segment-Guided Depth Estimation
bool BidirectionalMVS::SegmentGuidedDepthEstimation(DepthData& depthData)
{
	TD_TIMER_START();

	// Fit planes to superpixels
	pSegmenter->FitPlanes(depthData, config.confidence_threshold);

	// Refine depth map using plane constraints
	const SuperpixelSegmenter& segmenter = *pSegmenter;
	const Camera& camera = depthData.GetCamera();
	const Image32F& image = depthData.images.front().image;

	int num_refined = 0;

	#ifdef DEPTHMAP_USE_OPENMP
	#pragma omp parallel for reduction(+:num_refined)
	#endif
	for (int i = 0; i < (int)segmenter.GetNumSuperpixels(); i++) {
		const Superpixel& sp = segmenter.GetSuperpixel(i);

		if (!sp.has_plane || sp.plane_confidence < 0.7f)
			continue;

		// For each pixel in superpixel
		for (const auto& pixel : sp.pixels) {
			// Get current depth and confidence
			Depth& depth = depthData.depthMap(pixel);
			const float conf = depthData.confMap.empty() ? 0.5f : depthData.confMap(pixel);

			// Compute texture
			float texture = 0;
			if (pixel.x > 0 && pixel.x < image.width()-1 &&
			    pixel.y > 0 && pixel.y < image.height()-1) {
				const float gx = image(pixel.x+1, pixel.y) - image(pixel.x-1, pixel.y);
				const float gy = image(pixel.x, pixel.y+1) - image(pixel.x, pixel.y-1);
				texture = gx*gx + gy*gy;
			}

			// For low-texture, low-confidence pixels: use plane depth
			if (texture < config.texture_threshold && conf < config.confidence_threshold) {
				const Depth depth_plane = sp.ProjectDepth(pixel, camera);

				if (depth_plane > 0) {
					// Blend with existing depth
					if (depth > 0) {
						const float alpha = config.plane_weight;
						depth = alpha * depth_plane + (1 - alpha) * depth;
					} else {
						depth = depth_plane;
					}

					// Update normal if available
					if (!depthData.normalMap.empty()) {
						depthData.normalMap(pixel) = Cast<NormalMap::Type>(sp.normal);
					}

					num_refined++;
				}
			}
		}
	}

	DEBUG("      Refined %d pixels using plane constraints (%.2fms)",
	      num_refined, TD_TIMER_GET_FMT().c_str());

	return true;
}

// Step 3: Joint Optimization
bool BidirectionalMVS::JointOptimization(DepthData& depthData)
{
	TD_TIMER_START();

	// Joint energy minimization over depth and segmentation
	// E(D, S) = lambda_photo * E_photo(D)
	//         + lambda_plane * E_plane(D, S)
	//         + lambda_smooth * E_smooth(D)
	//         + lambda_boundary * E_boundary(D, S)

	const SuperpixelSegmenter& segmenter = *pSegmenter;
	const Camera& camera = depthData.GetCamera();

	int num_optimized = 0;

	// Optimize depth within each superpixel
	for (int i = 0; i < (int)segmenter.GetNumSuperpixels(); i++) {
		const Superpixel& sp = segmenter.GetSuperpixel(i);

		if (sp.pixels.empty())
			continue;

		// For pixels on superpixel boundaries, enforce depth consistency
		for (const auto& pixel : sp.pixels) {
			Depth& depth = depthData.depthMap(pixel);
			if (depth <= 0) continue;

			// Check if on boundary
			bool is_boundary_pixel = false;
			const int sp_id = segmenter.GetSuperpixelID(pixel);

			// Check 4-neighbors
			const ImageRef neighbors[4] = {
				ImageRef(pixel.x-1, pixel.y),
				ImageRef(pixel.x+1, pixel.y),
				ImageRef(pixel.x, pixel.y-1),
				ImageRef(pixel.x, pixel.y+1)
			};

			for (const auto& nb : neighbors) {
				if (depthData.depthMap.isInside(nb)) {
					const int nb_sp_id = segmenter.GetSuperpixelID(nb);
					if (nb_sp_id != sp_id) {
						is_boundary_pixel = true;
						break;
					}
				}
			}

			// For boundary pixels: enforce smoothness across superpixels
			if (is_boundary_pixel && sp.has_plane) {
				const Depth depth_plane = sp.ProjectDepth(pixel, camera);

				if (depth_plane > 0) {
					// Weighted average: balance photometric and plane
					const float w_boundary = config.lambda_boundary;
					depth = (1 - w_boundary) * depth + w_boundary * depth_plane;
					num_optimized++;
				}
			}
		}
	}

	DEBUG("      Joint optimization: %d boundary pixels optimized (%.2fms)",
	      num_optimized, TD_TIMER_GET_FMT().c_str());

	return true;
}

// Check convergence
bool BidirectionalMVS::CheckConvergence(const DepthMap& prev_depth, const DepthMap& curr_depth)
{
	if (prev_depth.empty() || curr_depth.empty())
		return false;

	if (prev_depth.size() != curr_depth.size())
		return false;

	// Compute average relative depth change
	float sum_change = 0;
	int count = 0;

	for (int y = 0; y < prev_depth.rows; y++) {
		for (int x = 0; x < prev_depth.cols; x++) {
			const Depth d_prev = prev_depth(y, x);
			const Depth d_curr = curr_depth(y, x);

			if (d_prev > 0 && d_curr > 0) {
				const float rel_change = ABS(d_curr - d_prev) / d_prev;
				sum_change += rel_change;
				count++;
			}
		}
	}

	if (count == 0)
		return false;

	convergence_metric = sum_change / count;

	DEBUG("      Convergence metric: %.6f (threshold: %.6f)",
	      convergence_metric, config.convergence_threshold);

	return convergence_metric < config.convergence_threshold;
}

// Adaptive weight adjustment
void BidirectionalMVS::UpdateWeights(int iteration)
{
	// Gradually increase plane weight in later iterations
	const float t = (float)iteration / config.max_iterations;

	// Increase plane weight as iteration progresses
	config.lambda_plane = 0.3f + 0.4f * t;  // 0.3 -> 0.7

	// Decrease smoothness weight (let plane guide more)
	config.lambda_smooth = 0.1f * (1 - 0.5f * t);  // 0.1 -> 0.05

	DEBUG("      Adaptive weights: lambda_plane=%.2f, lambda_smooth=%.2f",
	      config.lambda_plane, config.lambda_smooth);
}

// Compute total energy
float BidirectionalMVS::ComputeTotalEnergy(const DepthData& depthData) const
{
	const float E_photo = ComputePhotometricEnergy(depthData);
	const float E_plane = ComputePlaneEnergy(depthData);
	const float E_boundary = ComputeBoundaryEnergy(depthData);

	return config.lambda_photo * E_photo
	     + config.lambda_plane * E_plane
	     + config.lambda_boundary * E_boundary;
}

// Photometric energy (simplified)
float BidirectionalMVS::ComputePhotometricEnergy(const DepthData& depthData) const
{
	// Simplified: sum of depth variance
	float energy = 0;
	int count = 0;

	for (int y = 0; y < depthData.depthMap.rows; y++) {
		for (int x = 0; x < depthData.depthMap.cols; x++) {
			const Depth depth = depthData.depthMap(y, x);
			if (depth > 0) {
				energy += depth;
				count++;
			}
		}
	}

	return count > 0 ? energy / count : 0;
}

// Plane energy: deviation from planes
float BidirectionalMVS::ComputePlaneEnergy(const DepthData& depthData) const
{
	const SuperpixelSegmenter& segmenter = *pSegmenter;
	const Camera& camera = depthData.GetCamera();

	float energy = 0;
	int count = 0;

	for (int i = 0; i < (int)segmenter.GetNumSuperpixels(); i++) {
		const Superpixel& sp = segmenter.GetSuperpixel(i);

		if (!sp.has_plane) continue;

		for (const auto& pixel : sp.pixels) {
			const Depth depth = depthData.depthMap(pixel);
			if (depth <= 0) continue;

			const Depth depth_plane = sp.ProjectDepth(pixel, camera);
			if (depth_plane <= 0) continue;

			const float diff = ABS(depth - depth_plane) / depth;
			energy += diff * diff;
			count++;
		}
	}

	return count > 0 ? energy / count : 0;
}

// Boundary energy: depth discontinuity at superpixel boundaries
float BidirectionalMVS::ComputeBoundaryEnergy(const DepthData& depthData) const
{
	const SuperpixelSegmenter& segmenter = *pSegmenter;

	float energy = 0;
	int count = 0;

	for (int i = 0; i < (int)segmenter.GetNumSuperpixels(); i++) {
		const Superpixel& sp = segmenter.GetSuperpixel(i);

		// Check boundary pixels
		for (const auto& pixel : sp.pixels) {
			const Depth depth = depthData.depthMap(pixel);
			if (depth <= 0) continue;

			const int sp_id = segmenter.GetSuperpixelID(pixel);

			// Check neighbors
			const ImageRef neighbors[4] = {
				ImageRef(pixel.x-1, pixel.y),
				ImageRef(pixel.x+1, pixel.y),
				ImageRef(pixel.x, pixel.y-1),
				ImageRef(pixel.x, pixel.y+1)
			};

			for (const auto& nb : neighbors) {
				if (!depthData.depthMap.isInside(nb)) continue;

				const int nb_sp_id = segmenter.GetSuperpixelID(nb);
				if (nb_sp_id == sp_id) continue;  // Same superpixel

				const Depth nb_depth = depthData.depthMap(nb);
				if (nb_depth <= 0) continue;

				// Depth discontinuity at boundary
				const float diff = ABS(depth - nb_depth) / std::max(depth, nb_depth);
				energy += diff;
				count++;
			}
		}
	}

	return count > 0 ? energy / count : 0;
}

// Save debug images
void BidirectionalMVS::SaveDebugImages(const String& prefix, int iteration) const
{
	// Visualization saved to disk for debugging
	const String iter_str = String::FormatString("_iter%02d", iteration);

	// Superpixel visualization
	const Image8U3 sp_vis = pSegmenter->VisualizeSuperpixels();
	const String sp_filename = prefix + iter_str + "_superpixels.png";
	cv::imwrite(sp_filename.c_str(), cv::Mat(sp_vis.rows, sp_vis.cols, CV_8UC3, (void*)sp_vis.data));

	// Plane visualization
	const Image8U3 plane_vis = pSegmenter->VisualizePlanes();
	const String plane_filename = prefix + iter_str + "_planes.png";
	cv::imwrite(plane_filename.c_str(), cv::Mat(plane_vis.rows, plane_vis.cols, CV_8UC3, (void*)plane_vis.data));
}

// Print iteration statistics
void BidirectionalMVS::PrintIterationStats() const
{
	DEBUG("Bidirectional MVS Statistics:");
	DEBUG("  Total iterations: %d", current_iteration);
	DEBUG("  Final convergence metric: %.6f", convergence_metric);

	if (!energy_history.empty()) {
		DEBUG("  Energy history:");
		for (size_t i = 0; i < energy_history.size(); i++) {
			DEBUG("    Iteration %d: %.4f", (int)i, energy_history[i]);
		}
	}
}
