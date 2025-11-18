/*
* MCMCPatchMatch.cpp
*
* Implementation of MCMC-Enhanced PatchMatch
*/

#include "Common.h"
#include "MCMCPatchMatch.h"
#include <random>
#include <algorithm>
#include <numeric>

using namespace MVS;

// ============================================================================
// Constructor / Destructor
// ============================================================================

MCMCPatchMatch::MCMCPatchMatch(const MCMCConfig& cfg)
	: config(cfg)
	, segmenter(nullptr)
	, global_median_depth(0)
	, rng(std::random_device{}())
	, uniform_dist(0.0f, 1.0f)
{
	segmenter = new SuperpixelSegmenter();
}

MCMCPatchMatch::~MCMCPatchMatch()
{
	delete segmenter;
}

// ============================================================================
// Main Processing
// ============================================================================

bool MCMCPatchMatch::Process(DepthData& depthData)
{
	TD_TIMER_STARTD();

	const auto& image = depthData.images.First();
	const int width = image.image.width();
	const int height = image.image.height();

	stats.total_pixels = width * height;

	DEBUG("MCMC-PatchMatch: Processing image %u (%dx%d)", image.GetID(), width, height);

	// Initialize auxiliary maps
	confidence_map = cv::Mat(height, width, CV_32F, cv::Scalar(0));
	fill_method_map = cv::Mat(height, width, CV_8U, cv::Scalar(0));
	texture_map = cv::Mat(height, width, CV_32F, cv::Scalar(0));

	// Phase 1: Superpixel segmentation & plane fitting
	{
		TD_TIMER_STARTD();
		DEBUG("  Phase 1: Superpixel segmentation & plane fitting...");

		if (!SegmentAndFitPlanes(depthData)) {
			DEBUG("WARNING: Superpixel segmentation failed");
			return false;
		}

		stats.time_superpixel = TD_TIMER_GET_SECSD();
		DEBUG("  -> Generated %u superpixels (%.3fs)", (unsigned)superpixels.size(), stats.time_superpixel);
	}

	// Compute global median depth for fallback
	{
		std::vector<Depth> valid_depths;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				Depth d = depthData.depthMap(y, x);
				if (d > 0) valid_depths.push_back(d);
			}
		}
		if (!valid_depths.empty()) {
			std::nth_element(valid_depths.begin(),
			                 valid_depths.begin() + valid_depths.size()/2,
			                 valid_depths.end());
			global_median_depth = valid_depths[valid_depths.size()/2];
		} else {
			global_median_depth = 5.0f; // Default fallback
		}
		DEBUG("  -> Global median depth: %.3f", global_median_depth);
	}

	// Phase 2: MCMC sampling on all pixels
	{
		TD_TIMER_STARTD();
		DEBUG("  Phase 2: MCMC sampling...");

		size_t converged = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				ImageRef px(x, y);

				// Compute texture variance
				float texture_var = ComputePhotometricUncertainty(depthData, px, depthData.depthMap(px));
				texture_map.at<float>(y, x) = texture_var;

				// Classify region
				if (texture_var > 0.1f) {
					stats.strong_texture_pixels++;
				} else if (texture_var > config.texture_threshold) {
					stats.medium_texture_pixels++;
				} else {
					stats.low_texture_pixels++;
				}

				// MCMC sampling (if initial depth exists)
				if (depthData.depthMap(px) > 0) {
					if (MCMCSample(depthData, px)) {
						converged++;
						fill_method_map.at<uint8_t>(y, x) = FILL_MCMC_CONVERGED;
					}
				}
			}
		}

		stats.converged_by_mcmc = converged;
		stats.time_mcmc = TD_TIMER_GET_SECSD();
		DEBUG("  -> MCMC converged: %u/%u pixels (%.3fs)",
		      (unsigned)converged, (unsigned)stats.total_pixels, stats.time_mcmc);
	}

	// Phase 3: Guaranteed hole filling
	if (config.guarantee_complete_coverage) {
		TD_TIMER_STARTD();
		DEBUG("  Phase 3: Guaranteed hole filling...");

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				ImageRef px(x, y);

				// Check if needs filling
				if (depthData.depthMap(px) <= 0 || depthData.confidenceMap(px) < config.min_fill_confidence) {
					FillResult result = ForceFillDepth(depthData, px);

					depthData.depthMap(px) = result.depth;
					depthData.confidenceMap(px) = result.confidence;
					confidence_map.at<float>(y, x) = result.confidence;
					fill_method_map.at<uint8_t>(y, x) = result.method;

					// Statistics
					switch (result.method) {
						case FILL_PLANE_PROJECTION: stats.filled_by_plane++; break;
						case FILL_NEIGHBOR_VOTING: stats.filled_by_voting++; break;
						case FILL_DEPTH_DIFFUSION: stats.filled_by_diffusion++; break;
						case FILL_GLOBAL_MEDIAN: stats.filled_by_global++; break;
						default: break;
					}
				}
			}
		}

		stats.time_hole_filling = TD_TIMER_GET_SECSD();
		DEBUG("  -> Filled holes: plane=%u, voting=%u, diffusion=%u, global=%u (%.3fs)",
		      (unsigned)stats.filled_by_plane,
		      (unsigned)stats.filled_by_voting,
		      (unsigned)stats.filled_by_diffusion,
		      (unsigned)stats.filled_by_global,
		      stats.time_hole_filling);
	}

	// Phase 4: Post-processing smooth
	SmoothLowConfidenceRegions(depthData);

	// Compute quality statistics
	size_t high_quality = stats.converged_by_mcmc + stats.filled_by_plane;
	size_t medium_quality = stats.filled_by_voting;
	size_t low_quality = stats.filled_by_diffusion + stats.filled_by_global;

	stats.high_quality_ratio = (float)high_quality / stats.total_pixels;
	stats.medium_quality_ratio = (float)medium_quality / stats.total_pixels;
	stats.low_quality_ratio = (float)low_quality / stats.total_pixels;

	// Compute average confidence
	float sum_confidence = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			sum_confidence += confidence_map.at<float>(y, x);
		}
	}
	stats.average_confidence = sum_confidence / stats.total_pixels;

	stats.time_total = TD_TIMER_GET_SECSD();

	return true;
}

// ============================================================================
// Core MCMC Functions
// ============================================================================

float MCMCPatchMatch::ComputePhotometricUncertainty(
	const DepthData& depthData, const ImageRef& x, Depth d) const
{
	if (d <= 0) return 1.0f; // Maximum uncertainty for invalid depth

	// Compute NCC variance across views
	const DepthData::ViewData& image = depthData.images.First();
	std::vector<float> ncc_values;

	for (const DepthData::ViewData& imageData : depthData.images) {
		if (&imageData == &image) continue;

		// Simple NCC computation (simplified for speed)
		const Point3 X(image.camera.TransformPointI2W(Point3(x.x, x.y, d)));
		const Point3 camX(imageData.camera.TransformPointW2C(X));
		if (camX.z <= 0) continue;

		const Point2f pt(imageData.camera.TransformPointC2I(camX));
		if (!imageData.image.isInside(pt)) continue;

		// Approximate NCC by intensity difference
		const float I_ref = image.image.sample(Point2f(x));
		const float I_src = imageData.image.sample(pt);
		const float ncc = 1.0f - MINF(ABS(I_ref - I_src) / 255.0f, 1.0f);

		ncc_values.push_back(ncc);
	}

	if (ncc_values.size() < 2) return 0.5f;

	// Compute variance
	const float mean_ncc = std::accumulate(ncc_values.begin(), ncc_values.end(), 0.0f) / ncc_values.size();
	float variance = 0;
	for (float ncc : ncc_values) {
		variance += (ncc - mean_ncc) * (ncc - mean_ncc);
	}
	variance /= ncc_values.size();

	return CLAMP(variance, 0.0f, 1.0f);
}

float MCMCPatchMatch::ComputeAdaptiveTemperature(float photometric_uncertainty) const
{
	// Adaptive temperature: low texture -> low beta (more exploration)
	// Formula: β(x) = β_max - (β_max - β_min) * U / (U + τ)
	const float ratio = photometric_uncertainty / (photometric_uncertainty + config.tau_uncertainty);
	const float beta = config.beta_max - (config.beta_max - config.beta_min) * ratio;

	return CLAMP(beta, config.beta_min, config.beta_max);
}

float MCMCPatchMatch::ComputeAcceptanceProbability(float delta_E, float beta) const
{
	// Softmax acceptance: α = 1 / (1 + exp(-β * ΔE))
	// This is the Metropolis-Hastings acceptance probability
	if (delta_E >= 0) {
		// Better or equal: high probability
		return 1.0f / (1.0f + expf(-beta * delta_E));
	} else {
		// Worse: still some probability (key for avoiding local optima)
		return 1.0f / (1.0f + expf(-beta * delta_E));
	}
}

bool MCMCPatchMatch::AcceptProposal(float acceptance_prob) const
{
	return uniform_dist(rng) < acceptance_prob;
}

// ============================================================================
// Energy Computation
// ============================================================================

float MCMCPatchMatch::ComputeEnergy(
	const DepthData& depthData, const ImageRef& x, Depth d, const Normal& n) const
{
	const float E_photo = ComputePhotometricEnergy(depthData, x, d, n);
	const float E_plane = ComputePlanarEnergy(x, d, GetSuperpixelID(x));

	return E_photo + config.lambda_prior * E_plane;
}

float MCMCPatchMatch::ComputePhotometricEnergy(
	const DepthData& depthData, const ImageRef& x, Depth d, const Normal& n) const
{
	// Negative because we want to maximize NCC (minimize -NCC)
	// This is simplified - full implementation would use proper NCC
	const DepthData::ViewData& image = depthData.images.First();

	float total_ncc = 0;
	int count = 0;

	for (const DepthData::ViewData& imageData : depthData.images) {
		if (&imageData == &image) continue;

		const Point3 X(image.camera.TransformPointI2W(Point3(x.x, x.y, d)));
		const Point3 camX(imageData.camera.TransformPointW2C(X));
		if (camX.z <= 0) continue;

		const Point2f pt(imageData.camera.TransformPointC2I(camX));
		if (!imageData.image.isInside(pt)) continue;

		const float I_ref = image.image.sample(Point2f(x));
		const float I_src = imageData.image.sample(pt);
		const float ncc = 1.0f - MINF(ABS(I_ref - I_src) / 255.0f, 1.0f);

		total_ncc += ncc;
		count++;
	}

	if (count == 0) return 0;

	// Return negative (want to maximize NCC = minimize -NCC)
	return -(total_ncc / count);
}

float MCMCPatchMatch::ComputePlanarEnergy(const ImageRef& x, Depth d, int superpixel_id) const
{
	if (superpixel_id < 0 || superpixel_id >= (int)superpixels.size())
		return 0;

	const Superpixel& sp = superpixels[superpixel_id];
	if (!sp.has_plane || sp.plane_confidence < config.plane_confidence_threshold)
		return 0;

	// Project depth from plane
	const Depth d_plane = sp.plane.m_fD > 0 ? -sp.plane.m_fD : d; // Simplified

	// Gaussian prior: E = (d - d_plane)² / (2σ²)
	const float diff = d - d_plane;
	const float energy = (diff * diff) / (2.0f * config.sigma_plane * config.sigma_plane);

	return energy;
}

// ============================================================================
// MCMC Sampling
// ============================================================================

bool MCMCPatchMatch::MCMCSample(DepthData& depthData, const ImageRef& x)
{
	Depth d_current = depthData.depthMap(x);
	if (d_current <= 0) return false;

	Normal n_current = depthData.normalMap(x);

	// Compute adaptive temperature
	const float U_photo = ComputePhotometricUncertainty(depthData, x, d_current);
	const float beta = ComputeAdaptiveTemperature(U_photo);

	// Number of samples based on texture
	const bool is_low_texture = U_photo > config.texture_threshold;
	const unsigned num_samples = is_low_texture ?
		config.num_samples_low_texture : config.num_samples_high_texture;

	// Current energy
	float E_current = ComputeEnergy(depthData, x, d_current, n_current);

	// MCMC iterations
	bool accepted_any = false;
	for (unsigned iter = 0; iter < num_samples; iter++) {
		// Propose new depth (random perturbation)
		const float sigma_propose = is_low_texture ? 0.5f : 0.1f;
		std::normal_distribution<float> normal_dist(0, sigma_propose);
		const Depth d_proposal = d_current + normal_dist(rng);

		if (d_proposal <= 0) continue;

		// Compute energy of proposal
		const float E_proposal = ComputeEnergy(depthData, x, d_proposal, n_current);

		// Energy difference
		const float delta_E = E_current - E_proposal; // Note: negative sign (minimizing E)

		// Acceptance probability
		const float alpha = ComputeAcceptanceProbability(delta_E, beta);

		// Accept/reject
		if (AcceptProposal(alpha)) {
			d_current = d_proposal;
			E_current = E_proposal;
			accepted_any = true;
		}
	}

	// Update if changed
	if (accepted_any) {
		depthData.depthMap(x) = d_current;
		confidence_map.at<float>(x.y, x.x) = MINF(1.0f - U_photo, 1.0f);
	}

	return accepted_any;
}

// ============================================================================
// Superpixel & Plane Fitting
// ============================================================================

bool MCMCPatchMatch::SegmentAndFitPlanes(const DepthData& depthData)
{
	const DepthData::ViewData& image = depthData.images.First();

	// Configure superpixel segmentation
	SuperpixelConfig sp_config;
	sp_config.region_size = config.superpixel_size;
	sp_config.ruler = config.superpixel_ruler;
	sp_config.depth_weight = config.superpixel_depth_weight;

	// Segment
	if (!segmenter->Segment(image.image, depthData.depthMap, sp_config)) {
		return false;
	}

	superpixel_labels = segmenter->GetLabels();

	// Fit planes
	const unsigned num_superpixels = segmenter->GetNumSuperpixels();
	superpixels.resize(num_superpixels);

	for (unsigned i = 0; i < num_superpixels; i++) {
		superpixels[i] = segmenter->GetSuperpixel(i);
		superpixels[i].FitPlane(depthData, 0.8f); // Confidence threshold
	}

	return true;
}

// ============================================================================
// Guaranteed Hole Filling
// ============================================================================

FillResult MCMCPatchMatch::ForceFillDepth(const DepthData& depthData, const ImageRef& x)
{
	// Strategy 1: Plane projection (highest quality)
	{
		FillResult result = FillByPlaneProjection(x);
		if (result.confidence >= 0.7f) {
			return result;
		}
	}

	// Strategy 2: Neighbor plane voting (medium quality)
	{
		FillResult result = FillByNeighborVoting(x);
		if (result.confidence >= 0.5f) {
			return result;
		}
	}

	// Strategy 3: Depth diffusion (low quality)
	{
		FillResult result = FillByDepthDiffusion(depthData, x);
		if (result.depth > 0) {
			return result;
		}
	}

	// Strategy 4: Global median (worst case)
	return FillByGlobalMedian();
}

FillResult MCMCPatchMatch::FillByPlaneProjection(const ImageRef& x) const
{
	const int sp_id = GetSuperpixelID(x);
	if (sp_id < 0 || sp_id >= (int)superpixels.size())
		return FillResult();

	const Superpixel& sp = superpixels[sp_id];
	if (!sp.has_plane || sp.plane_confidence < config.plane_confidence_threshold)
		return FillResult();

	// Project depth from plane (simplified)
	const Depth d = sp.plane.m_fD > 0 ? ABS(sp.plane.m_fD) : 0;

	if (d <= 0) return FillResult();

	return FillResult(d, sp.plane_confidence, FILL_PLANE_PROJECTION);
}

FillResult MCMCPatchMatch::FillByNeighborVoting(const ImageRef& x) const
{
	std::vector<std::pair<Depth, float>> depth_weight_pairs;

	// Search in expanding radius
	for (unsigned radius = 1; radius <= 3; radius++) {
		std::vector<int> neighbor_sps = GetNeighborSuperpixels(x, radius);

		for (int sp_id : neighbor_sps) {
			if (sp_id < 0 || sp_id >= (int)superpixels.size()) continue;

			const Superpixel& sp = superpixels[sp_id];
			if (!sp.has_plane || sp.plane_confidence < 0.5f) continue;

			const Depth d = sp.plane.m_fD > 0 ? ABS(sp.plane.m_fD) : 0;
			if (d <= 0) continue;

			const float weight = sp.plane_confidence / radius; // Distance decay
			depth_weight_pairs.push_back({d, weight});
		}

		if (!depth_weight_pairs.empty()) break;
	}

	if (depth_weight_pairs.empty())
		return FillResult();

	// Weighted median
	std::sort(depth_weight_pairs.begin(), depth_weight_pairs.end());

	float sum_weight = 0;
	for (const auto& pair : depth_weight_pairs) {
		sum_weight += pair.second;
	}

	float cumsum = 0;
	Depth median_depth = depth_weight_pairs[0].first;
	for (const auto& pair : depth_weight_pairs) {
		cumsum += pair.second;
		if (cumsum >= sum_weight / 2) {
			median_depth = pair.first;
			break;
		}
	}

	return FillResult(median_depth, 0.6f, FILL_NEIGHBOR_VOTING);
}

FillResult MCMCPatchMatch::FillByDepthDiffusion(const DepthData& depthData, const ImageRef& x) const
{
	const Depth d = FindNearestValidDepth(depthData, x, config.max_diffusion_radius);

	if (d > 0) {
		// Confidence decreases with distance (simplified)
		return FillResult(d, 0.3f, FILL_DEPTH_DIFFUSION);
	}

	return FillResult();
}

FillResult MCMCPatchMatch::FillByGlobalMedian() const
{
	return FillResult(global_median_depth, 0.1f, FILL_GLOBAL_MEDIAN);
}

// ============================================================================
// Utilities
// ============================================================================

bool MCMCPatchMatch::IsLowTexture(const DepthData& depthData, const ImageRef& x) const
{
	const float variance = ComputePhotometricUncertainty(depthData, x, depthData.depthMap(x));
	return variance < config.texture_threshold;
}

int MCMCPatchMatch::GetSuperpixelID(const ImageRef& x) const
{
	if (superpixel_labels.empty()) return -1;
	if (x.x < 0 || x.x >= superpixel_labels.cols || x.y < 0 || x.y >= superpixel_labels.rows)
		return -1;

	return superpixel_labels.at<int>(x.y, x.x);
}

std::vector<int> MCMCPatchMatch::GetNeighborSuperpixels(const ImageRef& x, unsigned radius) const
{
	std::set<int> unique_ids;

	const int x_min = MAXF(0, (int)x.x - (int)radius);
	const int x_max = MINF(superpixel_labels.cols - 1, (int)x.x + (int)radius);
	const int y_min = MAXF(0, (int)x.y - (int)radius);
	const int y_max = MINF(superpixel_labels.rows - 1, (int)x.y + (int)radius);

	for (int yy = y_min; yy <= y_max; yy++) {
		for (int xx = x_min; xx <= x_max; xx++) {
			const int sp_id = superpixel_labels.at<int>(yy, xx);
			if (sp_id >= 0) {
				unique_ids.insert(sp_id);
			}
		}
	}

	return std::vector<int>(unique_ids.begin(), unique_ids.end());
}

Depth MCMCPatchMatch::FindNearestValidDepth(
	const DepthData& depthData, const ImageRef& x, unsigned max_radius) const
{
	for (unsigned radius = 1; radius <= max_radius; radius++) {
		std::vector<Depth> valid_depths;

		const int x_min = MAXF(0, (int)x.x - (int)radius);
		const int x_max = MINF(depthData.depthMap.cols - 1, (int)x.x + (int)radius);
		const int y_min = MAXF(0, (int)x.y - (int)radius);
		const int y_max = MINF(depthData.depthMap.rows - 1, (int)x.y + (int)radius);

		for (int yy = y_min; yy <= y_max; yy++) {
			for (int xx = x_min; xx <= x_max; xx++) {
				const Depth d = depthData.depthMap(yy, xx);
				if (d > 0) {
					valid_depths.push_back(d);
				}
			}
		}

		if (!valid_depths.empty()) {
			// Return median
			std::nth_element(valid_depths.begin(),
			                 valid_depths.begin() + valid_depths.size()/2,
			                 valid_depths.end());
			return valid_depths[valid_depths.size()/2];
		}
	}

	return 0; // Not found
}

void MCMCPatchMatch::SmoothLowConfidenceRegions(DepthData& depthData)
{
	// Simple bilateral filter on low confidence regions
	const int width = depthData.depthMap.cols;
	const int height = depthData.depthMap.rows;

	DepthMap smoothed = depthData.depthMap.clone();

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			const float conf = confidence_map.at<float>(y, x);

			if (conf < 0.5f) { // Low confidence
				// 3x3 weighted average
				float sum_weight = 0;
				float sum_depth = 0;

				for (int dy = -1; dy <= 1; dy++) {
					for (int dx = -1; dx <= 1; dx++) {
						const Depth d = depthData.depthMap(y + dy, x + dx);
						if (d > 0) {
							const float w = 1.0f; // Could be distance-weighted
							sum_weight += w;
							sum_depth += w * d;
						}
					}
				}

				if (sum_weight > 0) {
					smoothed(y, x) = sum_depth / sum_weight;
				}
			}
		}
	}

	depthData.depthMap = smoothed;
}

// ============================================================================
// Statistics
// ============================================================================

void MCMCPatchMatch::Statistics::Print() const
{
	DEBUG("MCMC-PatchMatch Statistics:");
	DEBUG("  Total pixels: %u", (unsigned)total_pixels);
	DEBUG("  Region classification:");
	DEBUG("    - Strong texture: %u (%.1f%%)", (unsigned)strong_texture_pixels,
	      100.0f * strong_texture_pixels / total_pixels);
	DEBUG("    - Medium texture: %u (%.1f%%)", (unsigned)medium_texture_pixels,
	      100.0f * medium_texture_pixels / total_pixels);
	DEBUG("    - Low texture: %u (%.1f%%)", (unsigned)low_texture_pixels,
	      100.0f * low_texture_pixels / total_pixels);
	DEBUG("  Depth completion:");
	DEBUG("    - MCMC converged: %u (%.1f%%)", (unsigned)converged_by_mcmc,
	      100.0f * converged_by_mcmc / total_pixels);
	DEBUG("    - Plane projection: %u (%.1f%%)", (unsigned)filled_by_plane,
	      100.0f * filled_by_plane / total_pixels);
	DEBUG("    - Neighbor voting: %u (%.1f%%)", (unsigned)filled_by_voting,
	      100.0f * filled_by_voting / total_pixels);
	DEBUG("    - Depth diffusion: %u (%.1f%%)", (unsigned)filled_by_diffusion,
	      100.0f * filled_by_diffusion / total_pixels);
	DEBUG("    - Global median: %u (%.1f%%)", (unsigned)filled_by_global,
	      100.0f * filled_by_global / total_pixels);
	DEBUG("  Quality distribution:");
	DEBUG("    - High quality: %.1f%% (MCMC + plane)", high_quality_ratio * 100);
	DEBUG("    - Medium quality: %.1f%% (voting)", medium_quality_ratio * 100);
	DEBUG("    - Low quality: %.1f%% (diffusion + global)", low_quality_ratio * 100);
	DEBUG("  Average confidence: %.3f", average_confidence);
	DEBUG("  Timing:");
	DEBUG("    - Superpixel: %.3fs", time_superpixel);
	DEBUG("    - MCMC: %.3fs", time_mcmc);
	DEBUG("    - Hole filling: %.3fs", time_hole_filling);
	DEBUG("    - Total: %.3fs", time_total);
}

} // namespace MVS
