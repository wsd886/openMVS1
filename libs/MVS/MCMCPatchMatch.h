/*
* MCMCPatchMatch.h
*
* MCMC-Enhanced PatchMatch for Multi-View Stereo
*
* This implements a Markov Chain Monte Carlo framework for PatchMatch stereo,
* providing theoretical guarantees and improved performance in weakly-textured regions.
*
* Key innovations:
* 1. Softmax acceptance (non-greedy, avoids local optima)
* 2. Adaptive temperature scheduling (based on photometric uncertainty)
* 3. Planar prior integration via Bayesian posterior
* 4. Guaranteed hole filling with multi-level fallback
* 5. Region-specific optimization strategies
*/

#ifndef _MVS_MCMC_PATCHMATCH_H_
#define _MVS_MCMC_PATCHMATCH_H_

#include "DepthMap.h"
#include "SuperpixelSegmenter.h"

namespace MVS {

// MCMC Configuration
struct MCMCConfig {
	// Temperature scheduling
	float beta_0;                      // Initial inverse temperature (default: 5.0)
	float beta_min;                    // Minimum beta for low-texture (default: 2.0)
	float beta_max;                    // Maximum beta for strong-texture (default: 10.0)
	float tau_uncertainty;             // Uncertainty threshold (default: 0.05)

	// Planar prior
	float lambda_prior;                // Planar prior weight (default: 1.0)
	float sigma_plane;                 // Planar variance (default: 0.5)
	float plane_confidence_threshold;  // Minimum plane confidence (default: 0.7)

	// Superpixel segmentation
	unsigned superpixel_size;          // Target superpixel size (default: 15)
	float superpixel_ruler;            // Spatial smoothness (default: 20.0)
	float superpixel_depth_weight;     // Depth channel weight (default: 0.5)

	// MCMC sampling
	unsigned max_mcmc_iterations;      // MCMC iterations per pixel (default: 3)
	unsigned num_samples_low_texture;  // Samples in low-texture (default: 5)
	unsigned num_samples_high_texture; // Samples in high-texture (default: 1)

	// Hole filling
	bool guarantee_complete_coverage;  // Force fill all holes (default: true)
	unsigned max_diffusion_radius;     // Max radius for depth diffusion (default: 10)
	float min_fill_confidence;         // Minimum confidence for valid fill (default: 0.3)

	// Region-specific
	float texture_threshold;           // Low-texture detection threshold (default: 0.05)
	bool enable_region_specific;       // Enable different strategies per region (default: true)

	// Convergence
	float convergence_threshold;       // Convergence check threshold (default: 0.01)

	MCMCConfig()
		: beta_0(5.0f), beta_min(2.0f), beta_max(10.0f), tau_uncertainty(0.05f)
		, lambda_prior(1.0f), sigma_plane(0.5f), plane_confidence_threshold(0.7f)
		, superpixel_size(15), superpixel_ruler(20.0f), superpixel_depth_weight(0.5f)
		, max_mcmc_iterations(3), num_samples_low_texture(5), num_samples_high_texture(1)
		, guarantee_complete_coverage(true), max_diffusion_radius(10), min_fill_confidence(0.3f)
		, texture_threshold(0.05f), enable_region_specific(true)
		, convergence_threshold(0.01f)
	{}
};

// Fill quality tracking
enum FillMethod {
	FILL_MCMC_CONVERGED,      // MCMC converged to valid depth
	FILL_PLANE_PROJECTION,    // High quality: planar projection
	FILL_NEIGHBOR_VOTING,     // Medium quality: neighbor plane voting
	FILL_DEPTH_DIFFUSION,     // Low quality: nearest valid depth
	FILL_GLOBAL_MEDIAN        // Worst quality: global median fallback
};

struct FillResult {
	Depth depth;
	float confidence;
	FillMethod method;

	FillResult() : depth(0), confidence(0), method(FILL_MCMC_CONVERGED) {}
	FillResult(Depth d, float conf, FillMethod m)
		: depth(d), confidence(conf), method(m) {}
};

// MCMC-PatchMatch main class
class MCMCPatchMatch {
public:
	MCMCPatchMatch(const MCMCConfig& config = MCMCConfig());
	~MCMCPatchMatch();

	// Main processing function
	bool Process(DepthData& depthData);

	// Statistics
	struct Statistics {
		// Performance
		size_t total_pixels;
		size_t converged_by_mcmc;
		size_t filled_by_plane;
		size_t filled_by_voting;
		size_t filled_by_diffusion;
		size_t filled_by_global;

		// Region classification
		size_t strong_texture_pixels;
		size_t medium_texture_pixels;
		size_t low_texture_pixels;

		// Quality metrics
		float average_confidence;
		float high_quality_ratio;  // MCMC + plane projection
		float medium_quality_ratio; // neighbor voting
		float low_quality_ratio;    // diffusion + global

		// Timing
		double time_superpixel;
		double time_mcmc;
		double time_hole_filling;
		double time_total;

		Statistics() { memset(this, 0, sizeof(Statistics)); }

		void Print() const;
	};

	const Statistics& GetStatistics() const { return stats; }

private:
	// Core MCMC functions
	float ComputePhotometricUncertainty(const DepthData& depthData, const ImageRef& x, Depth d) const;
	float ComputeAdaptiveTemperature(float photometric_uncertainty) const;
	float ComputeAcceptanceProbability(float delta_E, float beta) const;
	bool AcceptProposal(float acceptance_prob) const;

	// Energy computation
	float ComputeEnergy(const DepthData& depthData, const ImageRef& x, Depth d, const Normal& n) const;
	float ComputePhotometricEnergy(const DepthData& depthData, const ImageRef& x, Depth d, const Normal& n) const;
	float ComputePlanarEnergy(const ImageRef& x, Depth d, int superpixel_id) const;

	// MCMC sampling
	bool MCMCSample(DepthData& depthData, const ImageRef& x);

	// Superpixel & plane fitting
	bool SegmentAndFitPlanes(const DepthData& depthData);

	// Hole filling strategies
	FillResult ForceFillDepth(const DepthData& depthData, const ImageRef& x);
	FillResult FillByPlaneProjection(const ImageRef& x) const;
	FillResult FillByNeighborVoting(const ImageRef& x) const;
	FillResult FillByDepthDiffusion(const DepthData& depthData, const ImageRef& x) const;
	FillResult FillByGlobalMedian() const;

	// Utilities
	bool IsLowTexture(const DepthData& depthData, const ImageRef& x) const;
	int GetSuperpixelID(const ImageRef& x) const;
	std::vector<int> GetNeighborSuperpixels(const ImageRef& x, unsigned radius) const;
	Depth FindNearestValidDepth(const DepthData& depthData, const ImageRef& x, unsigned max_radius) const;

	// Post-processing
	void SmoothLowConfidenceRegions(DepthData& depthData);

	// Data members
	MCMCConfig config;
	Statistics stats;

	// Superpixel data
	SuperpixelSegmenter* segmenter;
	cv::Mat superpixel_labels;
	std::vector<Superpixel> superpixels;

	// Auxiliary data
	cv::Mat confidence_map;       // Pixel-wise confidence [0, 1]
	cv::Mat fill_method_map;      // Which method filled each pixel
	cv::Mat texture_map;          // Local texture variance
	float global_median_depth;    // Fallback depth

	// Random number generator
	mutable std::mt19937 rng;
	mutable std::uniform_real_distribution<float> uniform_dist;
};

} // namespace MVS

#endif // _MVS_MCMC_PATCHMATCH_H_
