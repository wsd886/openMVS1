/*
* SuperpixelSegmenter.cpp
*/

#include "SuperpixelSegmenter.h"
#include "../Common/AutoEstimator.h"
#include <opencv2/imgproc.hpp>

using namespace MVS;

// Superpixel plane fitting
void Superpixel::FitPlane(const DepthData& depthData, float conf_threshold)
{
	const Camera& camera = depthData.GetCamera();

	// Collect reliable 3D points
	Point3fArr points;
	points.reserve(pixels.size());

	num_reliable_pixels = 0;
	float conf_sum = 0, depth_sum = 0;

	for (const auto& pixel : pixels) {
		const Depth depth = depthData.depthMap(pixel);
		if (depth <= 0) continue;

		const float conf = depthData.confMap.empty() ? 1.f : depthData.confMap(pixel);
		if (conf < conf_threshold) continue;

		// Backproject to 3D
		const Point3 X = camera.TransformPointI2W(Point3(Point2f(pixel), depth));
		points.emplace_back((float)X.x, (float)X.y, (float)X.z);

		num_reliable_pixels++;
		conf_sum += conf;
		depth_sum += depth;
	}

	avg_confidence = num_reliable_pixels > 0 ? conf_sum / num_reliable_pixels : 0;
	avg_depth = num_reliable_pixels > 0 ? depth_sum / num_reliable_pixels : 0;

	// Need at least 3 points for plane fitting
	if (points.size() < 3) {
		has_plane = false;
		plane_confidence = 0;
		return;
	}

	// RANSAC plane fitting
	double max_threshold = 0.1; // 10cm error
	const unsigned num_inliers = EstimatePlaneTh(points, plane, max_threshold);

	// Confidence based on inlier ratio
	plane_confidence = (float)num_inliers / points.size();
	has_plane = (plane_confidence > 0.7f);

	if (has_plane) {
		normal = Cast<Normal::Type>(plane.m_vN);

		// Compute depth at centroid
		const Point3f X_center = camera.TransformPointI2C(Point3f(
			(float)center.x, (float)center.y, 1.f
		));
		depth_center = (Depth)(-plane.m_fD / plane.m_vN.dot(X_center));

		// Ensure normal points toward camera
		if (normal.dot(Cast<Normal::Type>(X_center)) > 0)
			normal = -normal;
	}
}

// Project pixel to plane
Depth Superpixel::ProjectDepth(const ImageRef& x, const Camera& camera) const
{
	if (!has_plane)
		return 0;

	// Ray direction in camera space
	const Point3f ray = camera.TransformPointI2C(Point3f((float)x.x, (float)x.y, 1.f));

	// Ray-plane intersection: n·(t*ray) + d = 0
	const float denominator = plane.m_vN.dot(ray);
	if (ABS(denominator) < 1e-6f)
		return 0;

	const float t = -plane.m_fD / denominator;

	return (t > 0) ? (Depth)t : 0;
}

// Compute superpixel statistics
void Superpixel::ComputeStatistics(const DepthData& depthData)
{
	if (pixels.empty()) return;

	const Image32F& image = depthData.images.front().image;
	const DepthMap& depthMap = depthData.depthMap;

	// Color mean
	Pixel64F color_sum(0, 0, 0);
	float depth_sum = 0, depth_sqsum = 0;
	float texture_sum = 0;
	int valid_count = 0;

	for (const auto& pixel : pixels) {
		// Color
		if (image.isInside(pixel)) {
			const float val = image(pixel);
			color_sum.r += val;
			color_sum.g += val;
			color_sum.b += val;
		}

		// Depth
		const Depth depth = depthMap(pixel);
		if (depth > 0) {
			depth_sum += depth;
			depth_sqsum += depth * depth;
			valid_count++;
		}

		// Texture (gradient magnitude)
		if (pixel.x > 0 && pixel.x < image.width()-1 &&
		    pixel.y > 0 && pixel.y < image.height()-1) {
			const float gx = image(pixel.x+1, pixel.y) - image(pixel.x-1, pixel.y);
			const float gy = image(pixel.x, pixel.y+1) - image(pixel.x, pixel.y-1);
			texture_sum += gx*gx + gy*gy;
		}
	}

	const int n = (int)pixels.size();
	avg_color = Pixel32F((float)(color_sum.r/n), (float)(color_sum.g/n), (float)(color_sum.b/n));
	texture_variance = texture_sum / n;

	if (valid_count > 0) {
		avg_depth = depth_sum / valid_count;
		depth_variance = depth_sqsum / valid_count - avg_depth * avg_depth;
	}
}

// Check depth consistency with plane
bool Superpixel::IsDepthConsistentWithPlane(const ImageRef& x, Depth depth, const Camera& camera, float threshold) const
{
	if (!has_plane || depth <= 0)
		return false;

	const Depth depth_plane = ProjectDepth(x, camera);
	if (depth_plane <= 0)
		return false;

	const float rel_diff = ABS(depth - depth_plane) / depth;
	return rel_diff < threshold;
}

// Constructor
SuperpixelSegmenter::SuperpixelSegmenter(const SegmentationConfig& config)
	: config(config)
{
}

SuperpixelSegmenter::~SuperpixelSegmenter()
{
}

// Basic segmentation
int SuperpixelSegmenter::Segment(const Image32F& image, const BitMatrix* pMask)
{
	TD_TIMER_START();

	image_size = cv::Size(image.cols, image.rows);

	// Convert to OpenCV format
	cv::Mat cv_image;
	cv::cvtColor(cv::Mat(image.rows, image.cols, CV_32FC1, (void*)image.data),
	             cv_image, cv::COLOR_GRAY2BGR);
	cv_image.convertTo(cv_image, CV_8UC3, 255.0);

	// Apply mask if provided
	if (pMask) {
		for (int y = 0; y < image.rows; y++) {
			for (int x = 0; x < image.cols; x++) {
				if (!(*pMask)(y, x)) {
					cv_image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
				}
			}
		}
	}

	// Segment
	SegmentSLIC(cv_image);

	// Build superpixel list
	BuildSuperpixelList();
	ComputeNeighborhood();

	DEBUG_ULTIMATE("Superpixel segmentation: %d superpixels (%s)",
	               GetNumSuperpixels(), TD_TIMER_GET_FMT().c_str());

	return GetNumSuperpixels();
}

// Depth-guided segmentation (INNOVATION!)
int SuperpixelSegmenter::SegmentWithDepthGuidance(const Image32F& image, const DepthMap& depthMap,
                                                   const NormalMap* pNormalMap, const BitMatrix* pMask)
{
	TD_TIMER_START();

	image_size = cv::Size(image.cols, image.rows);

	// Convert image to OpenCV
	cv::Mat cv_image;
	cv::cvtColor(cv::Mat(image.rows, image.cols, CV_32FC1, (void*)image.data),
	             cv_image, cv::COLOR_GRAY2BGR);
	cv_image.convertTo(cv_image, CV_8UC3, 255.0);

	// Convert depth to OpenCV (normalized)
	cv::Mat cv_depth(depthMap.rows, depthMap.cols, CV_32FC1);
	Depth dMin = FLT_MAX, dMax = 0;
	for (int y = 0; y < depthMap.rows; y++) {
		for (int x = 0; x < depthMap.cols; x++) {
			const Depth d = depthMap(y, x);
			cv_depth.at<float>(y, x) = d;
			if (d > 0) {
				if (d < dMin) dMin = d;
				if (d > dMax) dMax = d;
			}
		}
	}

	// Normalize depth to [0, 255]
	if (dMax > dMin) {
		cv_depth = (cv_depth - dMin) / (dMax - dMin) * 255.0f;
	}
	cv_depth.convertTo(cv_depth, CV_8UC1);

	// Merge RGB and Depth as 4-channel image
	cv::Mat channels[4];
	cv::split(cv_image, channels);
	channels[3] = cv_depth;  // Add depth as 4th channel

	cv::Mat rgbd_image;
	cv::merge(channels, 4, rgbd_image);

	// Apply depth weight
	// Adjust depth channel intensity
	rgbd_image.at<cv::Vec<uint8_t, 4>>(0, 0)[3] *= config.depth_weight;

	// Segment with depth-augmented image
	SegmentSLIC(rgbd_image);

	// Build superpixels
	BuildSuperpixelList();
	ComputeNeighborhood();

	// Detect boundary superpixels
	DetectBoundarySuperpixels(depthMap, config.depth_discontinuity_threshold);

	DEBUG_ULTIMATE("Depth-guided segmentation: %d superpixels (%s)",
	               GetNumSuperpixels(), TD_TIMER_GET_FMT().c_str());

	return GetNumSuperpixels();
}

// SLIC segmentation
void SuperpixelSegmenter::SegmentSLIC(const cv::Mat& image)
{
	// Determine algorithm based on number of channels
	int algorithm;
	if (image.channels() == 3) {
		algorithm = cv::ximgproc::SLIC;
	} else if (image.channels() == 4) {
		algorithm = cv::ximgproc::SLICO;  // SLIC Zero for multi-channel
	} else {
		algorithm = cv::ximgproc::SLIC;
	}

	slic = cv::ximgproc::createSuperpixelSLIC(image, algorithm,
	                                          config.region_size, config.ruler);
	slic->iterate(config.num_iterations);
	slic->enforceLabelConnectivity(25);
	slic->getLabels(labels);
}

// Build superpixel list from label map
void SuperpixelSegmenter::BuildSuperpixelList()
{
	const int num_superpixels = slic->getNumberOfSuperpixels();
	superpixels.resize(num_superpixels);

	// Initialize
	for (int i = 0; i < num_superpixels; i++) {
		superpixels[i].id = i;
		superpixels[i].pixels.clear();
	}

	// Collect pixels
	for (int y = 0; y < labels.rows; y++) {
		for (int x = 0; x < labels.cols; x++) {
			const int sp_id = labels.at<int>(y, x);
			if (sp_id >= 0 && sp_id < num_superpixels) {
				superpixels[sp_id].pixels.emplace_back(x, y);
			}
		}
	}

	// Compute centroids
	for (auto& sp : superpixels) {
		if (sp.pixels.empty()) continue;

		int sum_x = 0, sum_y = 0;
		for (const auto& p : sp.pixels) {
			sum_x += p.x;
			sum_y += p.y;
		}
		sp.center.x = sum_x / sp.pixels.size();
		sp.center.y = sum_y / sp.pixels.size();
	}
}

// Compute superpixel neighborhood
void SuperpixelSegmenter::ComputeNeighborhood()
{
	std::set<std::pair<int, int>> edges;

	// Find adjacent superpixels
	for (int y = 0; y < labels.rows-1; y++) {
		for (int x = 0; x < labels.cols-1; x++) {
			const int id = labels.at<int>(y, x);

			// Check right neighbor
			const int id_right = labels.at<int>(y, x+1);
			if (id != id_right) {
				edges.insert({std::min(id, id_right), std::max(id, id_right)});
			}

			// Check bottom neighbor
			const int id_bottom = labels.at<int>(y+1, x);
			if (id != id_bottom) {
				edges.insert({std::min(id, id_bottom), std::max(id, id_bottom)});
			}
		}
	}

	// Build neighbor lists
	for (const auto& edge : edges) {
		const int id1 = edge.first;
		const int id2 = edge.second;

		if (id1 >= 0 && id1 < (int)superpixels.size())
			superpixels[id1].neighbor_ids.push_back(id2);

		if (id2 >= 0 && id2 < (int)superpixels.size())
			superpixels[id2].neighbor_ids.push_back(id1);
	}
}

// Detect boundary superpixels based on depth discontinuity
void SuperpixelSegmenter::DetectBoundarySuperpixels(const DepthMap& depthMap, float threshold)
{
	for (auto& sp : superpixels) {
		sp.is_boundary = false;

		if (sp.pixels.empty()) continue;

		// Check depth difference with neighbors
		for (const int neighbor_id : sp.neighbor_ids) {
			if (neighbor_id < 0 || neighbor_id >= (int)superpixels.size())
				continue;

			const Superpixel& neighbor = superpixels[neighbor_id];

			// Compare average depths
			if (sp.avg_depth > 0 && neighbor.avg_depth > 0) {
				const float depth_diff = ABS(sp.avg_depth - neighbor.avg_depth);
				const float rel_diff = depth_diff / std::max(sp.avg_depth, neighbor.avg_depth);

				if (rel_diff > threshold) {
					sp.is_boundary = true;
					break;
				}
			}
		}
	}
}

// Fit planes to all superpixels
void SuperpixelSegmenter::FitPlanes(const DepthData& depthData, float conf_threshold)
{
	TD_TIMER_START();

	int num_fitted = 0;

	#ifdef DEPTHMAP_USE_OPENMP
	#pragma omp parallel for reduction(+:num_fitted)
	#endif
	for (int i = 0; i < (int)superpixels.size(); i++) {
		superpixels[i].ComputeStatistics(depthData);
		superpixels[i].FitPlane(depthData, conf_threshold);
		if (superpixels[i].has_plane)
			num_fitted++;
	}

	DEBUG_ULTIMATE("Plane fitting: %d/%d superpixels have reliable planes (%s)",
	               num_fitted, (int)superpixels.size(), TD_TIMER_GET_FMT().c_str());
}

// Get superpixel ID for pixel
int SuperpixelSegmenter::GetSuperpixelID(const ImageRef& x) const
{
	if (x.x < 0 || x.x >= labels.cols || x.y < 0 || x.y >= labels.rows)
		return -1;
	return labels.at<int>(x.y, x.x);
}

// Visualize superpixels
Image8U3 SuperpixelSegmenter::VisualizeSuperpixels() const
{
	cv::Mat mask;
	slic->getLabelContourMask(mask, true);

	Image8U3 vis(labels.rows, labels.cols);
	for (int y = 0; y < labels.rows; y++) {
		for (int x = 0; x < labels.cols; x++) {
			if (mask.at<uchar>(y, x) == 255) {
				vis(y, x) = Pixel8U(255, 0, 0); // Red boundary
			} else {
				int sp_id = labels.at<int>(y, x);
				// Random color per superpixel
				unsigned char r = (sp_id * 71) % 255;
				unsigned char g = (sp_id * 151) % 255;
				unsigned char b = (sp_id * 211) % 255;
				vis(y, x) = Pixel8U(r, g, b);
			}
		}
	}
	return vis;
}

// Visualize planes
Image8U3 SuperpixelSegmenter::VisualizePlanes() const
{
	Image8U3 vis(labels.rows, labels.cols, Pixel8U(0, 0, 0));

	for (const auto& sp : superpixels) {
		if (!sp.has_plane) continue;

		// Color based on plane confidence
		const unsigned char intensity = (unsigned char)(sp.plane_confidence * 255);
		const Pixel8U color(intensity, intensity, intensity);

		for (const auto& pixel : sp.pixels) {
			vis(pixel) = color;
		}
	}
	return vis;
}

// Query superpixel ID
int SuperpixelSegmenter::GetSuperpixelID(const ImageRef& x) const
{
	if (x.x < 0 || x.x >= labels.cols || x.y < 0 || x.y >= labels.rows)
		return -1;
	return labels.at<int>(x.y, x.x);
}
