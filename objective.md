
## FP.1 Match 3D Objects
the bounding Box (bbox) matching search bbox paris and compare the keypoints within each roi, and find the highest matched pairs
(camFusion_Student.cpp:274)

````cpp
void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {
  // push best match bounding box from previous frame into current frame
  std::unordered_multimap<int, int>
      bbPair; // box id pair <previous BoxID, current BoxID>
  std::multimap<int, int> currKptBboxPair; // match index <-> box id pair
  std::multimap<int, int> prevKptBboxPair; // match index <-> box id pair

  // search in the current bounding boxes
  // assume each keypoint can exist in multiple bounding boxes
  // search in the previous bounding boxes
  // if has matched in both frames, push to the multimap

  for (auto it = currFrame.boundingBoxes.begin();
       it != currFrame.boundingBoxes.end(); it++) {
    for (auto match : matches) {
      cv::KeyPoint currKpt = currFrame.keypoints.at(match.trainIdx);
      if (it->roi.contains(currKpt.pt)) {
        currKptBboxPair.insert(std::make_pair(match.trainIdx, it->boxID));
      }
    }
  }

  for (auto it = prevFrame.boundingBoxes.begin();
       it != prevFrame.boundingBoxes.end(); it++) {
    for (auto match : matches) {
      cv::KeyPoint prevKpt = prevFrame.keypoints.at(match.queryIdx);
      if (it->roi.contains(prevKpt.pt)) {
        prevKptBboxPair.insert(std::make_pair(match.queryIdx, it->boxID));
      }
    }
  }

  for (auto match : matches) {
    auto currRange = currKptBboxPair.equal_range(match.trainIdx);
    auto prevRange = prevKptBboxPair.equal_range(match.queryIdx);

    for (auto prevIt = prevRange.first; prevIt != prevRange.second; prevIt++) {

      for (auto currIt = currRange.first; currIt != currRange.second;
           currIt++) {
        bbPair.insert(std::make_pair(prevIt->second, currIt->second));
      }
    }
  }

  // loop through each bounding box in previous frame
  for (auto it = prevFrame.boundingBoxes.begin();
       it != prevFrame.boundingBoxes.end(); ++it) {

    int prevBoxID = it->boxID;
    auto range = bbPair.equal_range(prevBoxID);
    int maxMatches = 0;
    int currBoxID = -1;

    std::map<int, int> cntBbox; // boxid , count
    //
    for (auto it2 = range.first; it2 != range.second; ++it2) {
      if (it2->first == prevBoxID) {
        cntBbox[it2->second]++;
        if (cntBbox[it2->second] > maxMatches) {
          maxMatches = cntBbox[it2->second];
          currBoxID = it2->second;
        }
      }
    }
    // bounding box pair should be <previous boxid, current boxid>
    bbBestMatches[prevBoxID] = currBoxID;
  }
  // bounding box no kpt and dmatch, only have lidar point
}
````

## FP.2 Compute Lidar-based TTC
The approach:
- sort the lidar points base on forward distance (x)
- find the mode of first 5% qunatile of elements base on the histogram distribution
- however the bins are continuous, so we comupte the mean of first 5% qunatile instead 

(camFusion_Student.cpp:239)

````cpp
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate,
                     double &TTC) {
  // TODO : Compute Time to Collision (TTC) based on Lidar point
  // correspondences
  double dT = 1 / frameRate;
  double minXCurr = 0;
  double minXPrev = 0;

  // sort lidar points with it distance to car in forward direction(X)
  std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(),
            [](LidarPoint a, LidarPoint b) { return a.x < b.x; });
  std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(),
            [](LidarPoint a, LidarPoint b) { return a.x < b.x; });

  double distDiffFrame = 0.0;
  double iqrCurr = lidarPointsCurr.at((int)(0.75 * lidarPointsCurr.size())).x -
                   lidarPointsCurr.at((int)(0.25 * lidarPointsCurr.size())).x;

  double iqrPrev = lidarPointsPrev.at((int)(0.75 * lidarPointsPrev.size())).x -
                   lidarPointsPrev.at((int)(0.25 * lidarPointsPrev.size())).x;

  double fqrCurr = lidarPointsCurr.at((int)(0.1 * lidarPointsCurr.size())).x;
  double fqrPrev = lidarPointsPrev.at((int)(0.1 * lidarPointsPrev.size())).x;

  double lqrCurr = lidarPointsCurr.at((int)(0.3 * lidarPointsCurr.size())).x;
  double lqrPrev = lidarPointsPrev.at((int)(0.3 * lidarPointsPrev.size())).x;

  int cntCurr = 0, cntPrev = 0;

  for (auto pt : lidarPointsCurr) {
    double xCurr = pt.x;
    if (xCurr < fqrCurr - iqrCurr || xCurr > lqrCurr + iqrCurr) {

    } else {
      minXCurr += xCurr;
      cntCurr++;
    }
  }
  for (auto pt : lidarPointsPrev) {
    double xPrev = pt.x;
    if (xPrev < fqrPrev - iqrPrev || xPrev > lqrPrev + iqrPrev) {
    } else {
      minXPrev += xPrev;
      cntPrev++;
    }
  }

  minXPrev /= cntPrev;
  minXCurr /= cntCurr;

  // cout << "minXPrev: " << minXPrev << ", minXCurr: " << minXCurr << endl;
  // compute velocity
  double vel = -(minXCurr - minXPrev) / dT;

  // compute time-to-collision (TTC) based on velocity
  TTC = minXCurr / vel;
}
````

## FP.3 Associate Keypoint Correspondences with Bounding Boxes
- compute the meanMatchDistance
- filter out elements outside the threshold
- puts keypoints from matches into the bbox if it also lies in the bbox
- 
````cpp
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches) {

  // compute euclidian distance of matches
  std::vector<double> matchDistances;
  auto computeEculidianDistance = [&](cv::DMatch it) {
    cv::KeyPoint l = kptsPrev[it.queryIdx];
    cv::KeyPoint r = kptsCurr[it.trainIdx];
    double x = l.pt.x - r.pt.x;
    double y = l.pt.y - r.pt.y;
    return sqrt(x * x + y * y);
  };
  for (auto it : kptMatches) {
    double dist = computeEculidianDistance(it);
    matchDistances.push_back(dist);
  }

  std::sort(matchDistances.begin(), matchDistances.end());
  double medianMatchDistance = matchDistances[matchDistances.size() / 2];
  double threshold = 1.8 * medianMatchDistance;

  // Find keypoints in the Bounding Box
  for (auto match : kptMatches) {
    if (computeEculidianDistance(match) > threshold) {
      continue;
    }
    bool
        // inPrev = boundingBox.roi.contains(kptsPrev[match.queryIdx].pt),
        inCurr = boundingBox.roi.contains(kptsCurr[match.trainIdx].pt);
    if (inCurr) {
      boundingBox.kptMatches.push_back(match);
      boundingBox.keypoints.push_back(kptsCurr[match.trainIdx]);
    }
  }
}
````

## FP.4 Compute Camera-based TTC
- compute the TTC with keypoints distance ratios from delta frame
- 
````cpp
// Compute time-to-collision (TTC) based on keypoint correspondences in
// successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate,
                      double &TTC, cv::Mat *visImg) {
  // compute distance ratios between all matched keypoints
  vector<double> distRatios; // stores the distance ratios for all keypoints
                             // between curr. and prev. frame

  //  Associate keypoint correspondences in successive images with
  // bounding Boxes
  for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1;
       ++it1) { // outer keypoint loop

    // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
    cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

    for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end();
         ++it2) { // inner keypoint loop

      double minDist = 100;
      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
      cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

      // compute distances and distance ratios
      double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
      double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

      if (distPrev > std::numeric_limits<double>::epsilon() &&
          distCurr >= minDist) { // avoid division by zero

        double distRatio = distCurr / distPrev;
        if (distRatio > 1) {
          distRatios.push_back(distRatio);
        }
      }
    } // eof inner loop over all matched kpts
  }   // eof outer loop over all matched kpts

  // only continue if list of distance ratios is not empty
  if (distRatios.size() == 0) {
    TTC = NAN;
    return;
  }

  //  Compute TTC based on keypoint correspondences in successive images
  // compute camera-based TTC from distance ratios
  std::sort(distRatios.begin(), distRatios.end(), [](double a, double b) {
    return a < b;
  }); // sort in ascending order
  // pick median distance ratio
  double medianDistRatio = distRatios[(int)(distRatios.size() * 0.5)];
  // cout << "medianDistRatio = " << medianDistRatio << endl;

  // double meanDistRatio =
  //     std::accumulate(distRatios.begin(), distRatios.end(), 0.0) /
  //     distRatios.size();
  double dT = 1 / frameRate;
  TTC = -dT / (1 - medianDistRatio);

  //  Compute TTC based on keypoint correspondences in successive images
  // compute camera-based TTC from distance ratios
  double meanDistRatio =
      std::accumulate(distRatios.begin(), distRatios.end(), 0.0) /
      distRatios.size();

  double dT = 1 / frameRate;
  TTC = -dT / (1 - meanDistRatio);
}
````

## FP.5 Performance Evaluation 1
Because the TTC are base on the estimation of velocity, which are compute from the difference of lidar point from previous frame and current frame, it creates a problem when the estimation are inaccurate, such as when the lidar distance between car and sensors does not changed between previous frame and current frame. Thus, the estimation of velocity suddenly become extreme small, and the TTC become extremely long.

For example in this project, the velocity supposed to be consistant throughout different frames which TTC lies on average of 13s, however there is a frame that the minX produce a insignificant difference from previous frame (0.01m), it creates an estimation of very small velocity (0.1m/s), therefore the TTC become inaccurate for such scenario. 

The weakness of this discrete time velocity model, relies on the detection of successful consecutive frame. Therefore outliers and sparse lidar points distribute over the surface, which could occur when objects surface a too large and curve, far sight object at high chasing velocity, creates an inaccurate observation or unable to observe. 
Due to too little samples for:
- velocity calculation
- distance estimation
- no matching frame

Thus lidar estimation might not fit into these scenario.



## FP.6 Performance Evaluation 2
if we look at the comparison chart below, BRISK descriptor has the best accuracy, Combine with FAST and AKAZE, they produce a steady and confidence TTC,
however, other descriptor and detectors often encounter problems in keypoints matching, thus no keypoints match, and therefore no dist_ratio or dist_ratio small in compute_ttc_camera
![](./data_comparison.jpeg)
The chart below also can see some of the combination of descriptor + detector produce a negative TTC, sometimes NAN value.
![](./perf_eval.jpeg)
