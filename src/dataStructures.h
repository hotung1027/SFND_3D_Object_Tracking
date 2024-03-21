
#ifndef dataStructures_h
#define dataStructures_h

#include <map>
#include <opencv2/core.hpp>
#include <queue>
#include <vector>

struct LidarPoint {  // single lidar point in space
  double x, y, z, r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both
                     // 2D and 3D data)

  int boxID;   // unique identifier for this bounding box
  int trackID; // unique identifier for the track to which this bounding box
               // belongs

  cv::Rect roi;      // 2D region-of-interest in image coordinates
  int classID;       // ID based on class file provided to YOLO framework
  double confidence; // classification trust

  std::vector<LidarPoint>
      lidarPoints; // Lidar 3D points which project into 2D image roi
  std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
  std::vector<cv::DMatch> kptMatches;  // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same
                   // time instance

  cv::Mat cameraImg; // camera image

  std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
  cv::Mat descriptors;                 // keypoint descriptors
  std::vector<cv::DMatch>
      kptMatches; // keypoint matches between previous and current frame
  std::vector<LidarPoint> lidarPoints;

  std::vector<BoundingBox>
      boundingBoxes; // ROI around detected objects in 2D image coordinates
  std::map<int, int>
      bbMatches; // bounding box matches between previous and current frame
};
// data structure for queue the frame and allows to push the new frame to the
// back and pop the old frame from the front
struct FrameQueue {
  std::queue<DataFrame> dataBuffer;

  int BUFFER_SIZE;

  DataFrame lastFrame;
  DataFrame currentFrame;

public:
  // Constructor
  FrameQueue(int bufferSize) { this->BUFFER_SIZE = bufferSize; }
  // Destructor
  ~FrameQueue() {}

  // get the size of the queue
  int size() { return this->dataBuffer.size(); }
  // obtain the next frame from the queue
  DataFrame *getNextFrame();
  // obtain the last frame from the queue
  DataFrame *getLastFrame();
  // obtain the second last frame from the queu
  DataFrame *getSecondLastFrame();
  // Push the new frame to the back of the queue
  void push(DataFrame frame);
};
inline void FrameQueue::push(DataFrame frame) {
  if (this->dataBuffer.empty()) {
    this->dataBuffer.push(frame);
    this->lastFrame = frame;
    this->currentFrame = frame;

    return;
  }
  if (this->dataBuffer.size() < (unsigned long)BUFFER_SIZE) {
    this->dataBuffer.push(frame);
    this->currentFrame = this->lastFrame;
    this->lastFrame = frame;

  } else {
    this->dataBuffer.pop();
    this->dataBuffer.push(frame);
    this->currentFrame = this->lastFrame;
    this->lastFrame = frame;
  }
}
inline DataFrame *FrameQueue::getNextFrame() {
  DataFrame *frame = &(this->dataBuffer.front());
  this->dataBuffer.pop();
  return frame;
};
inline DataFrame *FrameQueue::getLastFrame() { return &(this->lastFrame); };
inline DataFrame *FrameQueue::getSecondLastFrame() {
  return &(this->currentFrame);
}

#endif /* dataStructures_h */
