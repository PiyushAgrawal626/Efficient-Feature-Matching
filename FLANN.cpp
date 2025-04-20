//FLANN
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <algorithm> 

using namespace cv;
using namespace std;

int main()
{
    // Load images
    Mat refImg = imread("../images/ref.jpg", IMREAD_COLOR);
    Mat targetImg = imread("../images/target8.jpg", IMREAD_COLOR);

    if (refImg.empty() || targetImg.empty())
    {
        cerr << "Error loading images!" << endl;
        return -1;
    }

    // 1. Feature Detection - Using built-in SIFT
    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> refKp, targetKp;
    Mat refDesc, targetDesc;

    detector->detectAndCompute(refImg, noArray(), refKp, refDesc);
    detector->detectAndCompute(targetImg, noArray(), targetKp, targetDesc);

    // 2. Feature Matching with FLANN
    if (refDesc.type() != CV_32F) refDesc.convertTo(refDesc, CV_32F);
    if (targetDesc.type() != CV_32F) targetDesc.convertTo(targetDesc, CV_32F);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knnMatches;
    matcher->knnMatch(targetDesc, refDesc, knnMatches, 2);

    // 3. Lowe's Ratio Test
    vector<DMatch> goodMatches;
    float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knnMatches.size(); i++)
    {
        if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance)
        {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    // 4. Retry with relaxed threshold if needed
    if (goodMatches.size() < 15)
    {
        ratio_thresh = 0.8f;
        goodMatches.clear();
        for (size_t i = 0; i < knnMatches.size(); i++)
        {
            if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance)
            {
                goodMatches.push_back(knnMatches[i][0]);
            }
        }
    }

    // 5. Geometric Verification
    if (goodMatches.size() >= 10)
    {
        vector<Point2f> targetPts, refPts;
        for (const auto &m : goodMatches)
        {
            targetPts.push_back(targetKp[m.queryIdx].pt);
            refPts.push_back(refKp[m.trainIdx].pt);
        }

        Mat H = findHomography(targetPts, refPts, RANSAC, 3.0);
        if (!H.empty())
        {
            vector<Point2f> targetCorners(4);
            targetCorners[0] = Point2f(0, 0);
            targetCorners[1] = Point2f((float)targetImg.cols, 0);
            targetCorners[2] = Point2f((float)targetImg.cols, (float)targetImg.rows);
            targetCorners[3] = Point2f(0, (float)targetImg.rows);

            vector<Point2f> refCorners(4);
            perspectiveTransform(targetCorners, refCorners, H);

            Point2f center(0, 0);
            for (const auto &p : refCorners)
                center += p;
            center *= 0.25f;

            float radius = 0;
            for (const auto &p : refCorners)
                radius = std::max(radius, static_cast<float>(norm(p - center)));

            Mat result = refImg.clone();
            circle(result, center, radius * 1.1f, Scalar(0, 0, 255), 3);
            imshow("Result - Target Detected", result);
            imwrite("matched_result.jpg", result); // Optional: save output
            waitKey(0);
        }
        else
        {
            cerr << "Homography could not be computed!" << endl;
        }
    }
    else
    {
        cerr << "Insufficient matches found (" << goodMatches.size() << ")" << endl;
    }

    return 0;
}

