//K-D Tree
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

int main()
{
    Mat refImg = imread("../images/ref.jpg", IMREAD_COLOR);
    Mat targetImg = imread("../images/target3.jpg", IMREAD_COLOR);

    if (refImg.empty() || targetImg.empty())
    {
        cerr << "Error loading images!" << endl;
        return -1;
    }

    Ptr<SIFT> detector = SIFT::create();
    vector<KeyPoint> refKp, targetKp;
    Mat refDesc, targetDesc;

    detector->detectAndCompute(refImg, noArray(), refKp, refDesc);
    detector->detectAndCompute(targetImg, noArray(), targetKp, targetDesc);

    if (refDesc.type() != CV_32F) refDesc.convertTo(refDesc, CV_32F);
    if (targetDesc.type() != CV_32F) targetDesc.convertTo(targetDesc, CV_32F);

    // Build KD-Tree using reference descriptors
    flann::Index kdTree(refDesc, flann::KDTreeIndexParams(5));

    // Perform k-NN search for each target descriptor
    int k = 2;
    Mat indices(targetDesc.rows, k, CV_32S);
    Mat dists(targetDesc.rows, k, CV_32F);
    kdTree.knnSearch(targetDesc, indices, dists, k, flann::SearchParams());

    // Apply Lowe’s ratio test
    vector<DMatch> goodMatches;
    float ratio_thresh = 0.7f;
    for (int i = 0; i < indices.rows; i++)
    {
        if (dists.at<float>(i, 0) < ratio_thresh * dists.at<float>(i, 1))
        {
            DMatch m;
            m.queryIdx = i;                      
            m.trainIdx = indices.at<int>(i, 0);  // Closest in reference
            m.distance = dists.at<float>(i, 0);
            goodMatches.push_back(m);
        }
    }

    // Retry with relaxed threshold if too few matches
    if (goodMatches.size() < 15)
    {
        ratio_thresh = 0.8f;
        goodMatches.clear();
        for (int i = 0; i < indices.rows; i++)
        {
            if (dists.at<float>(i, 0) < ratio_thresh * dists.at<float>(i, 1))
            {
                DMatch m;
                m.queryIdx = i;
                m.trainIdx = indices.at<int>(i, 0);
                m.distance = dists.at<float>(i, 0);
                goodMatches.push_back(m);
            }
        }
    }

    // Geometric verification
    if (goodMatches.size() >= 10)
    {
        vector<Point2f> targetPts, refPts;
        for (const auto& m : goodMatches)
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
            for (const auto& p : refCorners)
                center += p;
            center *= 0.25f;

            float radius = 0;
            for (const auto& p : refCorners)
                radius = std::max(radius, static_cast<float>(norm(p - center)));

            Mat result = refImg.clone();
            circle(result, center, radius * 1.1f, Scalar(0, 0, 255), 3);
            imshow("Result - KDTree Matching", result);
            imwrite("kdtree_matched_result.jpg", result);
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