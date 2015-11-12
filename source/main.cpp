#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <list>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	const int MIN_SMOOTH_FRAMES = 10; // minimum number of consecutive non-jump frames before switching to "hard" tracking
	const int LAST_POS_BUFFER_SIZE = 6; // number of saved previous positions to average from

	// variable initialization
	int keyInput = 0;
	int nFrames = 0, nSmoothFrames = 0, nFailedFrames = 0;
	bool bJump = false;
	bool bHardTracking = false;
	Mat curFrame, fgMaskMOG2, fgMaskKNN, bgImg, frameDelta, frameDeltaThr, frameDil, grayFrame;
	const char* filename = "../test/Video01_09Nov2015.mp4\0";
	Ptr<BackgroundSubtractor> pKNN;
	Ptr<BackgroundSubtractor> pMOG2;
	Rect lastSmoothRect;
	Point lastSmoothPos;

	VideoWriter outputVideo;

	const unsigned int DELTA_SQ_THRESH = 20000;
	const unsigned int CONTOUR_AREA_THRESH = 400;

	Point lastPos, pRectMid;

	vector<Mat> contours;
	vector<Vec4i> hierarchy;
	list<Point> lastKnownPositions;

	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(9, 9));

	// debug output windows
	//namedWindow("Frame");
	namedWindow("Motion tracking");

	VideoCapture capture(filename);
	if (!capture.isOpened())
	{
		cerr << "Unable to open file '" << filename << "'." << endl;
		return EXIT_FAILURE;
	}
	else
	{
		cout << "Successfully opened file '" << filename << "'." << endl;
	}

	Size vidS = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	int fourcc = capture.get(CV_CAP_PROP_FOURCC);
	outputVideo.open("test_out.avi", CV_FOURCC('P','I','M','1'), capture.get(CV_CAP_PROP_FPS), vidS, true);
	if (!outputVideo.isOpened())
	{
		cerr << "Unable to write to output video." << endl;
		return EXIT_FAILURE;
	}

	// build frame buffer and background subtractor
	pKNN = createBackgroundSubtractorKNN();
	pMOG2 = createBackgroundSubtractorMOG2(500, 30., true);

	// build mog once
	//while (capture.read(curFrame))
	//{
	//	++nFrames;
	//	pMOG2->apply(curFrame, fgMaskMOG2);
	//}
	//capture.release();
	//capture.open(filename); // free and open once more
	//cout << "Processed " << nFrames << " frames." << endl;

	while (capture.read(curFrame) && (char)keyInput != 'q')
	{
		++nFrames;
		//GaussianBlur(curFrame, curFrame, Size(15, 15), 0, 0);
		
		cvtColor(curFrame, grayFrame, CV_BGR2GRAY);	// convert to grayscale
		
		// try to eliminate (white) reflections by truncating the current frame
		threshold(grayFrame, grayFrame, 128., 0., CV_THRESH_TRUNC);
		GaussianBlur(grayFrame, grayFrame, Size(7, 7), 0, 0);

		pMOG2->apply(grayFrame, fgMaskMOG2);

		//absdiff(bgImg, curFrame, frameDelta); // absolute difference between background image and current frame
		//threshold(fgMaskMOG2, frameDeltaThr, 10., 255., THRESH_BINARY);
		dilate(fgMaskMOG2, frameDil, dilateElement);

		Mat canny_out;
		Canny(frameDil, canny_out, 100, 200, 3);

		Mat copy;
		canny_out.copyTo(copy);

		findContours(copy, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		// determine largest "moving" object
		Rect maxRect;
		int iMaxSize = 0;
		for (unsigned int i = 0; i < contours.size(); i++)
		{
			if (contourArea(contours[i]) < CONTOUR_AREA_THRESH) // ignore contours which are too small (noise)
				continue;
			Rect br = boundingRect(contours[i]); // bounding rect

			if (br.width*br.height > iMaxSize)
			{
				maxRect = br;
				iMaxSize = br.width*br.height;
				pRectMid = 0.5 * (maxRect.tl() + maxRect.br());
			}
		}

		if (nFrames > 1)
		{
			// determine "medium" recent coordinates
			uint mx = 0; uint my = 0;

			if (bHardTracking)
			{
				for (list<Point>::iterator it = lastKnownPositions.begin(); it != lastKnownPositions.end(); it++)
				{
					mx += (*it).x; my += (*it).y;
				}

				mx /= lastKnownPositions.size(); my /= lastKnownPositions.size();
			}
			else
			{
				mx = lastPos.x; my = lastPos.y;
			}
			
			int dx = pRectMid.x - mx;
			int dy = pRectMid.y - my;
			unsigned int deltaSq = dx*dx + dy*dy;
			//cout << deltaSq << endl;
			
			if (deltaSq > DELTA_SQ_THRESH)
			{
				bJump = true;
				//if (!bHardTracking && nSmoothFrames > MIN_SMOOTH_FRAMES)
				//{
				//	cout << nSmoothFrames << " smooth frames. Switching to hard tracking." << endl;
				//	bHardTracking = true;
				//}
				nSmoothFrames = 0;
				++nFailedFrames;
				//if (bHardTracking && nFailedFrames > MIN_SMOOTH_FRAMES) // lost track
				//{
				//	cout << "Lost track. Back to soft mode." << endl;
				//	bHardTracking = false;
				//}
			}
			else
			{
				bJump = false;
				++nSmoothFrames;
				nFailedFrames = 0;
				lastSmoothRect = maxRect;
				lastSmoothPos = pRectMid;
			}
		}
		lastPos = pRectMid;

		// draw only the largest object
		if (!bJump)
		{
			if (lastKnownPositions.size() > LAST_POS_BUFFER_SIZE)
				lastKnownPositions.pop_front();
			lastKnownPositions.push_back(pRectMid);

			//list<Point>::iterator it;
			//int i = 0;
			//for (it = lastKnownPositions.begin(); it != lastKnownPositions.end(); it++)
			//{
			//	Scalar color(150, 150, 150);
			//	circle(curFrame, *it, 5, color, 2 * i);
			//	++i;
			//}

			rectangle(curFrame, maxRect, Scalar(0, 255, 0), 3);
			circle(curFrame, pRectMid, 5, Scalar(255, 0, 0), 10, 8);
		}
		else
		{
			// guess position based on last position
			//circle(curFrame, lastKnownPositions.back(), 5, Scalar(128, 0, 0), 10, 8);

			//rectangle(curFrame, lastSmoothRect, Scalar(0, 160, 0), 3);
			//circle(curFrame, lastSmoothPos, 5, Scalar(0, 0, 255), 10, 8);
		}

		// draw text overlay
		ostringstream str, str2, str3;
		int line = 0;
		const int lineSkip = 16;

		// frame counter
		str << "Frame: " << nFrames;
		putText(curFrame, str.str(), Point(10, 22+line*lineSkip), CV_FONT_HERSHEY_PLAIN, 1., Scalar(180., 0., 0.));
		++line;

		// motion coordinates
		str2 << "Motion X: " << pRectMid.x;
		putText(curFrame, str2.str(), Point(10, 22 + line*lineSkip), CV_FONT_HERSHEY_PLAIN, 1., Scalar(180., 0., 0.));
		++line;
		str3 << "Motion Y: " << pRectMid.y;
		putText(curFrame, str3.str(), Point(10, 22 + line*lineSkip), CV_FONT_HERSHEY_PLAIN, 1., Scalar(180., 0., 0.));
		++line;

		contours.clear();
		hierarchy.clear();

		outputVideo << curFrame;

		imshow("Motion tracking", curFrame);

		keyInput = waitKey(30);
	}

	waitKey(0);

	outputVideo.release();

	cout << "Read " << nFrames << " frames." << endl;
	capture.release();


	return 0;
}