#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <list>

using namespace cv;
using namespace std;

const unsigned int MIN_SMOOTH_FRAMES = 10; // minimum number of consecutive non-jump frames before switching to "hard" tracking
const unsigned int LAST_POS_BUFFER_SIZE = 6; // number of saved previous positions to average from
const unsigned int MAX_DIST_FRAME_TO_FRAME = 150; // maximum pixel distance to allow between frames
const unsigned int DELTA_SQ_THRESH = MAX_DIST_FRAME_TO_FRAME*MAX_DIST_FRAME_TO_FRAME;
const unsigned int CONTOUR_AREA_THRESH = 200; // minimum contour area to not ignore as noise

// comparison function (descending) for contour sizes
bool rvs_cmp_contour_area(Mat A, Mat B) {
	return contourArea(A) > contourArea(B);
}

int main(int argc, char** argv)
{
	// variable initialization
	int keyInput = 0;
	int nFrames = 0, nSmoothFrames = 0, nFailedFrames = 0, nBlindFrames = 0;
	int lastDx = 0, lastDy = 0;
	
	bool bOverlay = true;			// plot overlay?
	bool bTrace = true & bOverlay;	// plot 'bubble' trace? (only when overlay active)
	
	Ptr<BackgroundSubtractor> pMOG2;

	VideoCapture capture;		// input video capture
	VideoWriter outputVideo;	// output video writer

	Mat curFrame,		// current original frame
		fgMaskMOG2,		// foreground mask from MOG2 algorithm
		bgImg,			// container for background image from MOG2
		grayFrame,		// grayscale conversion of original frame
		frameDil,		// dilated grayscale frame
		canny_out;		// output of Canny algorithm for shape outline detection

	Mat *pOutMat = &curFrame;	// pointer to image that will be rendered once per input video frame
	Mat strucElem = getStructuringElement(MORPH_RECT, Size(3, 3)); // dilatation base element

	// containers for output of findContours()
	vector<Mat> contours;
	vector<Vec4i> hierarchy;
	
	// read video input filename from command line and construct output filename
	if (argc < 2) {
		cerr << "Please provide input video filename." << endl;
		return EXIT_FAILURE;
	}
	string filename(argv[1]);
	string outName = filename.substr(0, filename.length() - 4) + "_out.avi";

	Rect lastKnownRect, lastRect;
	Point lastKnownPos, lastPos, estimatePos, plotPos;
	list<Point> lastKnownPositions;

	// init 'live' video output window
	namedWindow("Motion tracking");

	// try to open input file
	capture.open(filename);
	if (!capture.isOpened()) {
		cerr << "Unable to open file '" << filename << "'." << endl;
		return EXIT_FAILURE;
	} else	{
		cout << "Successfully opened file '" << filename << "'." << endl;
	}

	// try to write to output file
	Size vidS = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	outputVideo.open(outName, CV_FOURCC('P','I','M','1'), capture.get(CV_CAP_PROP_FPS), vidS, true);
	if (!outputVideo.isOpened()) {
		cerr << "Unable to write to output video." << endl;
		return EXIT_FAILURE;
	}

	// build frame buffer and background subtractor
	pMOG2 = createBackgroundSubtractorMOG2(500, 30., true);
	
	// main loop over frames
	while (capture.read(curFrame) && (char)keyInput != 'q')
	{
		++nFrames;
		
		cvtColor(curFrame, grayFrame, CV_BGR2GRAY);	// convert to grayscale
		threshold(grayFrame, grayFrame, 128., 0., CV_THRESH_TRUNC); // try to mitigate (white) reflections by truncating the current frame
		GaussianBlur(grayFrame, grayFrame, Size(7, 7), 0, 0);

		pMOG2->apply(grayFrame, fgMaskMOG2);
		
		// erode and dilate to remove some noise
		erode(fgMaskMOG2, frameDil, strucElem);
		dilate(frameDil, frameDil, strucElem);

		// dilate and erode to remove holes from foreground
		dilate(frameDil, frameDil, strucElem);
		erode(frameDil, frameDil, strucElem);

		// canny to find foreground outlines
		Canny(frameDil, canny_out, 100, 200, 3);

		// find contours, sort by contour size (descending)
		findContours(canny_out, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0)); // find contours
		sort(contours.begin(), contours.end(), rvs_cmp_contour_area); // sort by contour area, beginning with the largest

		// determine largest "moving" object
		int iMaxSize = 0;
		bool bFoundCloseContour = false;
		for (unsigned int i = 0; i < contours.size(); i++)
		{
			if (contourArea(contours[i]) < CONTOUR_AREA_THRESH) // ignore contours which are too small (noise)
				break;

			// ignore contours which are too far away from the last frame
			Rect boun = boundingRect(contours[i]); // bounding rect
			Point bounCenter = (boun.tl() + boun.br())/2;

			if (i == 0) // preemptively save largest contour to get back to if no "close" contour is found.
			{
				lastRect = boun;
				lastPos = bounCenter;
			}

			// distance validity check, but only if we recently had track of the object
			if (nFrames > 1 && nFailedFrames < 10)
			{
				int dx = bounCenter.x - lastPos.x;
				int dy = bounCenter.y - lastPos.y;
				int dist2 = dx*dx + dy*dy;
				//cout << bounCenter << " " << lastPos << endl;
				if (dist2 > DELTA_SQ_THRESH) // too far away... try next contour
					continue;
			}

			lastRect = boun;
			lastPos = bounCenter;
			bFoundCloseContour = true;
			++nSmoothFrames;
			break;
		}

		if (contours.size() == 0) {
			// we don't see anything.
			++nBlindFrames;
		} else { nBlindFrames = 0; }

		// update last known position if smooth transition occured
		if (bFoundCloseContour) {
			nFailedFrames = 0;
			lastDx = lastPos.x - lastKnownPos.x;
			lastDy = lastPos.y - lastKnownPos.y;

			lastKnownRect = lastRect;
			lastKnownPos = lastPos;

			plotPos = lastKnownPos;

			if (bTrace) { // draw trace
				if (lastKnownPositions.size() > LAST_POS_BUFFER_SIZE)
					lastKnownPositions.pop_front();
				lastKnownPositions.push_back(lastPos);
				
				list<Point>::iterator it;
				int i = 0;
				for (it = lastKnownPositions.begin(); it != lastKnownPositions.end(); it++)	{
					Scalar color(180, 90, 30);
					circle(*pOutMat, *it, 5, color, 2 * i);
					++i;
				}
			}
		} else {
			++nFailedFrames;
			// guess based on velocity extrapolation
			estimatePos.x = lastKnownPos.x + nFailedFrames*lastDx;
			estimatePos.y = lastKnownPos.y + nFailedFrames*lastDy;

			if (estimatePos.x < 0 || estimatePos.y < 0 || estimatePos.x >= capture.get(CV_CAP_PROP_FRAME_WIDTH) ||
				estimatePos.y >= capture.get(CV_CAP_PROP_FRAME_HEIGHT || nFailedFrames >= 10)) {
				// we've totally lost track, cancel velocity extrapolation guess
				plotPos = lastKnownPos;
				nFailedFrames = 0;
			} else {
				plotPos = estimatePos;
			}
		}

		// draw overlay (rect frame, mid point and text)
		if (bOverlay) {
			if (nBlindFrames < 6 && bFoundCloseContour) {
				circle(*pOutMat, plotPos, 5, Scalar(255, 120, 0), 10, 8);
				rectangle(*pOutMat, lastKnownRect, Scalar(0, 255, 0), 3);
			}

			vector<ostringstream> text(4);
			const int lineSkip = 16;
			text[0] << "Frame: " << nFrames; // frame counter
			text[1] << "Object X: " << lastKnownPos.x; // moving object coordinates
			text[2] << "Object Y: " << lastKnownPos.y;
			text[3] << "Smooth rate: " << setprecision(3) << 100.0*nSmoothFrames / nFrames << "%"; // tracking percentage

			for (unsigned int line = 0; line < text.size(); line++) {
				putText(*pOutMat, text[line].str(), Point(10, 22 + line*lineSkip), CV_FONT_HERSHEY_PLAIN, 1., Scalar(180., 0., 0.));
			}
		}
		
		// cleanup temporary vectors (VS2013 stability issues)
		contours.clear();
		hierarchy.clear();

		outputVideo << *pOutMat; // add output video frame
		imshow("Motion tracking", *pOutMat); // draw frame
		keyInput = waitKey(5); // allow time for event loop
	}

	// release files
	outputVideo.release(); 
	capture.release();

	return EXIT_SUCCESS;
}