#include <opencv2\opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
	VideoCapture capture;
	capture.open("car.wmv"); //파일을 읽음
	Mat bckImage = imread("carBkg.jpg", 0); //평균영상
	Mat accImage = bckImage; //누적영상
	CV_Assert(capture.isOpened() || accImage.data); //예외처리

	double frame_rate = capture.get(CAP_PROP_FPS); 
	int delay = 1000 / frame_rate;

	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);

	Mat diffImage(height, width, CV_32F); //차영상
	Mat grayImage(height, width, CV_8U); //그레이 이미지
	Mat maskImage(height, width, CV_8U); //마스크

	double alpha = 0.1;
	int nFrameCount = 0;
	Mat frame;
	maskImage.setTo(0); //0으로 초기화
	Matx <uchar, 3, 3> mask; //Matx객체를 이용한
	mask << 0, 1, 0, 1, 1, 1, 0, 1, 0; //초기화

	char a[100] = { "A" };
	String b[100];
	
	Size sz(bckImage.cols, bckImage.rows);
	Mat morpimg;
	Mat labels, stats, centroids;
	Mat dst, tmp;
	Mat dst2(sz, CV_8UC3, Scalar(0, 0, 0));
	
	while (capture.read(frame)) {

		printf("nFrameCount=%d\n", nFrameCount);
		cvtColor(frame, grayImage, COLOR_BGR2GRAY); //그레이스케일로 바꿈
		grayImage.convertTo(grayImage, CV_32F); //float형으로
		accImage.convertTo(accImage, CV_32F); //float형으로
		absdiff(grayImage, accImage, diffImage); //차영상에 대해 절댓값 적용
		threshold(diffImage, maskImage, 50, 255, THRESH_BINARY_INV); //이진화

		maskImage.convertTo(maskImage, CV_8U);
		accumulateWeighted(grayImage, accImage, alpha, maskImage); //마스크 영상을 적용하여 평균영상을 구함
		grayImage.convertTo(grayImage, CV_8U);
		accImage.convertTo(accImage, CV_8U);
		maskImage = 255 - maskImage; //흰색->검은색, 검은색->흰색

		morphologyEx(maskImage, morpimg, MORPH_OPEN, mask, Point(-1, -1), 3); //열림 함수
		int cnt = connectedComponentsWithStats(morpimg, labels, stats, centroids); //레이블의 수
		dst = frame;

		sortIdx(stats, tmp, SORT_EVERY_COLUMN | SORT_DESCENDING); //내림차순으로 정렬

		for (int i = 1; i < cnt; i++) {
			int* p = stats.ptr<int>(i);  //통계정보
			double* q = centroids.ptr<double>(i); //무게중심 
			if (p[4] > 20) {
				b[i - 1] = char(a[0] + i - 1); //알파벳 증가
				putText(dst, b[i - 1], Point(q[0], q[1]), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255)); //글자
			}

			for (int k = 1; k < cnt; k++) {
				if (i == tmp.at<int>(k, 4)) { //가장 큰 움직임
					for (int n = 0; n < labels.rows; n++) {
						for (int j = 0; j < labels.cols; j++) {
							if (labels.at<int>(n, j) == k) {
								dst2.at<Vec3b>(n, j) = Vec3b(0, 0, 255); //빨간색 칠함
							}
						}
					}
					putText(dst2, "A", Point(q[0], q[1]), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255)); //글자
				}
			}

			imshow("bckImage", bckImage);
			imshow("grayImage", grayImage);
			imshow("accImage", accImage);
			imshow("maskImage", maskImage);
			imshow("labelingImage", dst);
			imshow("tracing1", dst2);

			char chKey = waitKey(5);
			if (chKey == 27) break;

			nFrameCount++;
		}
	}
}