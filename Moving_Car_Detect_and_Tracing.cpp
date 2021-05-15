#include <opencv2\opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
	VideoCapture capture;
	capture.open("car.wmv"); //������ ����
	Mat bckImage = imread("carBkg.jpg", 0); //��տ���
	Mat accImage = bckImage; //��������
	CV_Assert(capture.isOpened() || accImage.data); //����ó��

	double frame_rate = capture.get(CAP_PROP_FPS); 
	int delay = 1000 / frame_rate;

	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);

	Mat diffImage(height, width, CV_32F); //������
	Mat grayImage(height, width, CV_8U); //�׷��� �̹���
	Mat maskImage(height, width, CV_8U); //����ũ

	double alpha = 0.1;
	int nFrameCount = 0;
	Mat frame;
	maskImage.setTo(0); //0���� �ʱ�ȭ
	Matx <uchar, 3, 3> mask; //Matx��ü�� �̿���
	mask << 0, 1, 0, 1, 1, 1, 0, 1, 0; //�ʱ�ȭ

	char a[100] = { "A" };
	String b[100];
	
	Size sz(bckImage.cols, bckImage.rows);
	Mat morpimg;
	Mat labels, stats, centroids;
	Mat dst, tmp;
	Mat dst2(sz, CV_8UC3, Scalar(0, 0, 0));
	
	while (capture.read(frame)) {

		printf("nFrameCount=%d\n", nFrameCount);
		cvtColor(frame, grayImage, COLOR_BGR2GRAY); //�׷��̽����Ϸ� �ٲ�
		grayImage.convertTo(grayImage, CV_32F); //float������
		accImage.convertTo(accImage, CV_32F); //float������
		absdiff(grayImage, accImage, diffImage); //������ ���� ���� ����
		threshold(diffImage, maskImage, 50, 255, THRESH_BINARY_INV); //����ȭ

		maskImage.convertTo(maskImage, CV_8U);
		accumulateWeighted(grayImage, accImage, alpha, maskImage); //����ũ ������ �����Ͽ� ��տ����� ����
		grayImage.convertTo(grayImage, CV_8U);
		accImage.convertTo(accImage, CV_8U);
		maskImage = 255 - maskImage; //���->������, ������->���

		morphologyEx(maskImage, morpimg, MORPH_OPEN, mask, Point(-1, -1), 3); //���� �Լ�
		int cnt = connectedComponentsWithStats(morpimg, labels, stats, centroids); //���̺��� ��
		dst = frame;

		sortIdx(stats, tmp, SORT_EVERY_COLUMN | SORT_DESCENDING); //������������ ����

		for (int i = 1; i < cnt; i++) {
			int* p = stats.ptr<int>(i);  //�������
			double* q = centroids.ptr<double>(i); //�����߽� 
			if (p[4] > 20) {
				b[i - 1] = char(a[0] + i - 1); //���ĺ� ����
				putText(dst, b[i - 1], Point(q[0], q[1]), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255)); //����
			}

			for (int k = 1; k < cnt; k++) {
				if (i == tmp.at<int>(k, 4)) { //���� ū ������
					for (int n = 0; n < labels.rows; n++) {
						for (int j = 0; j < labels.cols; j++) {
							if (labels.at<int>(n, j) == k) {
								dst2.at<Vec3b>(n, j) = Vec3b(0, 0, 255); //������ ĥ��
							}
						}
					}
					putText(dst2, "A", Point(q[0], q[1]), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255)); //����
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