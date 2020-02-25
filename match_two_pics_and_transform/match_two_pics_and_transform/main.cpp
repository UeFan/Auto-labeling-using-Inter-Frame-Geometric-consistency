//
//  main.cpp
//  match_two_pics_and_transform
//
//  Created by fanyue on 2019/2/24.
//  Copyright © 2019 fanyue. All rights reserved.
//
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <math.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>

#include <dirent.h>
#include <unistd.h>

using namespace std;
using namespace cv;

Point last_roi_center = Point(0,0);
queue<Point> last_roi_center_queue;
Rect2d roi_0,roi_1;
Ptr<TrackerCSRT> tracker = TrackerCSRT::create();
vector<Point2f> Pts1;
vector<Point2f> Pts2;

Mat frame_1_trans, frame_0, frame_1;
double angle_pre, angle_prepre;
bool filter_the_points = 0;

const int g_nMaxAlphaValue_1 = 100;//alpha值得最大值
int g_nAlphaValueSlider_1;//滑动条对应的变量
double g_dAlphaValue_1;
double g_dBetaValue_1;
Mat g_dstImage_1;

void on_Trackbar_1(int, void*)
{
  //求出当前最大值相对于最大值比例
  g_dAlphaValue_1 = (double)g_nAlphaValueSlider_1 / g_nMaxAlphaValue_1;
  g_dBetaValue_1 = 1.0 - g_dAlphaValue_1;
  
  //根据alpha与beta的值进行现行混合
  addWeighted(frame_0,g_dAlphaValue_1,frame_1_trans,g_dBetaValue_1,0.0,g_dstImage_1);
  imshow("frame_1_trans", g_dstImage_1);
}

double getDistance (Point pointO,Point pointA ) //计算两点之间距离
{
  double distance;
  distance = powf((pointO.x - pointA.x),2) + powf((pointO.y - pointA.y),2);
  
  distance = sqrtf(distance);
  return distance;
}

void detect_and_match(Mat& img_1, Mat& img_2)
{
  //Create SIFT class pointer
  Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
  //SiftFeatureDetector siftDetector;
  //Loading images
  
  if (!img_1.data || !img_2.data)
  {
    cout << "Reading picture error！" << endl;
    return;
  }
  //Detect the keypoints
  double t0 = getTickCount();//当前滴答数
  vector<KeyPoint> keypoints_1, keypoints_2;
  f2d->detect(img_1, keypoints_1);
  f2d->detect(img_2, keypoints_2);
  cout << "The keypoints number of img1 is:" << keypoints_1.size() << endl;
  cout << "The keypoints number of img2 is:" << keypoints_2.size() << endl;
  //Calculate descriptors (feature vectors)
  Mat descriptors_1, descriptors_2;
  f2d->compute(img_1, keypoints_1, descriptors_1);
  f2d->compute(img_2, keypoints_2, descriptors_2);
  double freq = getTickFrequency();
  double tt = ((double)getTickCount() - t0) / freq;
  cout << "Extract SIFT Time:" <<tt<<"ms"<< endl;
  //画关键点
  Mat img_keypoints_1, img_keypoints_2;
  drawKeypoints(img_1,keypoints_1,img_keypoints_1,Scalar::all(-1),0);
  drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), 0);
  
  //Matching descriptor vector using BFMatcher
  BFMatcher matcher;
  vector<DMatch> matches;
  matcher.match(descriptors_1, descriptors_2, matches);
  cout << "The number of match:" << matches.size()<<endl;
  //绘制匹配出的关键点
  Mat img_matches;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
//    imshow("Match image",img_matches);
  //计算匹配结果中距离最大和距离最小值
  double min_dist = matches[0].distance, max_dist = matches[0].distance;
  for (int m = 0; m < matches.size(); m++)
  {
    if (matches[m].distance<min_dist)
    {
      min_dist = matches[m].distance;
    }
    if (matches[m].distance>max_dist)
    {
      max_dist = matches[m].distance;
    }
  }
  cout << "min dist=" << min_dist << endl;
  cout << "max dist=" << max_dist << endl;
  //筛选出较好的匹配点
  vector<DMatch> goodMatches;
  for (int m = 0; m < matches.size(); m++)
  {
    if (matches[m].distance < 0.6*max_dist)
    {
      goodMatches.push_back(matches[m]);
    }
  }
  cout << "The number of good matches:" <<goodMatches.size()<< endl;
  //画出匹配结果
  Mat img_out;
  //红色连接的是匹配的特征点数，绿色连接的是未匹配的特征点数
  //matchColor – Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1) , the color is generated randomly.
  //singlePointColor – Color of single keypoints(circles), which means that keypoints do not have the matches.If singlePointColor == Scalar::all(-1), the color is generated randomly.
  //CV_RGB(0, 255, 0)存储顺序为R-G-B,表示绿色
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, goodMatches, img_out, Scalar::all(-1), CV_RGB(0, 0, 255), Mat(), 2);
//    imshow("good Matches",img_out);
  //RANSAC匹配过程
  vector<DMatch> m_Matches;
  m_Matches = goodMatches;
  int ptCount = goodMatches.size();
  if (ptCount < 100)
  {
    cout << "Don't find enough match points" << endl;
    return;
  }
  
  //坐标转换为float类型
  vector <KeyPoint> RAN_KP1, RAN_KP2;
  //size_t是标准C库中定义的，应为unsigned int，在64位系统中为long unsigned int,在C++中为了适应不同的平台，增加可移植性。
  for (size_t i = 0; i < m_Matches.size(); i++)
  {
    RAN_KP1.push_back(keypoints_1[goodMatches[i].queryIdx]);
    RAN_KP2.push_back(keypoints_2[goodMatches[i].trainIdx]);
    //RAN_KP1是要存储img01中能与img02匹配的点
    //goodMatches存储了这些匹配点对的img01和img02的索引值
  }
  //坐标变换
  vector <Point2f> p01, p02;
  for (size_t i = 0; i < m_Matches.size(); i++)
  {
    p01.push_back(RAN_KP1[i].pt);
    p02.push_back(RAN_KP2[i].pt);
  }
  
  
  //////////////////////////////////////////////////////////////////////////
  //重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
  //求基础矩阵 Fundamental,3*3的基础矩阵
  //////////////////////////////////////////////////////////////////////////
  vector<uchar> RansacStatus;
  Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);
  
  vector <KeyPoint> RR_KP1, RR_KP2;
  vector <DMatch> RR_matches;
  Pts1.clear();
  Pts2.clear();
  int index = 0;
  for (size_t i = 0; i < m_Matches.size(); i++)
  {
    if (RansacStatus[i] != 0)
    {
      
      if(index%1 == 0)
      {
        if(filter_the_points)//对算出的匹配特征点对进行筛选
        {
          if(getDistance(RAN_KP1[i].pt, (roi_0.tl()+roi_0.br())/2)>200)
          {
            Pts1.push_back(RAN_KP1[i].pt);
            Pts2.push_back(RAN_KP2[i].pt);
            
            RR_KP1.push_back(RAN_KP1[i]);
            RR_KP2.push_back(RAN_KP2[i]);
            m_Matches[i].queryIdx = index;
            m_Matches[i].trainIdx = index;
            RR_matches.push_back(m_Matches[i]);
            //由前面两行可以看出，m_Matches里保存的是n对匹配的特征点的序号，因为特征点已经排序好，所以m_Matches的queryIdx==trainIdx
            index++;
          }
        }
        
        if(!filter_the_points)//不筛选
        {
          Pts1.push_back(RAN_KP1[i].pt);
          Pts2.push_back(RAN_KP2[i].pt);
          
          RR_KP1.push_back(RAN_KP1[i]);
          RR_KP2.push_back(RAN_KP2[i]);
          m_Matches[i].queryIdx = index;
          m_Matches[i].trainIdx = index;
          RR_matches.push_back(m_Matches[i]);
          //由前面两行可以看出，m_Matches里保存的是n对匹配的特征点的序号，因为特征点已经排序好，所以m_Matches的queryIdx==trainIdx
          index++;
        }
      }
    }
  }
  cout << "RANSAC后匹配点数" <<RR_matches.size()<< endl;
  Mat img_RR_matches;
  drawMatches(img_1, RR_KP1, img_2, RR_KP2, RR_matches, img_RR_matches);
  imshow("After RANSAC",img_RR_matches);
  
  //  matches.~ vector <DMatch>();
  //  goodMatches.~ vector <DMatch>();
  //  RansacStatus.~ vector <uchar>();
  //  RR_matches.~ vector <DMatch>();
  //  m_Matches.~ vector <DMatch>();
}
string int2str(int i)
{
  char ss[10];
  sprintf(ss,"%06d",i);
  return ss;
}

double matchPics()
{
  Mat m_homography;
  vector<uchar> m;
  vector<Point2f> scene_corners(4);
  vector<Point2f> roi_center(1), roi_center_trans(1);
  
  //  point[0] = Point(0,0);
  //  point[1] = Point(frame_roi.size().height,0);
  //  point[2] = Point(0,frame_roi.size().width);
  //  point[3] = Point(frame_roi.size().height,frame_roi.size().width);
  //
  //
  //
  //  detect_and_match(frame_0, frame_roi);
  ////  这里使用findHomography函数，这个函数的返回值才是真正的变换矩阵
  //  m_homography=findHomography(Pts2,Pts1,CV_RANSAC,3,m);
  //  perspectiveTransform(point, scene_corners, m_homography);
  //  for(size_t i = 0; i< 4; i++)
  //  {
  //    circle(frame_0, scene_corners[i], 5, Scalar(255, 0, 0), -1);
  //    cout<<point[i]<<scene_corners[i]<<endl;
  //  }
  //  waitKey(0);
  //
  
  detect_and_match(frame_0, frame_1);
  //这里使用findHomography函数，这个函数的返回值才是真正的变换矩阵
  m_homography=findHomography(Pts2,Pts1,CV_RANSAC,3,m);
  
  //  waitKey(0);
  
  
  
  vector<Point2f> points_ori, points_trans;
  for(int i=0;i<frame_1.size().height;i++){
    for(int j=0;j<frame_1.size().width;j++){
      /*与一般习惯中的（行，列）相反, 之所以这样，是因为下面perspectiveTransform函数是这么要求的*/
      points_ori.push_back(Point2f(j,i));
    }
  }
  
  perspectiveTransform(points_ori, points_trans, m_homography);
  
  
  
  frame_1_trans = Mat::zeros(frame_0.size(),frame_0.type());
  int count_ = 0;
  
  for(int i=0;i<frame_1.size().height;i++){
    for(int j=0;j<frame_1.size().width;j++){
      int y = points_trans[count_].y;
      int x = points_trans[count_].x;
      if(y < frame_0.size().height && y>0 && x < frame_0.size().width && x>0)
      {
        frame_1_trans.at<cv::Vec3b>(y,x) = frame_1.at<cv::Vec3b>(i,j);
      }
      count_++;
    }
    
  }
//  rectangle(frame_0, roi_1, Scalar(255, 0, 0), 2, 1);
  
  tracker->update(frame_1, roi_1);
  roi_center[0] = (roi_1.tl()+roi_1.br())/2;
  last_roi_center_queue.push(roi_center[0]);
  
  perspectiveTransform(roi_center, roi_center_trans, m_homography);
//  rectangle(frame_1, roi_1, Scalar(255, 0, 0), 2, 1);
//  imshow("frame_1", frame_1);
  if(getDistance(last_roi_center, roi_center_trans[0])<20)
    return -2;
  
  double angle = atan((double)(roi_center_trans[0].x - last_roi_center.x)/(roi_center_trans[0].y - last_roi_center.y))/3.14*2;
  line(frame_0, last_roi_center, roi_center_trans[0], Scalar(255, 0, 255), 3);
//  line(frame_0, Point(frame_0.size().width-960, frame_0.size().height), Point(frame_0.size().width-960 - tan(angle*3.14/2)*200, frame_0.size().height - 200), Scalar(255, 255, 0), 3);
////////////// To smooth it /////////
  if(angle_prepre != 0)
  {
    angle = (angle_prepre + angle_pre + angle)/3;
  }
  angle_prepre = angle_pre;
  angle_pre = angle;
////////////////////////////////////
  
  
  //+ to_string(last_roi_center.x) + "," + to_string(last_roi_center.y) + "\n" + to_string((int)roi_center_trans[0].x) + "," + to_string((int)roi_center_trans[0].y);
  
  last_roi_center = last_roi_center_queue.front();
  last_roi_center_queue.pop();
  //cout<<Point(frame_0.size().height-500,  frame_0.size().width-500)<<endl;
  cout<<last_roi_center<<endl;
  
  g_nAlphaValueSlider_1 = 50;
  namedWindow("frame_1_trans", CV_WINDOW_NORMAL);
  createTrackbar("TrackBarName_1","frame_1_trans",&g_nAlphaValueSlider_1,g_nMaxAlphaValue_1,on_Trackbar_1);
  on_Trackbar_1(g_nAlphaValueSlider_1, 0);
  
//  imshow("frame_0", frame_0);
//  imshow("frame_1", frame_1);
  waitKey(100);
//  waitKey(0);
  return angle;
}

int fileRemove(string fname)
{
  return remove(fname.c_str());
}

vector<string> getFiles(string cate_dir)
{
  vector<string> files;//存放文件名
  DIR *dir;
  struct dirent *ptr;
  char base[1000];
  
  if ((dir=opendir(cate_dir.c_str())) == NULL)
  {
    perror("Open dir error...");
    exit(1);
  }
  
  while ((ptr=readdir(dir)) != NULL)
  {
    if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
      continue;
    else if (ptr->d_name[0]=='.')
      continue;
    else if(ptr->d_type == 8)    ///file
      //printf("d_name:%s/%s\n",basePath,ptr->d_name);
      files.push_back(ptr->d_name);
    else if(ptr->d_type == 10)    ///link file
      //printf("d_name:%s/%s\n",basePath,ptr->d_name);
      continue;
    else if(ptr->d_type == 4)    ///dir
    {
      files.push_back(ptr->d_name);
      /*
       memset(base,'\0',sizeof(base));
       strcpy(base,basePath);
       strcat(base,"/");
       strcat(base,ptr->d_nSame);
       readFileList(base);
       */
    }
  }
  closedir(dir);
  //排序，按从小到大排序
  sort(files.begin(), files.end());
  return files;
}

void loop_in_folders(string data_path, string floder_name)
{
  int start_num = 0, step = 1;
  string label_path = data_path + floder_name;
  string pics_path = data_path + floder_name + "/images/";
  cout<<"Star to loop in folder. Enter the start_num"<<endl;
  cin>>start_num;
  
  string old_labels = "";
  string old_trans_labels = "";
  //annotate the middle of the group
  if(start_num>0)
  {
    //read in old annotation.
    ifstream OpenFile(label_path + "/direction_n_filted" + ".txt");
    ifstream OpenTransFile(label_path + "/translation" + ".txt");

    for(int i = 0;i<start_num;i++)
    {
      char angle[20];
      char trans[20];
      OpenFile.getline(angle,20);
      OpenTransFile.getline(trans,20);
      old_trans_labels += (std::string(trans) + '\n');
      old_labels += (std::string(angle) + '\n');
    }
  }
  
  vector<string> pics = getFiles(pics_path);
  
  frame_0 = imread(pics_path + pics[0+start_num], CV_LOAD_IMAGE_COLOR);
  
  
  roi_0 = selectROI("tracker", frame_0);
  
  waitKey(0);
  //quit if ROI was not selected
  if (roi_0.width <5 || roi_0.height <5)
  {
    cout<<"ROI was not selected"<<endl;
    return ;
  }
  
  ofstream OpenFile(label_path + "/direction_n_filted" + ".txt");
  ofstream OpenTransFile(label_path + "/translation" + ".txt");
  //ofstream OpenFile_output_flip(label_path + "../" + floder_name + "-flip/" + "direction" + ".txt");
  OpenFile<<old_labels;
  OpenTransFile<<old_trans_labels;
  
  double ratio;
  // initialize the tracker
  tracker->init(frame_0, roi_0);
  last_roi_center = (roi_0.tl()+roi_0.br())/2;
  for(int i = start_num + 1, j = start_num + 1; i<pics.size(); i++)
  {
    cout<<start_num+i<<",  ";
    frame_1 = imread(pics_path + pics[i], CV_LOAD_IMAGE_COLOR);
    
    if(getDistance(last_roi_center, Point(frame_0.size().height, frame_0.size().width/2))>1100)
    {
      cout<<"current pic:"+pics[i-1]<<endl;
      OpenFile.close();
      OpenTransFile.close();
//      waitKey(0);
      return;
    }
    
    double angle = matchPics();
    if(angle == -2.) //if the pilot's movement is too little.
    {
      cout<<"jump"<<endl;
      // fileRemove(pics_path + pics[i]);
    }
    else
    {
      string str_ = to_string(angle) + "\n";
      OpenFile<<str_;
      cout<<"last_roi_c"<<last_roi_center<<endl;
      double trans_ = tan(-angle*3.14/2)*(roi_0.height) + last_roi_center.x;
//      circle(frame_0, Point((int)trans_, (last_roi_center.y - roi_0.height)), 8, Scalar(0, 255, 0));
      ratio = (last_roi_center.y - roi_0.height)/frame_0.size().height;
      Mat frame_roi = frame_0(Rect((1-ratio)*frame_0.size().width/2, 0, ratio*frame_0.size().width, ratio*frame_0.size().height));
      imshow("frame_roi", frame_roi);
      resize(frame_roi, frame_roi, Size(frame_0.size().width,frame_0.size().height));
      imwrite(pics_path + "../" + pics[j-1], frame_roi);
      OpenTransFile<< (to_string(trans_/frame_0.size().width) + "\n");
//      imshow("frame_0", frame_0);
      cout<<"translation_labels:"<<trans_<<endl;
      cout<<"ditection_labels:"<<angle<<endl;
      // !!! Comment to freeze to frame_0
      frame_0 = imread(pics_path + pics[j], CV_LOAD_IMAGE_COLOR);
      j++;
    }
    
//    imshow("frame_1", frame_1);
    waitKey(50);
//    waitKey(0);
  }
  OpenFile.close();
}

void test_with_two_pics(string pic1_path, string pic2_path)
{
  frame_0 = imread(pic1_path, CV_LOAD_IMAGE_COLOR);
  frame_1 = imread(pic2_path, CV_LOAD_IMAGE_COLOR);

  roi_0 = selectROI("tracker", frame_0);
  
  waitKey(0);
  //quit if ROI was not selected
  if (roi_0.width <2 && roi_0.height <2)
  {
    cout<<"ROI was not selected"<<endl;
      cout<<"roi_0.width, roi_0.height ="<<roi_0.width<<" ,"<<roi_0.height<<endl;
    return ;
  }
  

  // initialize the tracker
  tracker->init(frame_0, roi_0);
  last_roi_center = (roi_0.tl()+roi_0.br())/2;

  double angle = matchPics();
  if(angle == -2)
  {
    cout<<"jump"<<endl;
  }
  else
  {
    cout<<angle<<endl;
    // !!! Comment to freeze to frame_0      //frame_1.copyTo(frame_0);
  }

}

int main(int argc, const char * argv[]) {
  
  cout<<"do u want to start a test?"<<endl;
  string str;
  cin>>str;
  if(str == "y")
  {
    test_with_two_pics("/Users/fanyue/Downloads/帧-000014.jpg", "/Users/fanyue/Downloads/帧-000016.jpg");
    waitKey();
  }
  
  int skip_to = 0;
  string data_path = "/Users/fanyue/Downloads/ue-train/";
  vector<string> folders=getFiles(data_path);
  for(int i = 0; i<folders.size(); i++)
  {
    if(folders[i][0] == 'D' && folders[i][folders[i].length()-1] != 'p')
    {
      cout<<"i = "<<i<<"    "<<folders[i]<<endl;
      cin>>skip_to;
      if(skip_to>=0 && skip_to<=50)
      {
        i = skip_to;
        continue;
      }
      angle_pre = 0;
      angle_prepre = 0;
      loop_in_folders(data_path, folders[i]);
    }
  }
  
  return 0;
}
