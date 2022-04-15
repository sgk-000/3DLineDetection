#include <stdio.h>
#include <fstream>

#include "LineDetection3D.h"
#include "nanoflann.hpp"
#include "utils.h"
#include "Timer.h"
#include<iostream> 
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>

using namespace cv;
using namespace std;
using namespace nanoflann;

void readDataFromPCDFile( std::string filepath, PointCloud<float> &cloud )
{
	// cloud.pts.reserve(1);
	cout<<"Reading data ..."<<endl;

	// 1. read in point data
	std::ifstream ptReader( filepath );
	std::vector<cv::Point3d> lidarPoints;
	float x = 0, y = 0, z = 0, color = 0;
	float nx, ny, nz;
	int a = 0, b = 0, c = 0; 
	int labelIdx = 0;
	int count = 0;
	int countTotal = 0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filepath, *pcl_cloud) == -1)
	{
		PCL_ERROR("Couldn't read pcd file \n");
	}
	std::cout << "Loaded " << pcl_cloud->width * pcl_cloud->height << " data points from pcd file" << std::endl;
	for (const auto &point : *pcl_cloud)
	{
		// std::cout << "    " << point.x << " " << point.y << " " << point.z << std::endl;
		cloud.pts.push_back(PointCloud<float>::PtData(point.x, point.y, point.z));
	}
	std::cout << "Total num of points: " << cloud.pts.size() << "\n";
}

void writeOutPlanes( string filePath, std::vector<PLANE> &planes, double scale )
{
	// write out bounding polygon result
	string fileEdgePoints = filePath + "planes.txt";
	FILE *fp2 = fopen( fileEdgePoints.c_str(), "w");
	for (int p=0; p<planes.size(); ++p)
	{
		int R = rand()%255;
		int G = rand()%255;
		int B = rand()%255;

		for (int i=0; i<planes[p].lines3d.size(); ++i)
		{
			for (int j=0; j<planes[p].lines3d[i].size(); ++j)
			{
				cv::Point3d dev = planes[p].lines3d[i][j][1] - planes[p].lines3d[i][j][0];
				double L = sqrt(dev.x*dev.x + dev.y*dev.y + dev.z*dev.z);
				int k = L/(scale/10);

				double x = planes[p].lines3d[i][j][0].x, y = planes[p].lines3d[i][j][0].y, z = planes[p].lines3d[i][j][0].z;
				double dx = dev.x/k, dy = dev.y/k, dz = dev.z/k;
				for ( int j=0; j<k; ++j)
				{
					x += dx;
					y += dy;
					z += dz;

					fprintf( fp2, "%.6lf   %.6lf   %.6lf    ", x, y, z );
					fprintf( fp2, "%d   %d   %d   %d\n", R, G, B, p );
				}
			}
		}
	}
	fclose( fp2 );
}

void writeOutLines( string filePath, std::vector<std::vector<cv::Point3d> > &lines, double scale )
{
	// write out bounding polygon result
	string fileEdgePoints = filePath + "lines.txt";
	FILE *fp2 = fopen( fileEdgePoints.c_str(), "w");
	for (int p=0; p<lines.size(); ++p)
	{
		int R = rand()%255;
		int G = rand()%255;
		int B = rand()%255;

		cv::Point3d dev = lines[p][1] - lines[p][0];
		double L = sqrt(dev.x*dev.x + dev.y*dev.y + dev.z*dev.z);
		int k = L/(scale/10);

		double x = lines[p][0].x, y = lines[p][0].y, z = lines[p][0].z;
		double dx = dev.x/k, dy = dev.y/k, dz = dev.z/k;
		for ( int j=0; j<k; ++j)
		{
			x += dx;
			y += dy;
			z += dz;

			fprintf( fp2, "%.6lf   %.6lf   %.6lf    ", x, y, z );
			fprintf( fp2, "%d   %d   %d   %d\n", R, G, B, p );
		}
	}
	fclose( fp2 );
}
pcl::PointCloud<pcl::PointXYZ>::Ptr getLineCloud(PointCloud<float> &in_cloud, const std::vector<std::vector<cv::Point3d>> &lines, const double scale, int point_num)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	out_cloud->reserve(point_num);
	// write out bounding polygon result
	for (int p = 0; p < lines.size(); ++p)
	{
		int R = rand() % 255;
		int G = rand() % 255;
		int B = rand() % 255;

		cv::Point3d dev = lines[p][1] - lines[p][0];
		double L = sqrt(dev.x * dev.x + dev.y * dev.y + dev.z * dev.z);
		int k = L / (scale / 10);

		double x = lines[p][0].x, y = lines[p][0].y, z = lines[p][0].z;
		double dx = dev.x / k, dy = dev.y / k, dz = dev.z / k;
		for (int j = 0; j < k; ++j)
		{
			x += dx;
			y += dy;
			z += dz;

			pcl::PointXYZ point;
			point.x = x;
			point.y = y;
			point.z = z;
			out_cloud->push_back(point);
		}
	}
	return out_cloud;
}

void writeOutPCDFile(const string file_path, PointCloud<float> &in_cloud)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl_cloud->reserve(in_cloud.pts.size());
	for(const auto p : in_cloud.pts)
	{
		pcl::PointXYZ point;
		point.x = p.x;
		point.y = p.y;
		point.z = p.z;
		pcl_cloud->push_back(point);
	}
	pcl::io::savePCDFileBinary(file_path, *pcl_cloud);
}

	int
	main()
{
	string fileData = "/home/digital/sgk/data/campus/pointcloud_map.pcd";
	string fileOut = "/home/digital/sgk/data/campus//line_map.pcd";

	// read in data
	PointCloud<float> pointData; 
	readDataFromPCDFile( fileData, pointData );

	int k = 20;
	LineDetection3D detector;
	std::vector<PLANE> planes;
	std::vector<std::vector<cv::Point3d> > lines;
	std::vector<double> ts;
	detector.run( pointData, k, planes, lines, ts );
	cout<<"lines number: "<<lines.size()<<endl;
	cout<<"planes number: "<<planes.size()<<endl;

	PointCloud<float> line_pointcloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr line_pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	line_pcl_cloud = getLineCloud(line_pointcloud, lines, detector.scale, detector.pointNum);
	pcl::io::savePCDFileBinary(fileOut, *line_pcl_cloud);
	// writeOutPlanes( fileOut, planes, detector.scale );
	// writeOutLines( fileOut, lines, detector.scale );
	// writeOutPCDFile(fileOut, *line_pcl_cloud);
	return 0;
}