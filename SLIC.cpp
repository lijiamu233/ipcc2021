// SLIC.cpp: implementation of the SLIC class.
//===========================================================================
// This code implements the zero parameter superpixel segmentation technique
// described in:
//
//
//
// "SLIC Superpixels Compared to State-of-the-art Superpixel Methods"
//
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua,
// and Sabine Susstrunk,
//
// IEEE TPAMI, Volume 34, Issue 11, Pages 2274-2282, November 2012.
//
// https://www.epfl.ch/labs/ivrl/research/slic-superpixels/
//===========================================================================
// Copyright (c) 2013 Radhakrishna Achanta.
//
// For commercial use please contact the author:
//
// Email: firstname.lastname@epfl.ch
//===========================================================================

#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <cstring>
#include <fstream>
#include "SLIC.h"
#include <chrono>
#include <omp.h>
#include <immintrin.h>
using namespace std;

typedef chrono::high_resolution_clock Clock;

// For superpixels
const int dx4[4] = {-1,  0,  1,  0};
const int dy4[4] = { 0, -1,  0,  1};
//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

// For supervoxels
const int dx10[10] = {-1,  0,  1,  0, -1,  1,  1, -1,  0, 0};
const int dy10[10] = { 0, -1,  0,  1, -1, -1,  1,  1,  0, 0};
const int dz10[10] = { 0,  0,  0,  0,  0,  0,  0,  0, -1, 1};

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SLIC::SLIC()
{
	m_labvec = NULL;
	// m_avec = NULL;
	// m_bvec = NULL;

	m_lvecvec = NULL;
	m_avecvec = NULL;
	m_bvecvec = NULL;
}

SLIC::~SLIC()
{
	if(m_labvec) delete [] m_labvec;
	// if(m_avec) delete [] m_avec;
	// if(m_bvec) delete [] m_bvec;


	if(m_lvecvec)
	{
		for( int d = 0; d < m_depth; d++ ) delete [] m_lvecvec[d];
		delete [] m_lvecvec;
	}
	if(m_avecvec)
	{
		for( int d = 0; d < m_depth; d++ ) delete [] m_avecvec[d];
		delete [] m_avecvec;
	}
	if(m_bvecvec)
	{
		for( int d = 0; d < m_depth; d++ ) delete [] m_bvecvec[d];
		delete [] m_bvecvec;
	}
}

//==============================================================================
///	RGB2XYZ
///
/// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
void SLIC::RGB2XYZ(
	const int&		sR,
	const int&		sG,
	const int&		sB,
	double&			X,
	double&			Y,
	double&			Z)
{
	double R = sR/255.0;
	double G = sG/255.0;
	double B = sB/255.0;

	double r, g, b;

	if(R <= 0.04045)	r = R/12.92;
	else				r = pow((R+0.055)/1.055,2.4);
	if(G <= 0.04045)	g = G/12.92;
	else				g = pow((G+0.055)/1.055,2.4);
	if(B <= 0.04045)	b = B/12.92;
	else				b = pow((B+0.055)/1.055,2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}

//===========================================================================
///	RGB2LAB
//===========================================================================
void SLIC::RGB2LAB(const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double X, Y, Z;

	double R = sR/255.0;
	double G = sG/255.0;
	double B = sB/255.0;

	double r, g, b;

	if(R <= 0.04045)	r = R/12.92;
	else				r = pow((R+0.055)/1.055,2.4);
	if(G <= 0.04045)	g = G/12.92;
	else				g = pow((G+0.055)/1.055,2.4);
	if(B <= 0.04045)	b = B/12.92;
	else				b = pow((B+0.055)/1.055,2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;	//actual CIE standard
	double kappa   = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X/Xr;
	double yr = Y/Yr;
	double zr = Z/Zr;

	double fx, fy, fz;
	if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
	else				fx = (kappa*xr + 16.0)/116.0;
	if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
	else				fy = (kappa*yr + 16.0)/116.0;
	if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
	else				fz = (kappa*zr + 16.0)/116.0;

	lval = 116.0*fy-16.0;
	aval = 500.0*(fx-fy);
	bval = 200.0*(fy-fz);
}

//===========================================================================
///	DoRGBtoLABConversion
///
///	For whole image: overlaoded floating point version
//===========================================================================
void SLIC::DoRGBtoLABConversion(
	const unsigned int*&		ubuff,
		lab*&					labvec)
{
	int sz = m_width*m_height;
	labvec = new lab[sz];
	double epsilon = 0.008856;	//actual CIE standard
	double kappa   = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white
	#pragma omp parallel for simd
	for( int j = 0; j < sz; j++ )
	{
		int sR = (ubuff[j] >> 16) & 0xFF;
		int sG = (ubuff[j] >>  8) & 0xFF;
		int sB = (ubuff[j]      ) & 0xFF;
		double R = sR/255.0;
		double G = sG/255.0;
		double B = sB/255.0;

		double r, g, b;

		if(R <= 0.04045)	r = R/12.92;
		else				r = pow((R+0.055)/1.055,2.4);
		if(G <= 0.04045)	g = G/12.92;
		else				g = pow((G+0.055)/1.055,2.4);
		if(B <= 0.04045)	b = B/12.92;
		else				b = pow((B+0.055)/1.055,2.4);

		double xr = (r*0.4124564 + g*0.3575761 + b*0.1804375) / Xr;
		double yr = (r*0.2126729 + g*0.7151522 + b*0.0721750) / Yr;
		double zr = (r*0.0193339 + g*0.1191920 + b*0.9503041) / Zr;

		//------------------------
		// XYZ to LAB conversion
		//------------------------

		double fx, fy, fz;
		if(xr > epsilon)	fx = cbrt(xr);
		else				fx = (kappa*xr + 16.0)/116.0;
		if(yr > epsilon)	fy = cbrt(yr);
		else				fy = (kappa*yr + 16.0)/116.0;
		if(zr > epsilon)	fz = cbrt(zr);
		else				fz = (kappa*zr + 16.0)/116.0;

		labvec[j].l = 116.0*fy-16.0;
		labvec[j].a = 500.0*(fx-fy);
		labvec[j].b = 200.0*(fy-fz);
	}
}

//==============================================================================
///	DetectLabEdges
//==============================================================================
void SLIC::DetectLabEdges(
	const lab*					labvec,
	const int&					width,
	const int&					height,
	vector<double>&				edges)
{
	int sz = width*height;

	edges.resize(sz,0);
	for( int j = 1; j < height-1; j++ )
	{
		for( int k = 1; k < width-1; k++ )
		{
			int i = j*width+k;

			double dx = (labvec[i-1].l-labvec[i+1].l)*(labvec[i-1].l-labvec[i+1].l) +
						(labvec[i-1].a-labvec[i+1].a)*(labvec[i-1].a-labvec[i+1].a) +
						(labvec[i-1].b-labvec[i+1].b)*(labvec[i-1].b-labvec[i+1].b);

			double dy = (labvec[i-width].l-labvec[i+width].l)*(labvec[i-width].l-labvec[i+width].l) +
						(labvec[i-width].a-labvec[i+width].a)*(labvec[i-width].a-labvec[i+width].a) +
						(labvec[i-width].b-labvec[i+width].b)*(labvec[i-width].b-labvec[i+width].b);

			//edges[i] = (sqrt(dx) + sqrt(dy));
			edges[i] = (dx + dy);
		}
	}
}

//===========================================================================
///	PerturbSeeds
//===========================================================================
void SLIC::PerturbSeeds(
	vector<labxy>&				kseeds,
	const vector<double>&		edges)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	
	int numseeds = kseeds.size();

	for( int n = 0; n < numseeds; n++ )
	{
		int ox = kseeds[n].x;//original x
		int oy = kseeds[n].y;//original y
		int oind = oy*m_width + ox;

		int storeind = oind;
		for( int i = 0; i < 8; i++ )
		{
			int nx = ox+dx8[i];//new x
			int ny = oy+dy8[i];//new y

			if( nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				int nind = ny*m_width + nx;
				if( edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if(storeind != oind)
		{
			kseeds[n].x = storeind%m_width;
			kseeds[n].y = storeind/m_width;
			kseeds[n].l = m_labvec[storeind].l;
			kseeds[n].a = m_labvec[storeind].a;
			kseeds[n].b = m_labvec[storeind].b;
		}
	}
}

//===========================================================================
///	GetLABXYSeeds_ForGivenK
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLIC::GetLABXYSeeds_ForGivenK(
	vector<labxy>&				kseeds,
	const int&					K,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int sz = m_width*m_height;
	double step = sqrt(double(sz)/double(K));
	int T = step;
	int xoff = step/2;
	int yoff = step/2;
	
	int n(0);int r(0);
	for( int y = 0; y < m_height; y++ )
	{
		int Y = y*step + yoff;
		if( Y > m_height-1 ) break;

		for( int x = 0; x < m_width; x++ )
		{
			//int X = x*step + xoff;//square grid
			int X = x*step + (xoff<<(r&0x1));//hex grid
			if(X > m_width-1) break;

			int i = Y*m_width + X;

			//_ASSERT(n < K);
			
			//kseeds[n].l = m_labvec[i].l;
			//kseeds[n].a = m_labvec[i].a;
			//kseeds[n].b = m_labvec[i].b;
			//kseeds[n].x = X;
			//kseeds[n].y = Y;
			kseeds.push_back({m_labvec[i].l, m_labvec[i].a, m_labvec[i].b, X, Y});
			n++;
		}
		r++;
	}

	if(perturbseeds)
	{
		PerturbSeeds(kseeds, edgemag);
	}
}

//===========================================================================
///	PerformSuperpixelSegmentation_VariableSandM
///
///	Magic SLIC - no parameters
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
/// This function picks the maximum value of color distance as compact factor
/// M and maximum pixel distance as grid step size S from each cluster (13 April 2011).
/// So no need to input a constant value of M and S. There are two clear
/// advantages:
///
/// [1] The algorithm now better handles both textured and non-textured regions
/// [2] There is not need to set any parameters!!!
///
/// SLICO (or SLIC Zero) dynamically varies only the compactness factor S,
/// not the step size S.
//===========================================================================
void SLIC::PerformSuperpixelSegmentation_VariableSandM(
	vector<labxy>&				kseeds,
	int*						klabels,
	const int&					STEP,
	const int&					NUMITR)
{
	int sz = m_width*m_height;
	const int numk = kseeds.size();
	//double cumerr(99999.9);
	int numitr(0);

	//----------------
	int offset = STEP;
	if(STEP < 10) offset = STEP*1.5;
	//----------------
	vector<labxy> sigma(numk);
	/*
	double *sigmal = new double[numk];
	double *sigmaa = new double[numk];
	double *sigmab = new double[numk];
	double *sigmax = new double[numk];
	double *sigmay = new double[numk];
	double *inv = new double[numk];
	*/
	vector<int> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values
	vector<double> dist(sz);
	vector<double> distvec(sz);
	vector<double> maxlab(numk, double(100));
	// vector<double> maxlab(numk, 10*10);//THIS IS THE VARIABLE VALUE OF M, just start with 10
	// vector<double> maxxy(numk, STEP*STEP);//THIS IS THE VARIABLE VALUE OF M, just start with 10

	double invxywt = 1.0/(STEP*STEP);//NOTE: this is different from how usual SLIC/LKM works
	while( numitr < NUMITR )
	{
		//------
		//cumerr = 0;
		numitr++;
		//------
		memset(&distvec[0], 0x7f, sizeof(double) * sz);
		// distvec.assign(sz, DBL_MAX);
		vector<vector<int>> graph(numk);
		vector<int> in_degree(numk, 0);
		vector<vector<int>> layers;
		for (int i = 0; i < numk; ++i)
		{
			int y1 = max(0, (int)(kseeds[i].y - 2 * offset));
			int y2 = min(m_height, (int)(kseeds[i].y + 2 * offset));
			int x1 = max(0, (int)(kseeds[i].x - 2 * offset));
			int x2 = min(m_width, (int)(kseeds[i].x + 2 * offset));
			for (int j = i + 1; j < numk; ++j)
			{
				if (
						y1 <= kseeds[j].y && kseeds[j].y < y2 &&
						x1 <= kseeds[j].x && kseeds[j].x < x2)
				{
					graph[i].push_back(j);
					in_degree[j]++;
				}
			}
		}
		vector<char> vis(numk, 0);
		for (int ncnt = 0; ncnt < numk;)
		{
			vector<int> cur_layer;
			for (int i = 0; i < numk; ++i)
			{
				if (!vis[i] && in_degree[i] == 0)
				{
					cur_layer.push_back(i);
					vis[i] = 1;
					ncnt++;
				}
			}
			for (int i = 0; i < cur_layer.size(); ++i)
			{
				for (int j = 0; j < graph[cur_layer[i]].size(); ++j)
				{
					in_degree[graph[cur_layer[i]][j]]--;
				}
			}
			layers.push_back(std::move(cur_layer));
		}
		omp_set_nested(true);
		omp_set_dynamic(true);
		for (int layer = 0; layer < layers.size(); ++layer) {
			#pragma omp parallel for schedule(dynamic, 1) num_threads(4)
			for (int in = 0; in < layers[layer].size(); ++in) {
				int n = layers[layer][in];
				int y1 = max(0, (int)(kseeds[n].y-offset));
				int y2 = min(m_height, (int)(kseeds[n].y+offset));
				int x1 = max(0,	(int)(kseeds[n].x-offset));
				int x2 = min(m_width, (int)(kseeds[n].x+offset));
				#pragma omp parallel for simd collapse(2) num_threads(16)
				for (int y = y1; y < y2; y++)
				{
					for (int x = x1; x < x2; x++)
					{
						int i = y * m_width + x;
						//_ASSERT( y < m_height && x < m_width && y >= 0 && x >= 0 );

						double l = m_labvec[i].l;
						double a = m_labvec[i].a;
						double b = m_labvec[i].b;
						double lab = (l - kseeds[n].l) * (l - kseeds[n].l) +
												 (a - kseeds[n].a) * (a - kseeds[n].a) +
												 (b - kseeds[n].b) * (b - kseeds[n].b);
						double xy = (x - kseeds[n].x) * (x - kseeds[n].x) +
												(y - kseeds[n].y) * (y - kseeds[n].y);
						//------------------------------------------------------------------------
						double distv = lab / maxlab[n] + xy * invxywt; //only varying m, prettier superpixels
						//double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
						//------------------------------------------------------------------------
						dist[i] = lab;
						if (distv < distvec[i])
						{
							distvec[i] = distv;
							klabels[i] = n;
						}
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Assign the max color distance for a cluster
		//-----------------------------------------------------------------
		if(0 == numitr)
		{
			maxlab.assign(numk, 1.0);
		}
		clustersize.assign(numk, 0);
		sigma.assign(numk, labxy());
		struct labxydz
		{
			double l, a, b, x, y, dst;
			int sz;
		};
		vector<vector<labxydz>> sum_tmp(omp_get_max_threads(), vector<labxydz>(numk));
		for (int i = 0; i < sum_tmp.size(); ++i) {
				for (int j = 0; j < sum_tmp[i].size(); ++j) {
						sum_tmp[i][j].dst = maxlab[j];
				}
		}
		#pragma omp parallel for simd
		for( int j = 0; j < sz; j++ )
		{
				int temp = klabels[j];
				int id = omp_get_thread_num();
				sum_tmp[id][temp].dst = max(sum_tmp[id][temp].dst, dist[j]);
				// if(maxlab[temp] < dist[j])
				// maxlab[temp] = dist[j];
				sum_tmp[id][temp].l += m_labvec[j].l;
				sum_tmp[id][temp].a += m_labvec[j].a;
				sum_tmp[id][temp].b += m_labvec[j].b;
				sum_tmp[id][temp].x += (j%m_width);
				sum_tmp[id][temp].y += (j/m_width);
				sum_tmp[id][temp].sz ++;
				//_ASSERT(klabels[j] >= 0);
				// sigma[temp].l += m_labvec[j / vec_width].l[j % vec_width];
				// sigma[temp].a += m_labvec[j / vec_width].a[j % vec_width];
				// sigma[temp].b += m_labvec[j / vec_width].b[j % vec_width];
				// sigma[temp].x += (j%m_width);
				// sigma[temp].y += (j/m_width);

				// clustersize[temp]++;
		}
#pragma omp simd
		for (int i = 0; i < sum_tmp.size(); ++i) {
				for (int j = 0; j < sum_tmp[i].size(); ++j) {
						sigma[j].l += sum_tmp[i][j].l;
						sigma[j].a += sum_tmp[i][j].a;
						sigma[j].b += sum_tmp[i][j].b;
						sigma[j].x += sum_tmp[i][j].x;
						sigma[j].y += sum_tmp[i][j].y;
						clustersize[j] += sum_tmp[i][j].sz;
						maxlab[j] = max(sum_tmp[i][j].dst, maxlab[j]);
				}
		}

		{
			#pragma omp simd
			for( int k = 0; k < numk; k++ )
		{
			//_ASSERT(clustersize[k] > 0);
			if( clustersize[k] <= 0 ) clustersize[k] = 1;
			inv[k] = 1.0/double(clustersize[k]);//computing inverse now to multiply, than divide later
		}}


		{	
			#pragma omp simd
			for( int k = 0; k < numk; k++ )
			{
				kseeds[k].l = sigma[k].l*inv[k];
				kseeds[k].a = sigma[k].a*inv[k];
				kseeds[k].b = sigma[k].b*inv[k];
				kseeds[k].x = sigma[k].x*inv[k];
				kseeds[k].y = sigma[k].y*inv[k];
			}
			//__m256d kseedsl_v = _mm256_load_pd(kseedsl);
		}
	}
}

//===========================================================================
///	SaveSuperpixelLabels2PGM
///
///	Save labels to PGM in raster scan order.
//===========================================================================
void SLIC::SaveSuperpixelLabels2PPM(
	char*                           filename, 
	int *                           labels, 
	const int                       width, 
	const int                       height)
{
    FILE* fp;
    char header[20];
 
    fp = fopen(filename, "wb");
 
    // write the PPM header info, such as type, width, height and maximum
    fprintf(fp,"P6\n%d %d\n255\n", width, height);
 
    // write the RGB data
    unsigned char *rgb = new unsigned char [ (width)*(height)*3 ];
    int k = 0;
	unsigned char c = 0;
    for ( int i = 0; i < (height); i++ ) {
        for ( int j = 0; j < (width); j++ ) {
			c = (unsigned char)(labels[k]);
            rgb[i*(width)*3 + j*3 + 2] = labels[k] >> 16 & 0xff;  // r
            rgb[i*(width)*3 + j*3 + 1] = labels[k] >> 8  & 0xff;  // g
            rgb[i*(width)*3 + j*3 + 0] = labels[k]       & 0xff;  // b

			// rgb[i*(width) + j + 0] = c;
            k++;
        }
    }
    fwrite(rgb, width*height*3, 1, fp);

    delete [] rgb;
 
    fclose(fp);

}

//===========================================================================
///	EnforceLabelConnectivity
///
///		1. finding an adjacent label for each new component at the start
///		2. if a certain component is too small, assigning the previously found
///		    adjacent label to this component, and not incrementing the label.
//===========================================================================
void SLIC::EnforceLabelConnectivity(
	const int*					labels,//input labels that need to be corrected to remove stray labels
	const int&					width,
	const int&					height,
	int*						nlabels,//new labels
	int&						numlabels,//the number of labels changes in the end if segments are removed
	const int&					K) //the number of superpixels desired by the user
{
//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};

	const int sz = width*height;
	const int SUPSZ = sz/K;
	//nlabels.resize(sz, -1);
	for( int i = 0; i < sz; i++ ) nlabels[i] = -1;
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			if( 0 > nlabels[oindex] )
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{for( int n = 0; n < 4; n++ )
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = y*width + x;
						if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}}

				int count(1);
				for( int c = 0; c < count; c++ )
				{
					for( int n = 0; n < 4; n++ )
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
						{
							int nindex = y*width + x;

							if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if(count <= SUPSZ >> 2)
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]*width+xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;

	if(xvec) delete [] xvec;
	if(yvec) delete [] yvec;
}

//===========================================================================
///	PerformSLICO_ForGivenK
///
/// Zero parameter SLIC algorithm for a given number K of superpixels.
//===========================================================================
void SLIC::PerformSLICO_ForGivenK(
	const unsigned int*			ubuff,
	const int					width,
	const int					height,
	int*						klabels,
	int&						numlabels,
	const int&					K,//required number of superpixels
	const double&				m)//weight given to spatial distance
{
	vector<labxy> kseeds;
	kseeds.assign(kseeds.size(), labxy());

	//--------------------------------------------------
	m_width  = width;
	m_height = height;
	int sz = m_width*m_height;
	//--------------------------------------------------
	//if(0 == klabels) klabels = new int[sz];
	for( int s = 0; s < sz; s++ ) klabels[s] = -1;
	//--------------------------------------------------
	auto startTime = Clock::now();
	if(1)//LAB
	{
		DoRGBtoLABConversion(ubuff, m_labvec);
	}
	else//RGB
	{
		m_labvec = new lab[sz];
		for( int i = 0; i < sz; i++ )
		{
			m_labvec[i].l = ubuff[i] >> 16 & 0xff;
			m_labvec[i].a = ubuff[i] >>  8 & 0xff;
			m_labvec[i].b = ubuff[i]       & 0xff;
		}
	}
	//--------------------------------------------------
	auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
	cout << compTime.count() << endl;
	bool perturbseeds(true);
	vector<double> edgemag(0);
	
	startTime = Clock::now();
	if(perturbseeds) DetectLabEdges(m_labvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenK(kseeds, K, perturbseeds, edgemag);
	endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
	cout << compTime.count() << endl;

	startTime = Clock::now();
	int STEP = sqrt(double(sz)/double(K)) + 2.0;//adding a small value in the even the STEP size is too small.
	PerformSuperpixelSegmentation_VariableSandM(kseeds, klabels,STEP,10);
	numlabels = kseeds.size();
	endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
	cout << compTime.count() << endl;

	startTime = Clock::now();
	int* nlabels = new int[sz];
	EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, K);
	endTime = Clock::now();
    compTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
	cout << compTime.count() << endl;
	{for(int i = 0; i < sz; i++ ) klabels[i] = nlabels[i];}
	if(nlabels) delete [] nlabels;
}

//===========================================================================
/// Load PPM file
///
/// 
//===========================================================================
void LoadPPM(char* filename, unsigned int** data, int* width, int* height)
{
    char header[1024];
    FILE* fp = NULL;
    int line = 0;
 
    fp = fopen(filename, "rb");
 
    // read the image type, such as: P6
    // skip the comment lines
    while (line < 2) {    
        fgets(header, 1024, fp);
        if (header[0] != '#') {
            ++line;
        }
    }
    // read width and height
    sscanf(header,"%d %d\n", width, height);
 
    // read the maximum of pixels
    fgets(header, 20, fp);
 
    // get rgb data
    unsigned char *rgb = new unsigned char [ (*width)*(*height)*3 ];
    fread(rgb, (*width)*(*height)*3, 1, fp);

    *data = new unsigned int [ (*width)*(*height)*4 ];
    int k = 0;
    for ( int i = 0; i < (*height); i++ ) {
        for ( int j = 0; j < (*width); j++ ) {
            unsigned char *p = rgb + i*(*width)*3 + j*3;
                                      // a ( skipped )
            (*data)[k]  = p[2] << 16; // r
            (*data)[k] |= p[1] << 8;  // g
            (*data)[k] |= p[0];       // b
            k++;
        }
    }

    // ofc, later, you'll have to cleanup
    delete [] rgb;
 
    fclose(fp);
}

//===========================================================================
/// Load PPM file
///
/// 
//===========================================================================
int CheckLabelswithPPM(char* filename, int* labels, int width, int height)
{
    char header[1024];
    FILE* fp = NULL;
    int line = 0, ground = 0;
 
    fp = fopen(filename, "rb");
 
    // read the image type, such as: P6
    // skip the comment lines
    while (line < 2) {    
        fgets(header, 1024, fp);
        if (header[0] != '#') {
            ++line;
        }
    }
    // read width and height
	int w(0);
	int h(0);
    sscanf(header,"%d %d\n", &w, &h);
	if (w != width || h != height) return -1;
 
    // read the maximum of pixels
    fgets(header, 20, fp);
 
    // get rgb data
    unsigned char *rgb = new unsigned char [ (w)*(h)*3 ];
    fread(rgb, (w)*(h)*3, 1, fp);

    int num = 0, k = 0;
    for ( int i = 0; i < (h); i++ ) {
        for ( int j = 0; j < (w); j++ ) {
            unsigned char *p = rgb + i*(w)*3 + j*3;
                                  // a ( skipped )
            ground  = p[2] << 16; // r
            ground |= p[1] << 8;  // g
            ground |= p[0];       // b
            
			if (ground != labels[k])
				num++;

			k++;
        }
    }

    // ofc, later, you'll have to cleanup
    delete [] rgb;
 
    fclose(fp);

	return num;
}

//===========================================================================
///	The main function
///
//===========================================================================
int main (int argc, char **argv)
{
	unsigned int* img = NULL;
	int width(0);
	int height(0);

	LoadPPM((char *)"input_image.ppm", &img, &width, &height);
	if (width == 0 || height == 0) return -1;

	int sz = width*height;
	int* labels = new int[sz];
	int numlabels(0);
	SLIC slic;
	int m_spcount;
	double m_compactness;
	m_spcount = 200;
	m_compactness = 10.0;
    auto startTime = Clock::now();
	// cout << "start" << endl;
	slic.PerformSLICO_ForGivenK(img, width, height, labels, numlabels, m_spcount, m_compactness);//for a given number K of superpixels
    auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout <<  "Computing time=" << compTime.count()/1000 << " ms" << endl;

	int num = CheckLabelswithPPM((char *)"check.ppm", labels, width, height);
	if (num < 0) {
		cout <<  "The result for labels is different from output_labels.ppm." << endl;
	} else {
		cout <<  "There are " << num << " points' labels are different from original file." << endl;
	}
	
	slic.SaveSuperpixelLabels2PPM((char *)"output_labels.ppm", labels, width, height);
	if(labels) delete [] labels;
	
	if(img) delete [] img;

	return 0;
}
