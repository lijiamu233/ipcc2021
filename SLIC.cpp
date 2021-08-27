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
#include <mpi.h>

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
	if(m_labvec) _mm_free(m_labvec);
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
	vec_lab*&					labvec,
	int                         myrank)
{
	int sz = m_width*m_height;
	labvec = (vec_lab*)_mm_malloc(sizeof(vec_lab) * (sz / vec_width + 1), 32);
	double epsilon = 0.008856;	//actual CIE standard
	double kappa   = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	int local_start, local_end;
	if(myrank == 0)
	{
		local_start = 0;
		local_end = sz/2;
	}
	else
	{
		local_start = sz/2;
		local_end = sz;
	}
	#pragma omp parallel for
	for( int j = local_start; j < local_end; j += vec_width)
	{
		#pragma omp simd
		for (int i = 0; i < vec_width; i ++) {
			int sR = (ubuff[j + i] >> 16) & 0xFF;
			int sG = (ubuff[j + i] >>  8) & 0xFF;
			int sB = (ubuff[j + i]      ) & 0xFF;
			double R = sR/255.0;
			double G = sG/255.0;
			double B = sB/255.0;

			// __m256d drgb = _mm256_set_pd(0.0, R, G, B);
			// __m256d drgbdiv12 = _mm256_div_pd(drgb, _mm256_set1_pd(12.92));
			// __m256d drgbpow = _mm256_pow_pd(_mm256_fmadd_pd(drgb, _mm256_set1_pd(1.0/1.055), _mm256_set1_pd(0.05213270142180095)), _mm256_set1_pd(2.4)));
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

			labvec[j / vec_width].l[i] = 116.0*fy-16.0;
			labvec[j / vec_width].a[i] = 500.0*(fx-fy);
			labvec[j / vec_width].b[i] = 200.0*(fy-fz);
		}
	}

	if(myrank == 0)
	{
		MPI_Recv(&(labvec[0].l[0]) + 3 * sz / 2, 3 * sz / 2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else
	{
		MPI_Send(&(labvec[0].l[0]) + 3 * sz / 2, 3 * sz / 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
}

//==============================================================================
///	DetectLabEdges
//==============================================================================
void SLIC::DetectLabEdges(
	const vec_lab*				labvec,
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
			int im1a = (i - 1) / vec_width, im1b = (i - 1) % vec_width;
			int ip1a = (i + 1) / vec_width, ip1b = (i + 1) % vec_width;
			int imwa = (i - width) / vec_width, imwb = (i - width) % vec_width;
			int ipwa = (i + width) / vec_width, ipwb = (i + width) % vec_width;
			double dx = (labvec[im1a].l[im1b]-labvec[ip1a].l[ip1b])*(labvec[im1a].l[im1b]-labvec[ip1a].l[ip1b]) +
						(labvec[im1a].a[im1b]-labvec[ip1a].a[ip1b])*(labvec[im1a].a[im1b]-labvec[ip1a].a[ip1b]) +
						(labvec[im1a].b[im1b]-labvec[ip1a].b[ip1b])*(labvec[im1a].b[im1b]-labvec[ip1a].b[ip1b]);

			double dy = (labvec[imwa].l[imwb]-labvec[ipwa].l[ipwb])*(labvec[imwa].l[imwb]-labvec[ipwa].l[ipwb]) +
						(labvec[imwa].a[imwb]-labvec[ipwa].a[ipwb])*(labvec[imwa].a[imwb]-labvec[ipwa].a[ipwb]) +
						(labvec[imwa].b[imwb]-labvec[ipwa].b[ipwb])*(labvec[imwa].b[imwb]-labvec[ipwa].b[ipwb]);

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
			kseeds[n].l = m_labvec[storeind / vec_width].l[storeind % vec_width];
			kseeds[n].a = m_labvec[storeind / vec_width].a[storeind % vec_width];
			kseeds[n].b = m_labvec[storeind / vec_width].b[storeind % vec_width];
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
			kseeds.push_back({m_labvec[i / vec_width].l[i % vec_width], m_labvec[i / vec_width].a[i % vec_width], m_labvec[i / vec_width].b[i%vec_width], double(X), double(Y)});
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
alignas(32) int32_t lookuptable[16][4] = {
	{0, 0, 0, 0},
	{-1, 0, 0, 0},
	{0, -1, 0, 0},
	{-1, -1, 0, 0},
	{0, 0, -1, 0},
	{-1, 0, -1, 0},
	{0, -1, -1, 0},
	{-1, -1, -1, 0},
	{0, 0, 0, -1},
	{-1, 0, 0, -1},
	{0, -1, 0, -1},
	{-1, -1, 0, -1},
	{0, 0, -1, -1},
	{-1, 0, -1, -1},
	{0, -1, -1, -1},
	{-1, -1, -1, -1},
};

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
	vector<int> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values
	double* dist = (double *)_mm_malloc(sizeof(double) * sz, 32);
	// vector<vec_dist> dist(sz);
	double* distvec = (double *)_mm_malloc(sizeof(double) * (sz), 32);
	vector<double> maxlab(numk, double(100));
	// vector<double> maxlab(numk, 10*10);//THIS IS THE VARIABLE VALUE OF M, just start with 10
	// vector<double> maxxy(numk, STEP*STEP);//THIS IS THE VARIABLE VALUE OF M, just start with 10

	double invxywt = 1.0/(STEP*STEP);//NOTE: this is different from how usual SLIC/LKM works
	__m256d vinvxywt = _mm256_set1_pd(invxywt);

	while( numitr < NUMITR )
	{
		//------
		//cumerr = 0;
		numitr++;
		//------
		memset(&distvec[0], 0x7f, sizeof(double) * sz);
		vector<vector<int>> graph(numk);
		vector<int> in_degree(numk, 0);
		vector<vector<int>> layers;
		for (int i = 0; i < numk; ++i) {
			int y1 = max(0,			(int)(kseeds[i].y- 2 * offset));
			int y2 = min(m_height,	(int)(kseeds[i].y+ 2 * offset));
			int x1 = max(0,			(int)(kseeds[i].x- 2 * offset));
			int x2 = min(m_width,	(int)(kseeds[i].x+ 2 * offset));
			for (int j = i + 1; j < numk; ++j) {
				if (
					y1 <= kseeds[j].y && kseeds[j].y < y2 &&
					x1 <= kseeds[j].x && kseeds[j].x < x2
				) {
					graph[i].push_back(j);
					in_degree[j]++;
				}
			}
		}
		vector<char> vis(numk, 0);
		for (int ncnt = 0; ncnt < numk; ) {
			vector<int> cur_layer;
			for (int i = 0; i < numk; ++i) {
				if (!vis[i] && in_degree[i] == 0) {
					cur_layer.push_back(i);
					vis[i] = 1;
					ncnt++;
				}
			}
			for (int i = 0; i < cur_layer.size(); ++i) {
				for (int j = 0; j < graph[cur_layer[i]].size(); ++j) {
					in_degree[graph[cur_layer[i]][j]]--;
				}
			}
			layers.push_back(std::move(cur_layer));
		}
		// cout << "========================\n";
		// for (auto & i : layers) {
		// 	for (auto &j : i) {
		// 		cout << j << ' ';
		// 	}
		// 	cout << endl;
		// }
		
		for (int layer = 0; layer < layers.size(); ++layer) {
			#pragma omp parallel for schedule(dynamic, 1)
			for (int in = 0; in < layers[layer].size(); ++in) {
				int n = layers[layer][in];
				int y1 = max(0,			(int)(kseeds[n].y-offset));
				int y2 = min(m_height,	(int)(kseeds[n].y+offset));
				int x1 = max(0,			(int)(kseeds[n].x-offset));
				int x2 = min(m_width,	(int)(kseeds[n].x+offset));

				for( int y = y1; y < y2; y++ )
				{
					ptrdiff_t start_offset = (y * m_width + x1) % vec_width == 0 ? 0 : vec_width - (y * m_width + x1) % vec_width;
					ptrdiff_t end_offset = (y * m_width + x2) % vec_width;
					for (int x = x1; x < start_offset + x1; ++x) {
						int i = y*m_width + x;
						double l = m_labvec[i / vec_width].l[i % vec_width];
						double a = m_labvec[i / vec_width].a[i % vec_width];
						double b = m_labvec[i / vec_width].b[i % vec_width];
						double lab = (l - kseeds[n].l)*(l - kseeds[n].l) +
										(a - kseeds[n].a)*(a - kseeds[n].a) +
										(b - kseeds[n].b)*(b - kseeds[n].b);
						double xy = (x - kseeds[n].x)*(x - kseeds[n].x) +
										(y - kseeds[n].y)*(y - kseeds[n].y);							
						//------------------------------------------------------------------------
						double distv = lab/maxlab[n] + xy*invxywt;//only varying m, prettier superpixels
						//double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
						//------------------------------------------------------------------------
						dist[i] = lab;
						if( distv < distvec[i] )
						{
							distvec[i] = distv;
							klabels[i]  = n;
						}
					}
					__m256d vseedsl, vseedsa, vseedsb, vseedsx, vseedsy, vmaxlab;
					vseedsl = _mm256_set1_pd(kseeds[n].l);
					vseedsa = _mm256_set1_pd(kseeds[n].a);
					vseedsb = _mm256_set1_pd(kseeds[n].b);
					vseedsx = _mm256_set1_pd(kseeds[n].x);
					vseedsy = _mm256_set1_pd(kseeds[n].y);
					vmaxlab = _mm256_set1_pd(maxlab[n]);
					__m256d vy = _mm256_set1_pd(double(y));
					// #pragma omp parallel for
					for(int x = x1 + start_offset; x < x2 - end_offset; x += vec_width )
					{
						for (int t = 0; t < vec_width; t += avx_width) {
							// int t = 0;
							int i = y*m_width + x + t;
							__m256d vl, va, vb, vx;
							__m256d vlab, vxy, vdistv;
							vl = _mm256_load_pd(&m_labvec[i / vec_width].l[t]);
							// double l = m_labvec[i / vec_width].l[i % vec_width];
							va = _mm256_load_pd(&m_labvec[i / vec_width].a[t]);
							// double a = m_labvec[i / vec_width].a[i % vec_width];
							vb = _mm256_load_pd(&m_labvec[i / vec_width].b[t]);
							// double b = m_labvec[i / vec_width].b[i % vec_width];
							vlab = _mm256_fmadd_pd(
								_mm256_sub_pd(vb, vseedsb), 
								_mm256_sub_pd(vb, vseedsb),
								_mm256_fmadd_pd(
									_mm256_sub_pd(va, vseedsa), 
									_mm256_sub_pd(va, vseedsa), 
									_mm256_mul_pd(
										_mm256_sub_pd(vl, vseedsl), 
										_mm256_sub_pd(vl, vseedsl)
							)));
							vx = _mm256_set_pd(double(x + t + 3), double(x + t + 2), double(x + t + 1), double(x + t));

							// double lab = (l - kseeds[n].l)*(l - kseeds[n].l) +
											// (a - kseeds[n].a)*(a - kseeds[n].a) +
											// (b - kseeds[n].b)*(b - kseeds[n].b);
							vxy = _mm256_fmadd_pd(
								_mm256_sub_pd(vx, vseedsx),
								_mm256_sub_pd(vx, vseedsx),
								_mm256_mul_pd(
									_mm256_sub_pd(vy, vseedsy),
									_mm256_sub_pd(vy, vseedsy)
							));
							// double xy = (x - kseeds[n].x)*(x - kseeds[n].x) +
											// (y - kseeds[n].y)*(y - kseeds[n].y);	
							_mm256_stream_pd(&dist[i], vlab);						
							//------------------------------------------------------------------------
							vdistv = _mm256_fmadd_pd(
								vxy,
								vinvxywt,
								_mm256_div_pd(vlab, vmaxlab)
							);
							// double distv = lab/maxlab[n] + xy*invxywt;//only varying m, prettier superpixels
							//double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
							//------------------------------------------------------------------------
							// dist[i].lab = lab;
							__m256d mask = _mm256_cmp_pd(vdistv, _mm256_load_pd(distvec + i), _CMP_NGE_UQ);
							int imask = _mm256_movemask_pd(mask);
							_mm_maskstore_epi32(klabels + i, _mm_load_si128((__m128i *)(void*)(lookuptable + imask)), _mm_set1_epi32(n));
							_mm256_maskstore_pd(distvec + i, (__m256i)_mm256_castpd_ps(mask), vdistv);
						}
					}

					for (int x = x2 - end_offset; x < x2; ++x) {
						int i = y*m_width + x;
						//_ASSERT( y < m_height && x < m_width && y >= 0 && x >= 0 );
						double l = m_labvec[i / vec_width].l[i % vec_width];
						double a = m_labvec[i / vec_width].a[i % vec_width];
						double b = m_labvec[i / vec_width].b[i % vec_width];
						double lab = (l - kseeds[n].l)*(l - kseeds[n].l) +
										(a - kseeds[n].a)*(a - kseeds[n].a) +
										(b - kseeds[n].b)*(b - kseeds[n].b);
						double xy = (x - kseeds[n].x)*(x - kseeds[n].x) +
										(y - kseeds[n].y)*(y - kseeds[n].y);							
						//------------------------------------------------------------------------
						double distv = lab/maxlab[n] + xy*invxywt;//only varying m, prettier superpixels
						//double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
						//------------------------------------------------------------------------
						dist[i] = lab;
						if( distv < distvec[i] )
						{
							distvec[i] = distv;
							klabels[i]  = n;
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
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		sigma.assign(numk, labxy());
		clustersize.assign(numk, 0);
		#pragma omp simd
		for( int j = 0; j < sz; j++ )
		{
			int temp = klabels[j];
			if(maxlab[temp] < dist[j]) 
				maxlab[temp] = dist[j];
			//_ASSERT(klabels[j] >= 0);
			sigma[temp].l += m_labvec[j / vec_width].l[j % vec_width];
			sigma[temp].a += m_labvec[j / vec_width].a[j % vec_width];
			sigma[temp].b += m_labvec[j / vec_width].b[j % vec_width];
			sigma[temp].x += (j%m_width);
			sigma[temp].y += (j/m_width);

			clustersize[temp]++;
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
    //char header[20];
 
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
	const double&				m,//weight given to spatial distance
	int                         myrank)
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
		DoRGBtoLABConversion(ubuff, m_labvec, myrank);
	}
	else//RGB
	{
		// m_labvec = new lab[sz];
		// for( int i = 0; i < sz; i++ )
		// {
		// 	m_labvec[i].l = ubuff[i] >> 16 & 0xff;
		// 	m_labvec[i].a = ubuff[i] >>  8 & 0xff;
		// 	m_labvec[i].b = ubuff[i]       & 0xff;
		// }
	}
	//--------------------------------------------------
	auto endTime = Clock::now();
    auto compTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime);
	cout << compTime.count() << endl;
	bool perturbseeds(true);
	vector<double> edgemag(0);
	
	if(myrank == 0)
	{
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
            
			if (ground != labels[k]) {
				num++;
			}

			k++;
        }
    }

    // ofc, later, you'll have to cleanup
    delete [] rgb;
 
    fclose(fp);

	return num;
}

void SLIC::Build_mpi_type(double l_p[4], double a_p[4], double b_p[4], MPI_Datatype* mpi_lab)
{
	int array_of_blocklengths[3]={4, 4, 4};
	MPI_Datatype array_of_types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
	MPI_Aint l_addr, a_addr, b_addr;
	MPI_Aint array_of_displacements[3] = {0};
	MPI_Get_address(&l_p, &l_addr);
	MPI_Get_address(&a_p, &a_addr);
	MPI_Get_address(&b_p, &b_addr);
	array_of_displacements[1] = a_addr - l_addr;
	array_of_displacements[2] = b_addr - l_addr;
	MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacements, array_of_types, mpi_lab);
	MPI_Type_commit(mpi_lab);
}
//===========================================================================
///	The main function
///
//===========================================================================
int main (int argc, char **argv)
{
	int myrank, comm_sz;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	unsigned int* img = NULL;
	int width(0);
	int height(0);

	LoadPPM((char *)"input_image.ppm", &img, &width, &height);
	if (width == 0 || height == 0) return -1;

	int sz = width*height;
	int* labels = (int*)_mm_malloc(sizeof(int) * sz, 32);
	int numlabels(0);
	SLIC slic;
	int m_spcount;
	double m_compactness;
	m_spcount = 200;
	m_compactness = 10.0;
    auto startTime = Clock::now();
	// cout << "start" << endl;
	slic.PerformSLICO_ForGivenK(img, width, height, labels, numlabels, m_spcount, m_compactness, myrank);//for a given number K of superpixels
	if(myrank == 0)
	{
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
	}
    
	if(labels) _mm_free(labels);
	
	if(img) delete [] img;

	MPI_Finalize();

	return 0;
}
