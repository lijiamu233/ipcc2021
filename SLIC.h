// SLIC.h: interface for the SLIC class.
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
//
//===========================================================================
// Copyright (c) 2013 Radhakrishna Achanta.
//
// For commercial use please contact the author:
//
// Email: firstname.lastname@epfl.ch
//===========================================================================

#if !defined(_SLIC_H_INCLUDED_)
#define _SLIC_H_INCLUDED_


#include <vector>
#include <string>
#include <algorithm>
#include <immintrin.h>
#include <cinttypes>
using namespace std;

struct labxy {
	double l, a, b;
	double x, y;
	labxy(double l, double a, double b, double x, double y) {
		this->l = l, this->a = a, this->b = b, this->x = x, this->y = y;
	}
	labxy() {
		l = a = b = x = y = 0;
	};
};

struct lab {
	double l, a, b;
};
constexpr int avx_width = 4;
constexpr int vec_unroll = 4;
constexpr uintptr_t align_mask = 0x1f;
constexpr uintptr_t align_rev_mask = ~align_mask;
constexpr int vec_width = avx_width * vec_unroll;
struct vec_lab {
 	double l[vec_width], a[vec_width], b[vec_width];
};

struct vec_dist {
	double lab[vec_width], xy[vec_width];
};

struct dist_t {
	double xy, lab;
};

class SLIC  
{
public:
	SLIC();
	virtual ~SLIC();

	//============================================================================
	// Superpixel segmentation for a given number of superpixels
	//============================================================================
	void PerformSLICO_ForGivenK(
		const unsigned int*			ubuff,//Each 32 bit unsigned int contains ARGB pixel values.
		const int					width,
		const int					height,
		int*						klabels,
		int&						numlabels,
		const int&					K,
		const double&				m);

	//============================================================================
	// Save superpixel labels to pgm in raster scan order
	//============================================================================
	void SaveSuperpixelLabels2PPM(
		char*                       filename, 
		int *                       labels, 
		const int                   width, 
		const int                   height);

private:

	//============================================================================
	// Magic SLIC. No need to set M (compactness factor) and S (step size).
	// SLICO (SLIC Zero) varies only M dynamicaly, not S.
	//============================================================================
	void PerformSuperpixelSegmentation_VariableSandM(
		vector<labxy>&				kseeds,
		int*						klabels,
		const int&					STEP,
		const int&					NUMITR);

	//============================================================================
	// Pick seeds for superpixels when number of superpixels is input.
	//============================================================================
	void GetLABXYSeeds_ForGivenK(
		vector<labxy>&				kseeds,
		const int&					STEP,
		const bool&					perturbseeds,
		const vector<double>&		edges);

	//============================================================================
	// Move the seeds to low gradient positions to avoid putting seeds at region boundaries.
	//============================================================================
	void PerturbSeeds(
		vector<labxy>&				kseeds,
		const vector<double>&		edges);
	
	//============================================================================
	// Detect color edges, to help PerturbSeeds()
	//============================================================================
	void DetectLabEdges(
		const vec_lab*				labvec,
		const int&					width,
		const int&					height,
		vector<double>&				edges);
	
	//============================================================================
	// xRGB to XYZ conversion; helper for RGB2LAB()
	//============================================================================
	void RGB2XYZ(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						X,
		double&						Y,
		double&						Z);
	
	//============================================================================
	// sRGB to CIELAB conversion
	//============================================================================
	void RGB2LAB(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						lval,
		double&						aval,
		double&						bval);
	
	//============================================================================
	// sRGB to CIELAB conversion for 2-D images
	//============================================================================
	void DoRGBtoLABConversion(
		const unsigned int*&		ubuff,
		vec_lab*&					labvec);

	//============================================================================
	// Post-processing of SLIC segmentation, to avoid stray labels.
	//============================================================================
	void EnforceLabelConnectivity(
		const int*					labels,
		const int&					width,
		const int&					height,
		int*						nlabels,//input labels that need to be corrected to remove stray labels
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					K); //the number of superpixels desired by the user

private:
	int										m_width;
	int										m_height;
	int										m_depth;

	vec_lab*								m_labvec;

	double**								m_lvecvec;
	double**								m_avecvec;
	double**								m_bvecvec;
};

#endif // !defined(_SLIC_H_INCLUDED_)
