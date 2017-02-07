/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2014-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdio.h>
#include <assert.h>
#include <complex.h>
#include <string.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/init.h"
#include "num/ops.h"

#include "wavelet2/wavelet.h"
#include "wavelet3/wavthresh.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#define DIMS 8

static float peak_bw = 0.0f;    // bytes/sec

complex float chk(long dims[DIMS], complex float *x) {
    long N = md_calc_size(DIMS, dims);
    complex float sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++)
        sum += x[i] * i;
    return sum;
}

static double bench_generic_copy(long dims[DIMS], complex float *check, bool forloop)
{
	long strs[DIMS];

	md_calc_strides(DIMS, strs, dims, CFL_SIZE);
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dims, CFL_SIZE);

    long N = md_calc_size(DIMS, dims);

	md_gaussian_rand(DIMS, dims, x);

	double tic, toc;

    for (int warmup = 0; warmup < 2; warmup++) {

        tic = timestamp();

        if (forloop) {
            #pragma omp parallel for
            for (int i = 0; i < N; i++)
                y[i] = x[i];
        } else {
            md_copy2(DIMS, dims, strs, y, strs, x, CFL_SIZE);
        }

	    toc = timestamp();
    }

    *check = chk(dims, y);

	md_free(x);
	md_free(y);

    long x_size = md_calc_size(DIMS, dims);
    long y_size = md_calc_size(DIMS, dims);
    long nbytes = sizeof(complex float) * (
        x_size +  // read x
        y_size    // write y
    );
    double sec = toc - tic;
    float bw = nbytes / sec;

	return bw;
}

	
static double bench_generic_matrix_multiply(long dims[DIMS], complex float *check)
{
	long dimsX[DIMS];
	long dimsY[DIMS];
	long dimsZ[DIMS];

	md_select_dims(DIMS, 2 * 3 + 17, dimsX, dims);	// 1 110 1
	md_select_dims(DIMS, 2 * 6 + 17, dimsY, dims);	// 1 011 1
	md_select_dims(DIMS, 2 * 5 + 17, dimsZ, dims);	// 1 101 1

	long strsX[DIMS];
	long strsY[DIMS];
	long strsZ[DIMS];

	md_calc_strides(DIMS, strsX, dimsX, CFL_SIZE);
	md_calc_strides(DIMS, strsY, dimsY, CFL_SIZE);
	md_calc_strides(DIMS, strsZ, dimsZ, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);
	complex float* z = md_alloc(DIMS, dimsZ, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_gaussian_rand(DIMS, dimsY, y);

	md_clear(DIMS, dimsZ, z, CFL_SIZE);

    // warmup
	md_zfmac2(DIMS, dims, strsZ, z, strsX, x, strsY, y);

	double tic = timestamp();

	md_zfmac2(DIMS, dims, strsZ, z, strsX, x, strsY, y);

	double toc = timestamp();

    *check = chk(dimsZ, z);

	md_free(x);
	md_free(y);
	md_free(z);

    // zfmac2: optr = optr + iptr1 * iptr2
    // mbd: Why is this function named 'matrix_multiply'?
    //      It looks like an elementwise product.
    long x_bytes = md_calc_size(DIMS, dimsX);
    long y_bytes = md_calc_size(DIMS, dimsY);
    long z_bytes = md_calc_size(DIMS, dimsZ);
    long nbytes = sizeof(complex float) * (
        x_bytes + // read x
        y_bytes + // read y
        z_bytes + // read z
        z_bytes   // write z
    );
	double sec = toc - tic;
    double bw = nbytes / sec;

    return bw;
}


static double bench_generic_add(long dims[DIMS], unsigned int flags, complex float *check, bool forloop)
{
	long dimsX[DIMS];
	long dimsY[DIMS];

	long dimsC[DIMS];

	md_select_dims(DIMS, flags, dimsX, dims);
	md_select_dims(DIMS, ~flags, dimsC, dims);
	md_select_dims(DIMS, ~0u, dimsY, dims);

	long strsX[DIMS];
	long strsY[DIMS];

	md_calc_strides(DIMS, strsX, dimsX, CFL_SIZE);
	md_calc_strides(DIMS, strsY, dimsY, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_gaussian_rand(DIMS, dimsY, y);

	long L = md_calc_size(DIMS, dimsC);
	long T = md_calc_size(DIMS, dimsX);

    double tic, toc;

    for (int warmup = 0; warmup < 2; warmup++) {

        tic = timestamp();

        if (forloop) {

            #pragma omp parallel for schedule(static) collapse(2)
            for (long j = 0; j < T; j++)

                for (long i = 0; i < L; i++) {
                    y[j * L + i] += x[j];
            }

        } else {

            md_zaxpy2(DIMS, dims, strsY, y, 1., strsX, x);
        }

        toc = timestamp();
    }

    *check = chk(dimsY, y);

	md_free(x);
	md_free(y);

    long x_size = md_calc_size(DIMS, dimsX);
    long y_size = md_calc_size(DIMS, dimsY);
    long nbytes = sizeof(complex float) * (
        x_size + // read x
        y_size + // read y
        y_size   // write y
    );
	double sec = toc - tic;
    double bw = nbytes / sec;

    return bw;
}


static double bench_generic_sum(long dims[DIMS], unsigned int flags, complex float *check, bool forloop)
{
	long dimsX[DIMS];
	long dimsY[DIMS];
	long dimsC[DIMS];

	md_select_dims(DIMS, ~0u, dimsX, dims);
	md_select_dims(DIMS, flags, dimsY, dims);
	md_select_dims(DIMS, ~flags, dimsC, dims);

	long strsX[DIMS];
	long strsY[DIMS];

	md_calc_strides(DIMS, strsX, dimsX, CFL_SIZE);
	md_calc_strides(DIMS, strsY, dimsY, CFL_SIZE);

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);

	md_gaussian_rand(DIMS, dimsX, x);
	md_clear(DIMS, dimsY, y, CFL_SIZE);

	long L = md_calc_size(DIMS, dimsC);
	long T = md_calc_size(DIMS, dimsY);

    double tic, toc;

    for (int warmup = 0; warmup < 2; warmup++) {

        tic = timestamp();

        if (forloop) {

            #pragma omp parallel for schedule(static)
            for (long j = 0; j < T; j++)
                for (long i = 0; i < L; i++) {
                    y[j] += x[i + j * L];
            }

        } else {

            md_zaxpy2(DIMS, dims, strsY, y, 1., strsX, x);
        }

        toc = timestamp();
    }

    *check = chk(dimsY, y);

	md_free(x);
	md_free(y);

    long x_size = md_calc_size(DIMS, dimsX);
    long y_size = md_calc_size(DIMS, dimsY);
    long nbytes = sizeof(complex float) * (
        x_size + // read x
        y_size + // read y
        y_size   // write y
    );
	double sec = toc - tic;
    double bw = nbytes / sec;

    return bw;
}

static double bench_copy1(long scale, complex float *check)
{
	long dims[DIMS] = { 1, 128 * scale, 128 * scale, 1, 1, 16, 1, 16 };
	return bench_generic_copy(dims, check, 0);
}

static double bench_copy1f(long scale, complex float *check)
{
	long dims[DIMS] = { 1, 128 * scale, 128 * scale, 1, 1, 16, 1, 16 };
	return bench_generic_copy(dims, check, 1);
}

static double bench_copy2(long scale, complex float *check)
{
	long dims[DIMS] = { 262144 * scale, 16, 1, 1, 1, 1, 1, 1 };
	return bench_generic_copy(dims, check, 0);
}

static double bench_copy2f(long scale, complex float *check)
{
	long dims[DIMS] = { 262144 * scale, 16, 1, 1, 1, 1, 1, 1 };
	return bench_generic_copy(dims, check, 1);
}


static double bench_matrix_mult(long scale, complex float *check)
{
	long dims[DIMS] = { 1, 256 * scale, 256 * scale, 256 * scale, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims, check);
}



static double bench_batch_matmul1(long scale, complex float *check)
{
	long dims[DIMS] = { 30000 * scale, 8, 8, 8, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims, check);
}



static double bench_batch_matmul2(long scale, complex float *check)
{
	long dims[DIMS] = { 1, 8, 8, 8, 30000 * scale, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims, check);
}


static double bench_tall_matmul1(long scale, complex float *check)
{
	long dims[DIMS] = { 1, 8, 8, 100000 * scale, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims, check);
}


static double bench_tall_matmul2(long scale, complex float *check)
{
	long dims[DIMS] = { 1, 100000 * scale, 8, 8, 1, 1, 1, 1 };
	return bench_generic_matrix_multiply(dims, check);
}


static double bench_add(long scale, complex float *check)
{
	long dims[DIMS] = { 65536 * scale, 1, 50 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_add(dims, MD_BIT(2), check, false);
}

static double bench_addf(long scale, complex float *check)
{
	long dims[DIMS] = { 65536 * scale, 1, 50 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_add(dims, MD_BIT(2), check, true);
}

static double bench_add2(long scale, complex float *check)
{
	long dims[DIMS] = { 50 * scale, 1, 65536 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_add(dims, MD_BIT(0), check, false);
}

static double bench_add2f(long scale, complex float *check)
{
	long dims[DIMS] = { 50 * scale, 1, 65536 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_add(dims, MD_BIT(0), check, true);
}

static double bench_sum2(long scale, complex float *check)
{
	long dims[DIMS] = { 50 * scale, 1, 65536 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_sum(dims, MD_BIT(0), check, false);
}

static double bench_sum(long scale, complex float *check)
{
	long dims[DIMS] = { 65536 * scale, 1, 50 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_sum(dims, MD_BIT(2), check, false);
}

static double bench_sumf(long scale, complex float *check)
{
	long dims[DIMS] = { 65536 * scale, 1, 50 * scale, 1, 1, 1, 1, 1 };
	return bench_generic_sum(dims, MD_BIT(2), check, true);
}

static double bench_generic_transpose(long scale, complex float *check, bool forloop)
{
	long dims[DIMS] = { 2000 * scale, 2000 * scale, 1, 1, 1, 1, 1, 1 };

	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dims, CFL_SIZE);
	
	md_gaussian_rand(DIMS, dims, x);
	md_clear(DIMS, dims, y, CFL_SIZE);

    double tic, toc;
    long D0 = dims[0];
    long D1 = dims[1];

    for (int warmup = 0; warmup < 2; warmup++) {

        tic = timestamp();

        if (forloop) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < D0; i++) {
            for (int j = 0; j < D1; j++) {
                y[i*D0+j] = x[j*D1+i];
            }}
        } else {
            md_transpose(DIMS, 0, 1, dims, y, dims, x, CFL_SIZE);
        }

        toc = timestamp();
    }

    *check = chk(dims, y);

	md_free(x);
	md_free(y);

    long x_size = md_calc_size(DIMS, dims);
    long y_size = md_calc_size(DIMS, dims);
    long nbytes = sizeof(complex float) * (x_size + y_size);
    double sec = toc - tic;
    double bw = nbytes / sec;
	
	return bw;
}

static double bench_transpose(long scale, complex float *check) {
    return bench_generic_transpose(scale, check, false);
}

static double bench_transposef(long scale, complex float *check) {
    return bench_generic_transpose(scale, check, true);
}

static double bench_resize(long scale, complex float *check, bool forloop)
{
	long dimsX[DIMS] = { 2000 * scale, 1000 * scale, 1, 1, 1, 1, 1, 1 };
	long dimsY[DIMS] = { 1000 * scale, 2000 * scale, 1, 1, 1, 1, 1, 1 };

	complex float* x = md_alloc(DIMS, dimsX, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dimsY, CFL_SIZE);
	
	md_gaussian_rand(DIMS, dimsX, x);
	md_clear(DIMS, dimsY, y, CFL_SIZE);

    double tic, toc;
    long N = md_calc_size(DIMS, dimsX);

    for (int warmup = 0; warmup < 2; warmup++) {

        tic = timestamp();

        if (forloop) {
            #pragma omp parallel for schedule(static)
            for (long i = 0; i < N; i++)
                y[i] = x[i];
            
        } else {
	        md_resize(DIMS, dimsY, y, dimsX, x, CFL_SIZE);
        }

        toc = timestamp();
    } 

    *check = chk(dimsY, y);

	md_free(x);
	md_free(y);
	
    long x_size = md_calc_size(DIMS, dimsX);
    long y_size = md_calc_size(DIMS, dimsY);
    long nbytes = sizeof(complex float) * (x_size + y_size);
    double sec = toc - tic;
    double bw = nbytes / sec;

    return bw;
}

static double bench_resizem(long scale, complex float *check) {
    return bench_resize(scale, check, false);
}

static double bench_resizef(long scale, complex float *check) {
    return bench_resize(scale, check, true);
}

static double bench_norm(int s, long scale, complex float *check, bool forloop)
{
	long dims[DIMS] = { 256 * scale, 256 * scale, 1, 16, 1, 1, 1, 1 };
#if 0
	complex float* x = md_alloc_gpu(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc_gpu(DIMS, dims, CFL_SIZE);
#else
	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* y = md_alloc(DIMS, dims, CFL_SIZE);
#endif

    long N = md_calc_size(DIMS, dims);
    long nbytes;
	
	md_gaussian_rand(DIMS, dims, x);
	md_gaussian_rand(DIMS, dims, y);

    double tic, toc;

    for (int warmup = 0; warmup < 2; warmup++) {

        float fnorm = 0;
        complex float znorm = 0;

        tic = timestamp();

        switch (s) {
        case 0:
            if (forloop) {
                #pragma omp parallel for schedule(static) reduction(+:znorm)
                for (long i = 0; i < N; i++)
                    znorm += conjf( x[i] ) * y[i];
            } else {
                znorm = md_zscalar(DIMS, dims, x, y);
            }
            nbytes = 2 * N * sizeof(complex float);
            *check = znorm;
            break;
        case 1:
            if (forloop) {
                #pragma omp parallel for schedule(static) reduction(+:fnorm)
                for (long i = 0; i < N; i++)
                    fnorm += creal( conjf( x[i] ) * y[i] );
            } else {
                fnorm = md_zscalar_real(DIMS, dims, x, y);
            }
            nbytes = 2 * N * sizeof(complex float);
            *check = fnorm;
            break;
        case 2:
            if (forloop) {
                #pragma omp parallel for schedule(static) reduction(+:znorm)
                for (long i = 0; i < N; i++)
                    znorm += conjf( x[i] ) * x[i];
                znorm = csqrt( znorm );
            } else {
                znorm = md_znorm(DIMS, dims, x);
            }
            nbytes = N * sizeof(complex float);
            *check = znorm;
            break;
        case 3:
            if (forloop) {
                #pragma omp parallel for schedule(static) reduction(+:fnorm)
                for (long i = 0; i < N; i++)
                    fnorm += cabs( x[i] );
            } else {
                fnorm = md_z1norm(DIMS, dims, x);
            }
            nbytes = N * sizeof(complex float);
            *check = fnorm;
            break;
        }

        toc = timestamp();
    }

	md_free(x);
	md_free(y);

    double sec = toc - tic;
    double bw = nbytes / sec;
	
	return bw;
}

static double bench_zscalar(long scale, complex float *check)
{
	return bench_norm(0, scale, check, 0);
}

static double bench_zscalarf(long scale, complex float *check)
{
	return bench_norm(0, scale, check, 1);
}

static double bench_zscalar_real(long scale, complex float *check)
{
	return bench_norm(1, scale, check, 0);
}

static double bench_zscalar_realf(long scale, complex float *check)
{
	return bench_norm(1, scale, check, 1);
}

static double bench_znorm(long scale, complex float *check)
{
	return bench_norm(2, scale, check, 0);
}

static double bench_znormf(long scale, complex float *check)
{
	return bench_norm(2, scale, check, 1);
}

static double bench_zl1norm(long scale, complex float *check)
{
	return bench_norm(3, scale, check, 0);
}

static double bench_zl1normf(long scale, complex float *check)
{
	return bench_norm(3, scale, check, 1);
}


static double bench_wavelet_thresh(int version, long scale)
{
	long dims[DIMS] = { 1, 256 * scale, 256 * scale, 1, 16, 1, 1, 1 };
	long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
	minsize[0] = MIN(dims[0], 16);
	minsize[1] = MIN(dims[1], 16);
	minsize[2] = MIN(dims[2], 16);

	const struct operator_p_s* p;

	switch (version) {
	case 2:
		p = prox_wavethresh_create(DIMS, dims, 7, minsize, 1.1, true, false);
		break;
	case 3:
		p = prox_wavelet3_thresh_create(DIMS, dims, 6, minsize, 1.1, true);
		break;
	default:
		assert(0);
	}

	complex float* x = md_alloc(DIMS, dims, CFL_SIZE);
	md_gaussian_rand(DIMS, dims, x);

    // warmup
	operator_p_apply(p, 0.98, DIMS, dims, x, DIMS, dims, x);

	double tic = timestamp();

	operator_p_apply(p, 0.98, DIMS, dims, x, DIMS, dims, x);

	double toc = timestamp();

	md_free(x);
	operator_p_free(p);

    long x_size = md_calc_size(DIMS, dims);
    long nbytes = sizeof(complex float) * (
        x_size + // read x
        x_size   // write x
    );
    double sec = toc - tic;
    double bw = nbytes / sec;

	return bw;
}

static double bench_wavelet2(long scale, complex float *check)
{
	return bench_wavelet_thresh(2, scale);
}

static double bench_wavelet3(long scale, complex float *check)
{
	return bench_wavelet_thresh(3, scale);
}


enum bench_indices { REPETITION_IND, SCALE_IND, THREADS_IND, TESTS_IND, BENCH_DIMS };

typedef double (*bench_fun)(long scale, complex float *check);

static void do_test(const long dims[BENCH_DIMS], complex float* out, long scale, bench_fun fun, const char* str)
{
	printf("%30.30s |", str);
	
	int N = dims[REPETITION_IND];
	double sum = 0.;
	double min = 1.E10;
	double max = 0.;

    complex float check = 0;

	for (int i = 0; i < N; i++) {

        num_rand_init(1337);
		double dt = fun(scale, &check) * 1e-9;
		sum += dt;
		min = MIN(dt, min);
		max = MAX(dt, max);

		printf(" %6.2f", (float)dt);
		fflush(stdout);

		assert(0 == REPETITION_IND);
		out[i] = dt;
	}

    float avg = sum / N;

    if (peak_bw != 0.0f) {
        float frac = avg / peak_bw * 100.0;
	    printf(" | Frac: %6.2f%% of %6.2f GB/s | Check %6.2f + %6.2fj \n", frac, peak_bw, creal(check), cimag(check));
    } else {
	    printf(" | Avg: %3.4f Max: %3.4f Min: %3.4f\n", avg, max, min); 
    }
}


const struct benchmark_s {

	bench_fun fun;
	const char* str;

} benchmarks[] = {
	{ bench_addf,		"add (for loop)" },
	{ bench_add,		"add (md_zaxpy)" },
	{ bench_add2f,		"add (for loop), contiguous" },
	{ bench_add2,		"add (md_zaxpy), contiguous" },
	{ bench_sumf,   	"sum (for loop)" },
	{ bench_sum,   		"sum (md_zaxpy)" },
	{ bench_sum2,   	"sum (md_zaxpy), contiguous" },
	{ bench_transpose,	"complex transpose (md_trans)" },
	{ bench_transposef,	"complex transpose (for loop)" },
	{ bench_resizem,   	"complex resize (md_resize)" },
	{ bench_resizef,   	"complex resize (for loop) " },
	{ bench_matrix_mult,	"complex matrix multiply" },
	{ bench_batch_matmul1,	"batch matrix multiply 1" },
	{ bench_batch_matmul2,	"batch matrix multiply 2" },
	{ bench_tall_matmul1,	"tall matrix multiply 1" },
	{ bench_tall_matmul2,	"tall matrix multiply 2" },
	{ bench_zscalarf,	"complex dot prod (for loop)  " },
	{ bench_zscalar,	"complex dot prod (md_zscalar)" },
	{ bench_zscalar_realf,	"real dot prod (for loop)" },
	{ bench_zscalar_real,	"real dot prod (md_zs_r)" },
	{ bench_znormf,		"l2 norm (for loop)" },
	{ bench_znorm,		"l2 norm (md_znorm)" },
	{ bench_zl1normf,	"l1 norm (for loop) " },
	{ bench_zl1norm,	"l1 norm (md_z1norm)" },
	{ bench_copy1,		"copy 1 (md_copy2)" },
	{ bench_copy1f,		"copy 1 (for loop)" },
	{ bench_copy2,		"copy 2 (md_copy2)" },
	{ bench_copy2f,		"copy 2 (for loop)" },
	{ bench_wavelet2,	"wavelet soft thresh" },
	{ bench_wavelet3,	"wavelet soft thresh" },
};


static const char usage_str[] = "[<output>]";
static const char help_str[] = "Performs a series of micro-benchmarks.";



int main_bench(int argc, char* argv[])
{
	bool threads = false;
	bool scaling = false;

	const struct opt_s opts[] = {

		OPT_SET('T', &threads, "varying number of threads"),
		OPT_SET('S', &scaling, "varying problem size"),
		OPT_FLOAT('b', &peak_bw, "bandwidth", "STREAM memory bandwidth in GB/sec."),
	};

	cmdline(&argc, argv, 0, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	long dims[BENCH_DIMS] = MD_INIT_ARRAY(BENCH_DIMS, 1);
	long strs[BENCH_DIMS];
	long pos[BENCH_DIMS] = { 0 };

	dims[REPETITION_IND] = 5;
	dims[THREADS_IND] = threads ? 8 : 1;
	dims[SCALE_IND] = scaling ? 5 : 1;
	dims[TESTS_IND] = sizeof(benchmarks) / sizeof(benchmarks[0]);

	md_calc_strides(BENCH_DIMS, strs, dims, CFL_SIZE);

	bool outp = (2 == argc);
	complex float* out = (outp ? create_cfl : anon_cfl)(outp ? argv[1] : "", BENCH_DIMS, dims);

	num_init();

    printf("                     Test Name |      Measured Memory Bandwidths    |    Fraction of Peak          |  Checksum for correctness\n");

	do {
		if (threads) {

			num_set_num_threads(pos[THREADS_IND] + 1);
			debug_printf(DP_INFO, "%02d threads. ", pos[THREADS_IND] + 1);
		}

		do_test(dims, &MD_ACCESS(BENCH_DIMS, strs, pos, out), pos[SCALE_IND] + 1,
			benchmarks[pos[TESTS_IND]].fun, benchmarks[pos[TESTS_IND]].str);

	} while (md_next(BENCH_DIMS, dims, ~MD_BIT(REPETITION_IND), pos));

	unmap_cfl(BENCH_DIMS, dims, out);

	exit(0);
}


