/* Copyright 2014-2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014-2016	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016		Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"

#include "iter/iter.h"

#include "misc/misc.h"

#include "prox.h"

#define PTR_BASE(T, x, b) T* x = CONTAINER_OF(b, T, base)
#define FREE_BASE(T, b) free(CONTAINER_OF(b, T, base))


/** 
 * Proximal function of f is defined as
 * (prox_f)(z) = arg min_x 0.5 || z - x ||_2^2 + f(x)
 *
 * (prox_{mu f})(z) = arg min_x 0.5 || z - x ||_2^2 + mu f(x)
 */


/**
 * Data for computing prox_normaleq_fun: 
 * Proximal function for f(z) = 0.5 || y - A z ||_2^2.
 *
 * @param op operator that applies A^H A
 * @param cgconf conf file for conjugate gradient iter interface
 * @param adj A^H y
 * @param size size of z
 */
struct prox_normaleq_data {

	operator_data_t base;
	
	const struct linop_s* op;
	void* cgconf;
	float* adj;

	long size;
};

/**
 * Proximal function for f(z) = 0.5 || y - A z ||_2^2.
 * Solution is (A^H A + (1/mu) I)z = A^H y + (1/mu)(x_plus_u)
 *
 * @param prox_data should be of type prox_normaleq_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_normaleq_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	PTR_BASE(struct prox_normaleq_data, pdata, prox_data);

	if (0 == mu) {

		md_copy(1, MD_DIMS(pdata->size), z, x_plus_u, FL_SIZE);

	} else {

		float rho = 1. / mu;
		float* b = md_alloc_sameplace(1, MD_DIMS(pdata->size), FL_SIZE, x_plus_u);
		md_copy(1, MD_DIMS(pdata->size), b, pdata->adj, FL_SIZE);
		md_axpy(1, MD_DIMS(pdata->size), b, rho, x_plus_u);

		if (NULL == pdata->op->norm_inv) {

			struct iter_conjgrad_conf* cg_conf = pdata->cgconf;
			cg_conf->l2lambda = rho;
			iter_conjgrad(&cg_conf->base, pdata->op->normal, NULL, pdata->size, z, (float*)b, NULL, NULL, NULL);

		} else {

			linop_norm_inv_iter((struct linop_s*)pdata->op, rho, z, b);
		}

		md_free(b);
	}
}

static void prox_normaleq_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_normaleq_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_normaleq_del(const operator_data_t* _data)
{
	PTR_BASE(struct prox_normaleq_data, pdata, _data);

	free(pdata->cgconf);
	md_free(pdata->adj);
	free(pdata);
}

const struct operator_p_s* prox_normaleq_create(const struct linop_s* op, const complex float* y)
{
	PTR_ALLOC(struct prox_normaleq_data, pdata);
	PTR_ALLOC(struct iter_conjgrad_conf, cgconf);

	*cgconf = iter_conjgrad_defaults;
	cgconf->maxiter = 10;
	cgconf->l2lambda = 0;

	pdata->cgconf = PTR_PASS(cgconf);
	pdata->op = op;

	pdata->size = 2 * md_calc_size(linop_domain(op)->N, linop_domain(op)->dims);
	pdata->adj = md_alloc_sameplace(1, &(pdata->size), FL_SIZE, y);
	linop_adjoint_iter((struct linop_s*)op, pdata->adj, (const float*)y);

	return operator_p_create(linop_domain(op)->N, linop_domain(op)->dims, 
			linop_domain(op)->N, linop_domain(op)->dims, 
			&PTR_PASS(pdata)->base, prox_normaleq_apply, prox_normaleq_del, "prox_normaleq");
}


/**
 * Data for computing prox_leastsquares_fun: 
 * Proximal function for f(z) = lambda / 2 || y - z ||_2^2.
 *
 * @param y
 * @param lambda regularization
 * @param size size of z
 */
struct prox_leastsquares_data {

	operator_data_t base;
	
	const float* y;
	float lambda;

	long size;
};

/**
 * Proximal function for f(z) = lambda / 2 || y - z ||_2^2.
 * Solution is z =  (mu * lambda * y + x_plus_u) / (mu * lambda + 1)
 *
 * @param prox_data should be of type prox_leastsquares_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_leastsquares_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	PTR_BASE(struct prox_leastsquares_data, pdata, prox_data);

	md_copy(1, MD_DIMS(pdata->size), z, x_plus_u, FL_SIZE);

	if (0 != mu) {

		if (NULL != pdata->y)
			md_axpy(1, MD_DIMS(pdata->size), z, pdata->lambda * mu, pdata->y);

		md_smul(1, MD_DIMS(pdata->size), z, z, 1. / (mu * pdata->lambda + 1));
	}
}

static void prox_leastsquares_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_leastsquares_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_leastsquares_del(const operator_data_t* _data)
{
	FREE_BASE(struct prox_leastsquares_data, _data);
}

const struct operator_p_s* prox_leastsquares_create(unsigned int N, const long dims[N], float lambda, const complex float* y)
{
	PTR_ALLOC(struct prox_leastsquares_data, pdata);

	pdata->y = (const float*)y;
	pdata->lambda = lambda;
	pdata->size = md_calc_size(N, dims) * 2;

	return operator_p_create(N, dims, N, dims, &PTR_PASS(pdata)->base, prox_leastsquares_apply, prox_leastsquares_del, "prox_leastsquares");
}


/**
 * Data for computing prox_l2norm_fun:
 * Proximal function for f(z) = lambda || z ||_2.
 *
 * @param lambda regularization
 * @param size size of z
 */
struct prox_l2norm_data {

	operator_data_t base;
	
	float lambda;
	long size;
};

/**
 * Proximal function for f(z) = lambda  || z ||_2.
 * Solution is z =  ( 1 - lambda * mu / norm(z) )_+ * z,
 * i.e. block soft thresholding
 *
 * @param prox_data should be of type prox_l2norm_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_l2norm_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	PTR_BASE(struct prox_l2norm_data, pdata, prox_data);

	md_clear(1, MD_DIMS(pdata->size), z, FL_SIZE);

	double q1 = md_norm(1, MD_DIMS(pdata->size), x_plus_u);

	if (q1 != 0) {
		double q2 = 1 - pdata->lambda * mu / q1;

		if (q2 > 0.)
			md_smul(1, MD_DIMS(pdata->size), z, x_plus_u, q2);
	}
}

static void prox_l2norm_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_l2norm_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_l2norm_del(const operator_data_t* _data)
{
	FREE_BASE(struct prox_l2norm_data, _data);
}

const struct operator_p_s* prox_l2norm_create(unsigned int N, const long dims[N], float lambda)
{
	PTR_ALLOC(struct prox_l2norm_data, pdata);

	pdata->lambda = lambda;
	pdata->size = md_calc_size(N, dims) * 2;

	return operator_p_create(N, dims, N, dims, &PTR_PASS(pdata)->base, prox_l2norm_apply, prox_l2norm_del, "prox_l2norm");
}


/**
 * Data for computing prox_l2ball_fun: 
 * Proximal function for f(z) = Ind{ || y - z ||_2 < eps }
 *
 * @param center y
 * @param eps
 * @param size size of z
 */
struct prox_l2ball_data {

	operator_data_t base;

	const float* center;
	float eps;

	long size;
};

/**
 * Proximal function for f(z) = Ind{ || y - z ||_2 < eps }
 * 
 * @param prox_data should be of type prox_l2ball_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_l2ball_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	UNUSED(mu);
	PTR_BASE(struct prox_l2ball_data, pdata, prox_data);

	if (NULL != pdata->center)
		md_sub(1, MD_DIMS(pdata->size), z, x_plus_u, pdata->center);
	else
		md_copy(1, MD_DIMS(pdata->size), z, x_plus_u, FL_SIZE);

	float q1 = md_norm(1, MD_DIMS(pdata->size), z);

	if (q1 > pdata->eps)
		md_smul(1, MD_DIMS(pdata->size), z, z, pdata->eps / q1);

	if (NULL != pdata->center)
		md_add(1, MD_DIMS(pdata->size), z, z, pdata->center);
}

static void prox_l2ball_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_l2ball_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_l2ball_del(const operator_data_t* _data)
{
	FREE_BASE(struct prox_l2ball_data, _data);
}

const struct operator_p_s* prox_l2ball_create(unsigned int N, const long dims[N], float eps, const complex float* center)
{
	PTR_ALLOC(struct prox_l2ball_data, pdata);

	pdata->center = (const float*)center;
	pdata->eps = eps;
	pdata->size = md_calc_size(N, dims) * 2;

	return operator_p_create(N, dims, N, dims, &PTR_PASS(pdata)->base, prox_l2ball_apply, prox_l2ball_del, "prox_l2ball");
}




#if 0
/**
 * Data for computing prox_thresh_fun: 
 * Proximal function for f(z) = lambda || z ||_1
 *
 * @param thresh function to apply SoftThresh
 * @param data data used by thresh function
 * @param lambda regularization
 */
struct prox_thresh_data {

	void (*thresh)(void* _data, float lambda, float* _dst, const float* _src);
	void* data;
	float lambda;
};

/**
 * Proximal function for f(z) = lambda || z ||_1
 * Solution is z = SoftThresh(x_plus_u, lambda * mu)
 * 
 * @param prox_data should be of type prox_thresh_data
 */
void prox_thresh_fun(void* prox_data, float mu, float* z, const float* x_plus_u)
{
	struct prox_thresh_data* pdata = (struct prox_thresh_data*)prox_data;
	pdata->thresh(pdata->data, pdata->lambda * mu, z, x_plus_u);
}

static void prox_thresh_apply(const void* _data, float mu, complex float* dst, const complex float* src)
{
	prox_thresh_fun((void*)_data, mu, (float*)dst, (const float*)src);
}

static void prox_thresh_del(const void* _data)
{
	free((void*)_data);
}

const struct operator_p_s* prox_thresh_create(unsigned int N, const long dims[N], float lambda,
		void (*thresh)(void* _data, float lambda, float* _dst, const float* _src),
		void* data)
{
	PTR_ALLOC(struct prox_thresh_data, pdata);

	pdata->thresh = thresh;
	pdata->lambda = lambda;
	pdata->data = data;

	return operator_p_create(N, dims, dims, PTR_PASS(pdata), prox_thresh_apply, prox_thresh_del);
}
#endif


/**
 * Data for computing prox_zero_fun:
 * Proximal function for f(z) = 0
 *
 * @param size size of z
 */
struct prox_zero_data {

	operator_data_t base;

	long size;
};

/**
 * Proximal function for f(z) = 0
 * Solution is z = x_plus_u
 * 
 * @param prox_data should be of type prox_zero_data
 * @param mu proximal penalty
 * @param z output
 * @param x_plus_u input
 */
static void prox_zero_fun(const operator_data_t* prox_data, float mu, float* z, const float* x_plus_u)
{
	UNUSED(mu);
	PTR_BASE(struct prox_zero_data, pdata, prox_data);
	md_copy(1, MD_DIMS(pdata->size), z, x_plus_u, FL_SIZE);
}

static void prox_zero_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_zero_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_zero_del(const operator_data_t* _data)
{
	FREE_BASE(struct prox_zero_data, _data);
}

const struct operator_p_s* prox_zero_create(unsigned int N, const long dims[N])
{
	PTR_ALLOC(struct prox_zero_data, pdata);

	pdata->size = md_calc_size(N, dims) * 2;

	return operator_p_create(N, dims, N, dims, &PTR_PASS(pdata)->base, prox_zero_apply, prox_zero_del, "prox_zero");
}




/**
 * Data for computing prox_lineq_fun: 
 * Proximal function for f(z) = 1{ A z = y }
 * Assumes AA^T = I
 * Solution is z = x - A^T A x + A^T y
 *
 * @param op linop A
 * @param adj A^H y
 * @param tmp tmp
 */
struct prox_lineq_data {

	operator_data_t base;
	
	const struct linop_s* op;
	complex float* adj;
	complex float* tmp;
};

static void prox_lineq_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	UNUSED(mu);
	PTR_BASE(struct prox_lineq_data, pdata, _data);

	const struct linop_s* op = pdata->op;
	linop_normal(op, linop_domain(op)->N, linop_domain(op)->dims, pdata->tmp, src);

	md_zsub(linop_domain(op)->N, linop_domain(op)->dims, dst, src, pdata->tmp);
	md_zadd(linop_domain(op)->N, linop_domain(op)->dims, dst, dst, pdata->adj);
}

static void prox_lineq_del(const operator_data_t* _data)
{
	PTR_BASE(struct prox_lineq_data, pdata, _data);

	md_free(pdata->adj);
	md_free(pdata->tmp);
	free(pdata);
}

const struct operator_p_s* prox_lineq_create(const struct linop_s* op, const complex float* y)
{
	PTR_ALLOC(struct prox_lineq_data, pdata);

	unsigned int N = linop_domain(op)->N;
	const long* dims = linop_domain(op)->dims;

	pdata->op = op;

	pdata->adj = md_alloc_sameplace(N, dims, CFL_SIZE, y);
	linop_adjoint(op, N, dims, pdata->adj, N, linop_codomain(op)->dims, y);

	pdata->tmp = md_alloc_sameplace(N, dims, CFL_SIZE, y);

	return operator_p_create(N, dims, N, dims, &PTR_PASS(pdata)->base, prox_lineq_apply, prox_lineq_del, "prox_lineq");
}


/**
 * Data for computing prox_ineq_fun: 
 * Proximal function for f(z) = 1{ z <= b }
 *  and f(z) = 1{ z >= b }
 *
 * @param b b
 * @param size size of z
 */
struct prox_ineq_data {

	operator_data_t base;
	
	const float* b;
	long size;
	bool positive;
};

static void prox_ineq_fun(const operator_data_t* _data, float mu, float* dst, const float* src)
{
	UNUSED(mu);
	PTR_BASE(struct prox_ineq_data, pdata, _data);

	if (NULL == pdata->b)
		(pdata->positive ? md_smax : md_smin)(1, MD_DIMS(pdata->size), dst, src, 0.);
	else
		(pdata->positive ? md_max : md_min)(1, MD_DIMS(pdata->size), dst, src, pdata->b);
}

static void prox_ineq_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	prox_ineq_fun(_data, mu, (float*)dst, (const float*)src);
}

static void prox_ineq_del(const operator_data_t* _data)
{
	FREE_BASE(struct prox_ineq_data, _data);
}

static const struct operator_p_s* prox_ineq_create(unsigned int N, const long dims[N], const complex float* b, bool positive)
{
	PTR_ALLOC(struct prox_ineq_data, pdata);

	pdata->size = md_calc_size(N, dims) * 2;
	pdata->b = (const float*)b;
	pdata->positive = positive;

	return operator_p_create(N, dims, N, dims, &PTR_PASS(pdata)->base, prox_ineq_apply, prox_ineq_del, "prox_ineq");
}


/*
 * Proximal function for less than or equal to:
 * f(z) = 1{z <= b}
 */
const struct operator_p_s* prox_lesseq_create(unsigned int N, const long dims[N], const complex float* b)
{
	return prox_ineq_create(N, dims, b, false);
}

/*
 * Proximal function for greater than or equal to:
 * f(z) = 1{z >= b}
 */
const struct operator_p_s* prox_greq_create(unsigned int N, const long dims[N], const complex float* b)
{
	return prox_ineq_create(N, dims, b, true);
}

struct prox_rvc_data {

	operator_data_t base;

	long size;
};

static void prox_rvc_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	UNUSED(mu);
	PTR_BASE(struct prox_rvc_data, pdata, _data);

	md_zreal(1, MD_DIMS(pdata->size), dst, src);
}

static void prox_rvc_del(const operator_data_t* _data)
{
	FREE_BASE(struct prox_rvc_data, _data);
}

/*
 * Proximal function for real-value constraint
 */
const struct operator_p_s* prox_rvc_create(unsigned int N, const long dims[N])
{
	PTR_ALLOC(struct prox_rvc_data, pdata);

	pdata->size = md_calc_size(N, dims);
	return operator_p_create(N, dims, N, dims, &PTR_PASS(pdata)->base, prox_rvc_apply, prox_rvc_del, "prox_rvc");
}
