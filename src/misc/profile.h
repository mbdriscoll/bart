/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Michael Driscoll <driscoll@cs.berkeley.edu>
 *
 * support for ad-hoc profiling
 */

#include "omp.h"
#include "stdio.h"

#define PUSH(name) printf("s %s %f\n", name, omp_get_wtime());
#define POP(name)  printf("e %s %f\n", name, omp_get_wtime());
