#pragma once
#define REAL double

REAL orient3d(REAL *pa, REAL *pb, REAL *pc, REAL *pd);

REAL insphere(REAL *pa, REAL *pb, REAL *pc, REAL *pd, REAL *pe);

REAL orient4d(REAL *pa, REAL *pb, REAL *pc, REAL *pd, REAL *pe,
	REAL ah, REAL bh, REAL ch, REAL dh, REAL eh);