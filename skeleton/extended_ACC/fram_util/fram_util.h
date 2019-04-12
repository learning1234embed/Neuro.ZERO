#ifndef FRAM_UTIL_H_
#define FRAM_UTIL_H_

#include <msp430.h>
#include <QmathLib.h>
#include <IQmathLib.h>
#include "driverlib.h"

long lr(long *addr);
void lw(long *addr, long i);
int ir(int *addr);
void iw(int *addr, int f);
float fr(float *addr);
void fw(float *addr, float f);
_q qr(_q *addr);
void qw(_q *addr, _q q);
_iq30 iqr(_iq30 *addr);
void iqw(_iq30 *addr, _iq30 iq);
char cr(char *addr);
void cw(char *addr, char f);

#endif /* FRAM_UTIL_H_ */
