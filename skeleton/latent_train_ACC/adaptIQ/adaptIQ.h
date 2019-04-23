#ifndef ADAPTIQ_H_
#define ADAPTIQ_H_

#include <msp430.h>
#include <driverlib.h>
#include <DSPLib.h>
#include <IQmathLib.h>

typedef struct {
    _iq base;
    uint8_t iq;
} adaptIQ;

adaptIQ newAdaptIQ(float num);
adaptIQ newAdaptIQgiven(_iq base, uint8_t iq);
float adaptIQtoF(adaptIQ iq);

adaptIQ adaptIQmpy(adaptIQ x, adaptIQ y);
adaptIQ adaptIQadd(adaptIQ x, adaptIQ y);
adaptIQ adaptIQaddSame(adaptIQ x, adaptIQ y);
adaptIQ adaptIQconvert(adaptIQ x, uint8_t new_iq);

#endif /* ADAPTIQ_H_ */
