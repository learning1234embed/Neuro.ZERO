#include "adaptIQ.h"

static uint8_t find_iq(float num)
{
    int i;
    register int32_t inteager = (int32_t)num;

    if (num < 0) {
        inteager = ~inteager + 1;
    }

    inteager = (inteager <<  1);
    inteager = (inteager <<  1) | (inteager >> (32 - 1));

    for (i = 1; i <= 31; i++)
    {
        inteager = (inteager <<  1) | (inteager >> (32 - 1));
        if (inteager & 0x01)
        {
            break;
        }
    }

    if (i > 31)
        return 31;

    return i;
}

static _iq _IQ31mpy(register _iq iq31Arg1, register _iq iq31Arg2)
{
    register uint32_t ui32Result;
    register uint16_t ui16IntState;
    register uint16_t ui16MPYState;

    /* Disable interrupts and save multiplier mode. [optional] */
    ui16IntState = __get_interrupt_state();
    __disable_interrupt();
    ui16MPYState = MPY32CTL0;

    /* Set the multiplier to fractional mode. */
    MPY32CTL0 = MPYFRAC;

    /* Perform multiplication and save result. */
    MPYS32L = iq31Arg1;
    MPYS32H = iq31Arg1 >> 16;
    OP2L = iq31Arg2;
    OP2H = iq31Arg2 >> 16;
    __delay_cycles(5); //Delay for the result to be ready
    ui32Result = RES2;
    ui32Result |= (uint32_t) RES3 << 16;

    /* Restore multiplier mode and interrupts. [optional] */
    MPY32CTL0 = ui16MPYState;
    __set_interrupt_state(ui16IntState);

    return (_iq) ui32Result;
}

static uint8_t update_adaptIQmpy(adaptIQ x, adaptIQ y)
{
    register uint8_t s1 = x.iq + y.iq;
    register uint8_t s2;
    register uint8_t s;
    register uint8_t log2_x = 0, log2_y = 0;

    if (x.base < 0) {
        x.base = ~x.base + 1;
    }

    while (x.base) {
        x.base = x.base >> 1;
        log2_x++;
    }

    if (y.base < 0) {
        y.base = ~y.base + 1;
    }

    while (y.base) {
        y.base = y.base >> 1;
        log2_y++;
    }

    s2 = log2_x + log2_y;
    s = s1 > s2 ? s1 : s2;

    return x.iq + y.iq - s + 31;
}

adaptIQ adaptIQmpy(adaptIQ x, adaptIQ y)
{
    adaptIQ z;
    uint8_t original_iq = x.iq + y.iq - 31;
    uint8_t updated_iq = update_adaptIQmpy(x, y);

    z.iq = updated_iq;
    z.base = _IQ31mpy(x.base, y.base) << (updated_iq - original_iq);

    return z;
}

adaptIQ adaptIQaddSame(adaptIQ x, adaptIQ y)
{
    adaptIQ z;
    uint16_t status_reg;

    z.base = x.base + y.base;
    z.iq = x.iq;
    status_reg = __get_SR_register();

    if ((status_reg & 0x104) == 0x104) {
        z.base = (z.base >> 1) & ~(0x80000000);
        z.iq = z.iq - 1;
    } else if ((status_reg & 0x101) == 0x101) {
        z.base = (z.base >> 1) | 0x80000000;
        z.iq = z.iq - 1;
    }

    return z;
}

static uint8_t update_adaptIQadd(adaptIQ x)
{
    int8_t s1 = x.iq - 31;
    int8_t s2;
    int8_t s;
    uint8_t log2_x = 0;

    if (x.base < 0) {
        x.base = ~x.base + 1;
    }

    while (x.base) {
        x.base = x.base >> 1;
        log2_x++;
    }

    s2 = log2_x - 31;
    s = s1 > s2 ? s1 : s2;

    return x.iq - s;
}

adaptIQ adaptIQadd(adaptIQ x, adaptIQ y)
{
    adaptIQ x_aligned = x, y_aligned = y;
    adaptIQ z;

    // align IQ
    uint8_t large_iq = x.iq > y.iq ? x.iq : y.iq;
    uint8_t small_iq = x.iq < y.iq ? x.iq : y.iq;
    uint8_t diff_iq = large_iq - small_iq;

    if (small_iq == x.iq) {
        y_aligned.base = y.base >> diff_iq;
        y_aligned.iq = x.iq;
    } else {
        x_aligned.base = x.base >> diff_iq;
        x_aligned.iq = y.iq;
    }


    z = adaptIQaddSame(x_aligned, y_aligned);

    uint8_t original_iq = z.iq;
    uint8_t updated_iq = update_adaptIQadd(z);

    z.iq = updated_iq;
    z.base = z.base << (updated_iq - original_iq);

    return z;
}

float adaptIQtoF(adaptIQ iq)
{
    if (iq.iq == 31)
    {
        return (float) iq.base / ((_iq) 1 << iq.iq - 1) / 2;
    }
    else
    {
        return (float) iq.base / ((_iq) 1 << iq.iq);
    }
}

adaptIQ newAdaptIQ(float num)
{
    adaptIQ new;
    new.iq = find_iq(num);

    if (new.iq == 31) {
        new.base = _IQ31(num);
    } else {
        new.base = (_iq)((num) * ((_iq)1 << new.iq));
    }

    return new;
}

adaptIQ newAdaptIQgiven(_iq base, uint8_t iq)
{
    adaptIQ new;
    new.iq = iq;
    new.base = base;

    return new;
}

adaptIQ adaptIQconvert(adaptIQ x, uint8_t new_iq)
{
    adaptIQ z = x;

    if (new_iq >= z.iq)
    {
        z.base = (_iq) (z.base) << (new_iq - z.iq);
    }
    else
    {
        z.base = (_iq) (z.base) >> (z.iq - new_iq);
    }

    z.iq = new_iq;

    return z;
}
