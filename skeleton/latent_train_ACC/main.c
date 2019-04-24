#include <msp430.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <driverlib.h>
#include <DSPLib.h>
#include <IQmathLib.h>
#include "fram_util/fram_util.h"
#include "adaptIQ/adaptIQ.h"

#define MAX_NUM_IQ31_LEA        (946)
#define NUM_INPUT               (2)
#define NUM_HIDDEN              (64)
#define NUM_OUTPUT              (2)
#define NUM_HIDDEN_LAYER        (1)
#define NUM_TOTAL_LAYER         (NUM_HIDDEN_LAYER+2)

/* neurons */
#pragma PERSISTENT(input)
_iq31 input[NUM_INPUT] = { 0 };

#pragma PERSISTENT(hidden)
_iq31 hidden[NUM_HIDDEN_LAYER][NUM_HIDDEN] = { 0, };

#pragma PERSISTENT(output)
_iq31 output[NUM_OUTPUT] = { 0 };

#pragma PERSISTENT(ground_truth_output)
uint16_t ground_truth_output[NUM_OUTPUT] = { 0 };

/* weight */
#pragma PERSISTENT(weight_input)
_iq31 weight_input[NUM_HIDDEN][NUM_INPUT+1] = { 0, };

#pragma PERSISTENT(weight_hidden)
_iq31 weight_hidden[NUM_HIDDEN_LAYER-1][NUM_HIDDEN][NUM_HIDDEN+1] = { 0, };

#pragma PERSISTENT(weight_output)
_iq31 weight_output[NUM_OUTPUT][NUM_HIDDEN+1] = { 0, };

/* weight momentum */
#pragma PERSISTENT(weight_input_m)
_iq31 weight_input_m[NUM_HIDDEN][NUM_INPUT+1] = { 0, };

#pragma PERSISTENT(weight_hidden_m)
_iq31 weight_hidden_m[NUM_HIDDEN_LAYER-1][NUM_HIDDEN][NUM_HIDDEN+1] = { 0, };

#pragma PERSISTENT(weight_output_m)
_iq31 weight_output_m[NUM_OUTPUT][NUM_HIDDEN+1] = { 0, };

#pragma PERSISTENT(delta_weight_sum_a)
adaptIQ delta_weight_sum_a[NUM_TOTAL_LAYER - 2][NUM_HIDDEN] = { 0, };

#if 0
#define NUM_TRAIN_DATA         2 // mini batch size
#pragma PERSISTENT(train_input)
_iq31 train_input[NUM_TRAIN_DATA][NUM_INPUT]
                             = { { _IQ(0.9), _IQ(0.1), _IQ(0.1), _IQ(0.8), _IQ(0.8), _IQ(0.8), _IQ(0.8), _IQ(0.8),
                                   _IQ(0.9), _IQ(0.1), _IQ(0.1), _IQ(0.8), _IQ(0.8), _IQ(0.8), _IQ(0.8), _IQ(0.8), },
                                 { _IQ(0.1), _IQ(0.9), _IQ(0.3), _IQ(0.1), _IQ(0.1), _IQ(0.1), _IQ(0.2), _IQ(0.4),
                                   _IQ(0.1), _IQ(0.9), _IQ(0.3), _IQ(0.1), _IQ(0.1), _IQ(0.1), _IQ(0.2), _IQ(0.4), }, };

#pragma PERSISTENT(train_output)
uint16_t train_output[NUM_TRAIN_DATA][NUM_OUTPUT] = { { 1, 0, }, { 0, 1, }, };
#endif
#if 1
#define NUM_TRAIN_DATA         2
#pragma PERSISTENT(train_input)
_iq31 train_input[NUM_TRAIN_DATA][NUM_INPUT]
                             = { { _IQ(0.9), _IQ(0.1), },
                                 { _IQ(0.1), _IQ(0.9), }, };

#pragma PERSISTENT(train_output)
uint16_t train_output[NUM_TRAIN_DATA][NUM_OUTPUT] = { { 1, 0, }, { 0, 1, }, };
#endif

/* hyper parameters */
float eta = 0.01; // learning rate
float alpha = 0.9; // momentum alpha
float lambda = 0.0000001; // weight regularization
float skipout = 0.0; // skipout_rate
float avg_skipout = 0;
uint32_t count = 0;

adaptIQ eta_a;
adaptIQ alpha_a;
adaptIQ lambda_a;

static msp_status status;

#define VECTOR_SIZE             (256)

typedef union
{
    _iq31 raw[MAX_NUM_IQ31_LEA];
    struct FftData_s
    {
        _q15 input[VECTOR_SIZE * 2];
        _q15 samples[VECTOR_SIZE * 2];
        _q15 sourceA[VECTOR_SIZE];
        _q15 sourceB[VECTOR_SIZE];
        _q15 dest[VECTOR_SIZE];
        _iq31 result;
    } fftDataParam;
} LeaMem;

#pragma DATA_SECTION(leaMem, ".leaRAM")
LeaMem leaMem;

static void copyData(const void *src, void *dst, uint16_t length)
{
    uint16_t i;
    uint16_t *srcPtr;
    uint16_t *dstPtr;

    // Set src and dst pointers
    srcPtr = (uint16_t *) src;
    dstPtr = (uint16_t *) dst;

    for (i = 0; i < length / 2; i++)
    {
        *dstPtr++ = *srcPtr++;
    }
}

static void copyDataFF(const void *src, void *dst, uint16_t length)
{
    uint16_t i;
    uint16_t *srcPtr;
    uint16_t *dstPtr;

    // Set src and dst pointers
    srcPtr = (uint16_t *) src;
    dstPtr = (uint16_t *) dst;

    for (i = 0; i < length / 2; i++)
    {
        iw((int *)&dstPtr[i], (uint16_t)ir((int *)&srcPtr[i]));
    }
}

static float xavier_init(uint16_t num_input, uint16_t num_output)
{
    int r = rand() - RAND_MAX / 2;
    float f = (float) r / RAND_MAX * sqrt((float) 1 / (num_input + num_output));
    return f;
}

static void shuffle(uint16_t *array, uint16_t length)
{
    uint16_t i;
    for (i = 0; i < length - 1; i++)
    {
        uint16_t j = i + rand() / (RAND_MAX / (length - i) + 1);
        uint16_t temp = array[j];
        array[j] = array[i];
        array[i] = temp;
    }
}

static _iq31 vector_dot_product_iq31(_iq31 *vector1, _iq31 *vector2,
                                     uint16_t length1, uint16_t length2)
{
    uint16_t i;
    uint16_t iteration;
    uint16_t max_len = MAX_NUM_IQ31_LEA / 2 - 1;
    uint16_t residual;
    _iq31 *result = &leaMem.raw[MAX_NUM_IQ31_LEA - 1];
    _iq31 sum = 0;

    if (length1 != length2)
    {
        P1OUT |= BIT0;
        return 0;
    }

    msp_mac_iq31_params param;
    iteration = length1 / max_len;

    for (i = 0; i < iteration; i++)
    {
        copyData(&vector1[i*max_len], &leaMem.raw[0], max_len*4);
        copyData(&vector2[i*max_len], &leaMem.raw[max_len], max_len*4);

        param.length = max_len;
        status = msp_mac_iq31(&param, &leaMem.raw[0], &leaMem.raw[max_len], result);
        if (status != MSP_SUCCESS)
            P1OUT |= BIT0;

        sum += *result;
    }

    residual = length1 % max_len;

    if (residual)
    {
        copyData(&vector1[i*max_len], &leaMem.raw[0], residual*4);
        copyData(&vector2[i*max_len], &leaMem.raw[max_len], residual*4);

        param.length = residual;
        status = msp_mac_iq31(&param, &leaMem.raw[0], &leaMem.raw[max_len], result);
        if (status != MSP_SUCCESS)
            P1OUT |= BIT0;

        sum += *result;
    }

    return sum;
}

static float cross_entropy(uint16_t *label, float *output, uint16_t length)
{
    float loss = 0;
    uint16_t i;

    for (i = 0; i < length; i++)
    {
        if (label[i] != 0 || output[i] != 0) {
            loss -= ((float) label[i] * log(output[i]));
        }
    }

    return loss;
}

static _iq31 leakyReLU_iq31(_iq31 x)
{
    if (x <= _IQ(0)) {
        return _IQmpy(_IQ(0.3),x);
    } else {
        return x;
    }
}

static _iq31 leakyReLU_diff_iq31(_iq31 x)
{
    if (x < _IQ(0)) {
        return _IQ(0.3);
    } else if (x == _IQ(0)) {
        return _IQ(0.5);
    } else {
        return _IQ(1);
    }
}

static void softmax(float *x, float *y, uint16_t length)
{
    /* softmax function: return an array of softmax (float *y) */
    int i;
    float max;
    float exp_sum = 0;

    for (i = 0; i < length; i++) {
        if (i == 0) {
            max = x[i];
        }

        if (x[i] > max) {
            max = x[i];
        }
    }

    for (i = 0; i < length; i++) {
        y[i] = x[i] - max;
        y[i] = exp(y[i]);
        exp_sum += y[i];
    }

    for (i = 0; i < length; i++) {
        y[i] /= exp_sum;
    }
}

static float objective(uint16_t *label, float *output, uint16_t length)
{
    /* cross entropy */
    return cross_entropy(label, output, length);
}

static _iq31 inner_activate_iq31(_iq31 neuron)
{
    /* ReLU */
    return leakyReLU_iq31(neuron);
}

static _iq31 inner_activate_diff_iq31(_iq31 neuron)
{
    /* ReLU */
    return leakyReLU_diff_iq31(neuron);
}

static void final_activate(float *neuron, float *activated_neuron, uint16_t length)
{
    /* softmax */
    softmax(neuron, activated_neuron, length);
}

static void init_weight(void)
{
    uint16_t i, j;
    uint16_t hidden_layer;
    float random_weight;

    printf("weight_input\n");
    for (i = 0; i < NUM_HIDDEN; i++)
    {
        for (j = 0; j < NUM_INPUT+1; j++)
        {
            random_weight = xavier_init(NUM_INPUT, NUM_HIDDEN);
            iqw(&weight_input[i][j], _IQ(random_weight));
        }
    }

    for (hidden_layer = 0; hidden_layer < NUM_HIDDEN_LAYER - 1; hidden_layer++)
    {
        printf("weight_hidden[%d]\n", hidden_layer);
        for (i = 0; i < NUM_HIDDEN; i++)
        {
            for (j = 0; j < NUM_HIDDEN+1; j++)
            {
                random_weight = xavier_init(NUM_HIDDEN, NUM_HIDDEN);
                iqw(&weight_hidden[hidden_layer][i][j], _IQ(random_weight));
            }
        }
    }

    printf("weight_output\n");
    for (i = 0; i < NUM_OUTPUT; i++)
    {
        for (j = 0; j < NUM_HIDDEN+1; j++)
        {
            random_weight = xavier_init(NUM_HIDDEN, NUM_OUTPUT);
            iqw(&weight_output[i][j], _IQ(random_weight));
        }
    }
}

static void init_test_train_input(void)
{
    int i, j;
    for (i = 0; i < NUM_TRAIN_DATA; i++)
    {
        for (j = 17; j < NUM_INPUT; j++)
        {

            float rando = (float)rand() / (float) RAND_MAX;
            iqw(&train_input[i][j], _IQ(rando));
        }
    }
}

static uint16_t feed_forward()
{
    uint16_t i, j;
    _iq31 sum, neuron, bias;

    // input -> hidden
    for (i = 0; i < NUM_HIDDEN; i++)
    {
        bias = iqr(&weight_input[i][NUM_INPUT]); // GLOBAL_IQ
        sum = vector_dot_product_iq31(input, &weight_input[i], NUM_INPUT, NUM_INPUT); // IQ30 * GLOBAL_IQ = GLOBAL_IQ-1
        sum = sum << 7;
        sum = _IQ30mpy(sum, _IQ30((float)1.0 - avg_skipout));
        sum += bias; // GLOBAL_IQ - 1
        neuron = inner_activate_iq31(sum); // IQ30
        iqw(&hidden[0][i], neuron); // IQ30
    }

    // hidden -> hidden
    for (i = 0; i < NUM_HIDDEN_LAYER-1; i++)
    {
        for (j = 0; j < NUM_HIDDEN; j++)
        {
            bias = iqr(&weight_hidden[i][j][NUM_HIDDEN]); // GLOBAL_IQ
            sum = vector_dot_product_iq31(&hidden[i], &weight_hidden[i][j], NUM_HIDDEN, NUM_HIDDEN); // IQ30 * GLOBAL_IQ = GLOBAL_IQ-1
            sum = sum << 7;
            sum = _IQ30mpy(sum, _IQ30((float)1.0 - avg_skipout));
            sum += bias; // GLOBAL_IQ - 1
            neuron = inner_activate_iq31(sum); // IQ30
            iqw(&hidden[i + 1][j], neuron); // IQ30
        }
    }

    // hidden -> output
    for (i = 0; i < NUM_OUTPUT; i++)
    {
        bias = iqr(&weight_output[i][NUM_HIDDEN]); // GLOBAL_IQ
        sum = vector_dot_product_iq31(&hidden[NUM_HIDDEN_LAYER-1], &weight_output[i], NUM_HIDDEN, NUM_HIDDEN); // IQ30 * GLOBAL_IQ = GLOBAL_IQ-1
        sum = sum << 7;
        sum = _IQ30mpy(sum, _IQ30((float)1.0 - avg_skipout));
        sum += bias; // GLOBAL_IQ - 1
        iqw(&output[i], sum); // GLOBAL_IQ - 1
    }

    // softmax
    float output_f[NUM_OUTPUT];
    float final_output_f[NUM_OUTPUT];

    for (i = 0; i < NUM_OUTPUT; i++) {
        output_f[i] = _IQtoF(iqr(&output[i])); // GLOBAL_IQ * IQ30 = GLOBAL_IQ - 1
    }

    final_activate(output_f, final_output_f, NUM_OUTPUT);

    for (i = 0; i < NUM_OUTPUT; i++) {
        iqw(&output[i],  _IQ(final_output_f[i]));
    }

    // get the highest output
    copyData(output, leaMem.raw, sizeof(_iq31) * NUM_OUTPUT);

    msp_max_iq31_params max_param;
    max_param.length = NUM_OUTPUT;
    _iq31 max_output;
    uint16_t max_output_index;

    status = msp_max_iq31(&max_param, leaMem.raw, &max_output, &max_output_index);
    if (status != MSP_SUCCESS)
        P1OUT |= BIT0;

    return max_output_index;
}

static adaptIQ update_momentum(adaptIQ weight, adaptIQ gradient, adaptIQ momentum)
{
    adaptIQ first = adaptIQmpy(eta_a, gradient);
    adaptIQ second = adaptIQmpy(alpha_a, momentum);
    adaptIQ third = adaptIQmpy(eta_a, lambda_a);
    third = adaptIQmpy(third, weight);
    adaptIQ new_momentum = adaptIQadd(first, second);
    new_momentum = adaptIQadd(new_momentum, third);

    return new_momentum;
}

static void update_weight(_iq31 *weight_address, _iq31 *momenntum_address, adaptIQ gradient_a)
{
    adaptIQ weight_a, momentum_a;

    weight_a = newAdaptIQgiven(iqr(weight_address), GLOBAL_IQ);
    momentum_a = newAdaptIQgiven(iqr(momenntum_address), GLOBAL_IQ);
    momentum_a = update_momentum(weight_a, gradient_a, momentum_a);
    weight_a = adaptIQadd(weight_a, momentum_a);

    weight_a = adaptIQconvert(weight_a, GLOBAL_IQ);
    iqw(weight_address, weight_a.base);

    momentum_a = adaptIQconvert(momentum_a, GLOBAL_IQ);
    iqw(momenntum_address, momentum_a.base);
}

static void update_output_weight(uint16_t i, uint16_t j)
{
    adaptIQ delta_a;
    adaptIQ neuron_a, gradient_a, input_neuron_a, ground_truth_a, weight_a;
    adaptIQ sum;

    if (j == 0) {
        lw(&delta_weight_sum_a[NUM_TOTAL_LAYER - 3][i].base, 0);
        cw(&delta_weight_sum_a[NUM_TOTAL_LAYER - 3][i].iq, GLOBAL_IQ);
    }

    input_neuron_a = newAdaptIQgiven(iqr(&hidden[NUM_HIDDEN_LAYER - 1][i]), GLOBAL_IQ);
    ground_truth_a = newAdaptIQ(-(float)ir((int *)&ground_truth_output[j]));
    neuron_a = newAdaptIQgiven(iqr(&output[j]), GLOBAL_IQ);
    delta_a = adaptIQadd(neuron_a, ground_truth_a); // softmax_cross_entropy_diff
    gradient_a = adaptIQmpy(input_neuron_a, delta_a);
    update_weight(&weight_output[j][i], &weight_output_m[j][i], gradient_a);

    if (i == 0)
    {
        update_weight(&weight_output[j][NUM_HIDDEN], &weight_output_m[j][NUM_HIDDEN], delta_a);
    }

    weight_a = newAdaptIQgiven(iqr(&weight_output[j][i]), GLOBAL_IQ);
    sum = adaptIQadd(delta_weight_sum_a[NUM_TOTAL_LAYER - 3][i], adaptIQmpy(delta_a, weight_a));
    lw(&delta_weight_sum_a[NUM_TOTAL_LAYER - 3][i].base, sum.base);
    cw(&delta_weight_sum_a[NUM_TOTAL_LAYER - 3][i].iq, sum.iq);
}

static void update_hidden_weight(uint16_t hidden_layer, uint16_t i, uint16_t j)
{
    adaptIQ delta_a;
    adaptIQ weight_a, gradient_a, input_neuron_a;
    adaptIQ sum;

    input_neuron_a = newAdaptIQgiven(iqr(&hidden[hidden_layer-1][i]), GLOBAL_IQ);
    adaptIQ neuron_a_2 = newAdaptIQgiven(-iqr(&hidden[hidden_layer][j]), GLOBAL_IQ);
    adaptIQ inner_diff_a = newAdaptIQgiven(inner_activate_diff_iq31(iqr(&hidden[hidden_layer][j])), GLOBAL_IQ);

    sum.base = lr(&delta_weight_sum_a[hidden_layer][j].base);
    sum.iq = cr(&delta_weight_sum_a[hidden_layer][j].iq);
    delta_a = adaptIQmpy(sum, inner_diff_a);
    gradient_a = adaptIQmpy(input_neuron_a, delta_a);
    update_weight(&weight_hidden[hidden_layer-1][j][i], &weight_hidden_m[hidden_layer-1][j][i], gradient_a);

    if (i == 0)
    {
        update_weight(&weight_hidden[hidden_layer-1][j][NUM_HIDDEN], &weight_hidden_m[hidden_layer-1][j][NUM_HIDDEN], delta_a);
    }

    weight_a = newAdaptIQgiven(iqr(&weight_hidden[hidden_layer-1][j][i]), GLOBAL_IQ);
    sum = adaptIQadd(delta_weight_sum_a[hidden_layer-1][i], adaptIQmpy(delta_a, weight_a));
    lw(&delta_weight_sum_a[hidden_layer-1][i].base, sum.base);
    cw(&delta_weight_sum_a[hidden_layer-1][i].iq, sum.iq);
}

static void update_input_weight(uint16_t i, uint16_t j)
{
    adaptIQ delta_a;
    adaptIQ input_neuron_a, gradient_a;
    adaptIQ sum;

    input_neuron_a = newAdaptIQgiven(iqr(&input[i]), GLOBAL_IQ);
    adaptIQ inner_diff_a = newAdaptIQgiven(inner_activate_diff_iq31(iqr(&hidden[0][j])), GLOBAL_IQ);

    sum.base = lr(&delta_weight_sum_a[0][j].base);
    sum.iq = cr(&delta_weight_sum_a[0][j].iq);
    delta_a = adaptIQmpy(sum, inner_diff_a);
    gradient_a = adaptIQmpy(input_neuron_a, delta_a);
    update_weight(&weight_input[j][i], &weight_input_m[j][i], gradient_a);

    if (i == 0)
    {
        update_weight(&weight_input[j][NUM_INPUT], &weight_input_m[j][NUM_INPUT], delta_a);
    }
}

int drop()
{
    float r = rand();

    if (r < (float)RAND_MAX*skipout) {
        return 1;
    }

    return 0;
}

static void back_propagate(void)
{
    uint16_t i, j;

    /* output layer -> last hidden layer */
    for (i = 0; i < NUM_HIDDEN; i++)
    {
        for (j = 0; j < NUM_OUTPUT; j++)
        {
            if (drop()) {
                continue;
            }
            update_output_weight(i, j);
        }
    }

#if 1
    /* between hidden layers */
    uint16_t hidden_layer;
    for (hidden_layer = NUM_HIDDEN_LAYER - 1; hidden_layer >= 1; hidden_layer--)
    {
        for (i = 0; i < NUM_HIDDEN; i++)
        {
            for (j = 0; j < NUM_HIDDEN; j++)
            {
                if (drop()) {
                    continue;
                }
                update_hidden_weight(hidden_layer, i, j);
            }
        }
    }
#endif

    /* first hidden layer -> input layer */
    for (i = 0; i < NUM_INPUT; i++)
    {
        //input_neuron_a = newAdaptIQgiven(iqr(&input[i]), 30);
        for (j = 0; j < NUM_HIDDEN; j++)
        {
            if (drop()) {
                continue;
            }
            update_input_weight(i, j);
        }
    }
}

float random_skipout()
{
    return (float)rand() / (float)RAND_MAX;
}

void init_train(void)
{
    eta_a = newAdaptIQ(-eta);
    alpha_a = newAdaptIQ(alpha);
    lambda_a = newAdaptIQ(lambda);
#if 1
    init_weight();
    init_test_train_input();
#endif
}

void train(void)
{
    uint16_t i, j;
    uint16_t iteration;
    uint16_t order;
    uint16_t train_order[NUM_TRAIN_DATA];
    float loss;
    float total_loss = 0;

    // fetch train_data, train_output from an external storage ...

    for (i = 0; i < NUM_TRAIN_DATA; i++) {
        train_order[i] = i;
    }

    for (iteration = 0; iteration < 10000; iteration++) {
        printf("\niteration %05d\n", iteration);
        total_loss = 0;
        //shuffle(train_order, NUM_TRAIN_DATA);
        for (i = 0; i < NUM_TRAIN_DATA; i++) {
            //order = train_order[i];
            order = i;
            printf("[%d] ", order);

            copyDataFF(&train_input[order], &input, NUM_INPUT*4);
            copyDataFF(&train_output[order], ground_truth_output, NUM_OUTPUT*2);

            feed_forward();

            for (j = 0; j < NUM_INPUT; j++) {
                printf("%1.5f ", _IQtoF(iqr(&input[j])));
            }
            printf(": ");

            for (j = 0; j < NUM_OUTPUT; j++) {
                printf("%1.5f ", _IQtoF(iqr(&output[j])));
            }
            printf(": ");

            for (j = 0; j < NUM_OUTPUT; j++) {
                printf("%1.5f ", (float)ir(&ground_truth_output[j]));
            }

            float output_f[NUM_OUTPUT];
            for (j = 0; j < NUM_OUTPUT; j++) {
                output_f[j] = _IQtoF(iqr(&output[j]));
            }

            loss = objective(ground_truth_output, output_f, NUM_OUTPUT);
            printf(":loss = %.9f\n", loss);

            total_loss += loss;
        }

        printf("total_loss = %.9f\n\n", total_loss);

        if (total_loss < 0.001) {
            break;
        }

        shuffle(train_order, NUM_TRAIN_DATA);
        for (i = 0; i < NUM_TRAIN_DATA; i++) {
            order = train_order[i];
            //order = i;
            copyData(&train_input[order], &input, NUM_INPUT*4);
            copyData(&train_output[order], ground_truth_output, NUM_OUTPUT*2);

            feed_forward();
            skipout = random_skipout();
            printf("back_propagate[%d] skipout = %.9f\n", order, skipout);
            back_propagate();
            count++;
            avg_skipout = (avg_skipout * (float)(count-1) + skipout) / (float)count;
        }
    }
}


// Initializes the 32kHz crystal and MCLK to 8MHz
void initClock(void)
{
    // Disable the GPIO power-on default high-impedance mode to activate
    // previously configured port settings
    PM5CTL0 &= ~LOCKLPM5;

    // Configure one FRAM waitstate as required by the device datasheet for MCLK
    // operation beyond 8MHz _before_ configuring the clock system.
    FRCTL0 = FRCTLPW | NWAITS_1;

    // Clock System Setup
    CSCTL0_H = CSKEY_H;                     // Unlock CS registers
    CSCTL1 = DCOFSEL_0;                     // Set DCO to 1MHz
    // Set SMCLK = MCLK = DCO, ACLK = VLOCLK
    CSCTL2 = SELA__VLOCLK | SELS__DCOCLK | SELM__DCOCLK;
    // Per Device Errata set divider to 4 before changing frequency to
    // prevent out of spec operation from overshoot transient
    CSCTL3 = DIVA__4 | DIVS__4 | DIVM__4;   // Set all corresponding clk sources to divide by 4 for errata
    CSCTL1 = DCOFSEL_4 | DCORSEL;           // Set DCO to 16MHz
    // Delay by ~10us to let DCO settle. 60 cycles = 20 cycles buffer + (10us / (1/4MHz))
    __delay_cycles(60);
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1;   // Set all dividers to 1 for 16MHz operation
    CSCTL0_H = 0;
}

void initGpio(void)
{
    P1OUT = 0x00;
    P1DIR = 0xFF;

    P2OUT = 0x00;
    P2DIR = 0xFF;

    P3OUT = 0x00;
    P3DIR = 0xFF;

    P4OUT = 0x01;
    P4DIR = 0xFF;

    P5OUT = 0x00;
    P5DIR = 0xFF;

    P6OUT = 0x00;
    P6DIR = 0xFF;

    P7OUT = 0x00;
    P7DIR = 0xFF;

    P8OUT = 0x04;
    P8DIR = 0xFF;

    PJOUT = 0x00;
    PJDIR = 0xFF;
}

void runApplication(void)
{
    init_train();

    while (1) {
        train();
    }
}

int main(void)
{
    uint16_t gie;

    WDTCTL = WDTPW | WDTHOLD;       // Stop watchdog timer

    initClock();
    initGpio();

    PM5CTL0 &= ~LOCKLPM5;           // Clear lock bit
    gie = __get_SR_register() & GIE;
    __disable_interrupt();
    __bis_SR_register(gie);
    init_train();

    while (1) {
        train();
    }
}
