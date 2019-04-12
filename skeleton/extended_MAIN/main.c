#include <msp430.h> 
#include <stdio.h>
#include <math.h>
#define GLOBAL_Q 8
#include <QmathLib.h>
#include <DSPLib.h>
#include "fram_util/fram_util.h"
#include "spi/spi.h"
#include "adc/adc.h"
#include "baseline_param.h"
#include "extended_param.h"

#define MAX_INPUT_DIM           3
#define MAX_OUTPUT_DIM          MAX_INPUT_DIM
#define MAX_INPUT_SIZE          1024
#define MAX_OUTPUT_SIZE         100
#define MAX_NEURON_SIZE         16000
#define MAX_BIAS_SIZE           2048
#define MAX_VECTOR_LEN          2048
#define MAX_INTERIM_SIZE        4096
#define INTERIM_ADDR            0x40
#define INTERIM_LEN_ADDR        0x00
#define ADC_THRESHOLD           30816
#define ACC_OUT                 P5OUT
#define ACC_DIR                 P5DIR
#define ACC_PIN                 BIT7
#define LIVE_IN                 P3IN
#define LIVE_DIR                P3DIR
#define LIVE_PIN                BIT7

int neuron_toggle = 0;
_q *input_addr;
int32_t input_dim[MAX_INPUT_DIM] = { 0, };
_q *output_addr;
int32_t output_dim[MAX_OUTPUT_DIM] = { 0, };
_q *weight_addr;
_q *bias_addr;

#pragma PERSISTENT(input)
_q input[MAX_INPUT_SIZE] = { 0, };
_q output[MAX_OUTPUT_SIZE] = { 0, };

#pragma PERSISTENT(neuron)
_q neuron[2][MAX_NEURON_SIZE] = { 0, };

#pragma PERSISTENT(interim)
_q interim[MAX_INTERIM_SIZE] = { 0, };

#pragma PERSISTENT(vector1)
_q vector1[MAX_VECTOR_LEN] = { 0, };

#pragma PERSISTENT(vector2)
_q vector2[MAX_VECTOR_LEN] = { 0, };

uint16_t pooling_dim[4] = { 2, 2, 2, 2 };

#define MAX_NUM_Q_LEA           (944)

typedef struct
{
    _q raw1[MAX_NUM_Q_LEA];
    _q raw2[MAX_NUM_Q_LEA];
    _iq31 result;
} LeaMem;

#pragma DATA_SECTION(leaMem, ".leaRAM")
LeaMem leaMem;

static void copyDataNF(const void *src, void *dst, uint16_t length)
{
    uint16_t i;
    uint16_t *srcPtr;
    uint16_t *dstPtr;

    srcPtr = (uint16_t *) src;
    dstPtr = (uint16_t *) dst;

    for (i = 0; i < length / 2; i++)
    {
        *dstPtr++ = (uint16_t) ir((int *) &srcPtr[i]);
    }
}

_q vector_dot_product_lea(_q *vector1, _q *vector2, int len)
{
    msp_status status;
    uint16_t i;
    uint16_t iteration;
    uint16_t residual;
    _iq *result = &leaMem.result;
    _iq sum = 0;

    float result_f;
    _q result_q;

    msp_mac_q15_params param;
    iteration = len / MAX_NUM_Q_LEA;

    for (i = 0; i < iteration; i++)
    {
        copyDataNF(&vector1[i * MAX_NUM_Q_LEA], leaMem.raw1, MAX_NUM_Q_LEA * 2);
        copyDataNF(&vector2[i * MAX_NUM_Q_LEA], leaMem.raw2, MAX_NUM_Q_LEA * 2);

        param.length = MAX_NUM_Q_LEA;
        status = msp_mac_q15(&param, leaMem.raw1, leaMem.raw2, result);
        if (status != MSP_SUCCESS)
        {
            P1OUT |= BIT0;
        }

        sum += *result;
    }

    residual = len % MAX_NUM_Q_LEA;

    if (residual)
    {
        copyDataNF(&vector1[i * MAX_NUM_Q_LEA], leaMem.raw1, residual * 2);
        copyDataNF(&vector2[i * MAX_NUM_Q_LEA], leaMem.raw2, residual * 2);

        param.length = residual;

        if (residual % 2)
        {
            leaMem.raw1[residual] = 0;
            leaMem.raw2[residual] = 0;
            param.length += 1;
        }

        status = msp_mac_q15(&param, leaMem.raw1, leaMem.raw2, result);
        if (status != MSP_SUCCESS)
        {
            P1OUT |= BIT0;
        }

        sum += *result;
    }

    result_f = _IQ30toF(sum) / 2 * (1 << (2 * (15 - GLOBAL_Q)));
    result_q = _Q(result_f);

    return result_q;
}

_q vector_dot_product(_q *a, _q *b, int len)
{
    _q dot_product = 0;

    int i;
    for (i = 0; i < len; i++)
    {
        dot_product += _Qmpy(ir(&a[i]), ir(&b[i]));
    }

    return dot_product;
}

void conv2d(_q *input_addr, int32_t input_dim[3], _q *weight_addr,
            _q *bias_addr, uint16_t layer_dim[MAX_NUM_OF_LAYER_DIM],
            _q *output_addr, int32_t output_dim[MAX_OUTPUT_DIM])
{
    // sanity check
    if (input_dim[2] != layer_dim[2])
    {
        printf("input_dim[2] != layer_dim[2]\n");
        exit(1);
    }

    int w, h, d, k, w2, h2;
    int temp;
    int stride_w = layer_dim[4];
    int stride_h = layer_dim[5];
    int pad_w = layer_dim[6];
    int pad_h = layer_dim[7];

    uint32_t vector_len = layer_dim[0] * layer_dim[1] * layer_dim[2];
    uint32_t input_idx = 0;
    uint32_t output_idx = 0;
    uint32_t bias_vector_idx = 0;

    output_dim[0] = (input_dim[0] - layer_dim[0] + 2 * pad_w) / stride_w + 1; // width
    output_dim[1] = (input_dim[1] - layer_dim[1] + 2 * pad_h) / stride_h + 1; // height
    output_dim[2] = layer_dim[3]; // k

    // sanity check
    if (output_dim[0] < 0 || output_dim[1] < 0 || output_dim[2] < 0)
    {
        printf("output_dim[0] <= 0 || output_dim[1] <= 0 || output_dim[2] <= 0");
        exit(1);
    }

    for (k = 0; k < layer_dim[3]; k++)
    {
        uint32_t filter_idx = k * (layer_dim[0] * layer_dim[1] * layer_dim[2]);

        for (temp = 0; temp < vector_len; temp++)
        {
            iw(&vector2[temp], ir(&weight_addr[filter_idx++]));
        }

        _q bias = ir(&bias_addr[bias_vector_idx++]);
        for (h = 0; h + layer_dim[1] <= input_dim[1]; h += stride_h)
        {
            for (w = 0; w + layer_dim[0] <= input_dim[0]; w += stride_w)
            {
                uint32_t lea_input_idx = 0;
                for (d = 0; d < layer_dim[2]; d++)
                {
                    input_idx = input_dim[0] * input_dim[1] * d
                            + input_dim[0] * h + w;
                    for (h2 = 0; h2 < layer_dim[1]; h2++)
                    {
                        for (w2 = 0; w2 < layer_dim[0]; w2++)
                        {
                            iw(&vector1[lea_input_idx++],
                               ir(&input_addr[input_idx++]));
                        }
                        input_idx += input_dim[0] - layer_dim[0];
                    }
                }

                _q output = vector_dot_product_lea(vector1, vector2, vector_len)
                        + bias;
                iw(&output_addr[output_idx++], output);
            }
        }
    }

    // sanity check
    if (output_idx != output_dim[0] * output_dim[1] * output_dim[2])
    {
        printf("output_idx != output_dim[0]*output_dim[1]*output_dim[2]\n");
        printf("%ld != %ld\n", output_idx,
               output_dim[0] * output_dim[1] * output_dim[2]);
        exit(1);
    }
}

void softmax(_q *x, _q *y, int length)
{
    float x_f[MAX_OUTPUT_SIZE] = { 0, };
    float y_f[MAX_OUTPUT_SIZE] = { 0, };
    float max;
    float exp_sum = 0;
    int i;

    for (i = 0; i < length; i++)
    {
        x_f[i] = _QtoF(x[i]);
        if (i == 0)
        {
            max = x_f[i];
        }

        if (x_f[i] > max)
        {
            max = x_f[i];
        }
    }

    for (i = 0; i < length; i++)
    {
        y_f[i] = x_f[i] - max;
        y_f[i] = exp(y_f[i]);
        exp_sum += y_f[i];
    }

    for (i = 0; i < length; i++)
    {
        y_f[i] /= exp_sum;
        iw(&y[i], _Q(y_f[i]));
    }
}

_q leaky_relu(_q input)
{
    if (input < 0)
    {
        return _Qmpy(0.2, input);
    }
    else
    {
        return input;
    }
}

void activation(_q *input_addr, int32_t input_dim[MAX_INPUT_DIM],
                _q *output_addr, int32_t output_dim[MAX_OUTPUT_DIM])
{
    int i;
    uint32_t idx = 0;

    output_dim[0] = input_dim[0];
    output_dim[1] = input_dim[1];
    output_dim[2] = input_dim[2];

    int input_len = 1;

    for (i = 0; i < MAX_INPUT_DIM; i++)
    {
        if (input_dim[i] > 0)
        {
            input_len *= input_dim[i];
        }
    }

    for (i = 0; i < input_len; i++)
    {
        iw(&output_addr[idx], leaky_relu(ir(&input_addr[idx])));
        idx++;
    }
}

float q_max(_q *array, int len)
{
    _q max;
    int i;

    for (i = 0; i < len; i++)
    {
        if (i == 0)
        {
            max = ir(&array[i]);
        }
        else
        {
            if (ir(&array[i]) > max)
            {
                max = ir(&array[i]);
            }
        }
    }

    return max;
}

void max_pooling(_q *input_addr, int32_t input_dim[MAX_INPUT_DIM],
                 uint16_t pooling_dim[4], _q *output_addr,
                 int32_t output_dim[MAX_OUTPUT_DIM])
{

    int w, h, d, w2, h2;
    int stride_w = pooling_dim[2];
    int stride_h = pooling_dim[3];

    uint32_t pooling_idx = 0;
    uint32_t input_idx = 0;
    uint32_t output_idx = 0;

    int pooling_vector_len = pooling_dim[0] * pooling_dim[1];

    output_dim[0] = (input_dim[0] - pooling_dim[0]) / stride_w + 1; // width
    output_dim[1] = (input_dim[1] - pooling_dim[1]) / stride_h + 1; // height
    output_dim[2] = input_dim[2]; // d

    // sanity check
    if (output_dim[0] <= 0 || output_dim[1] <= 0 || output_dim[2] <= 0)
    {
        printf("output_dim[0] < 0 || output_dim[1] < 0 || output_dim[2] < 0");
        exit(1);
    }

    for (d = 0; d < input_dim[2]; d++)
    {
        for (h = 0; h + pooling_dim[1] <= input_dim[1]; h += stride_h)
        {
            for (w = 0; w + pooling_dim[0] <= input_dim[0]; w += stride_w)
            {
                pooling_idx = 0;
                input_idx = input_dim[0] * input_dim[1] * d + input_dim[0] * h
                        + w;
                for (w2 = 0; w2 < pooling_dim[0]; w2++)
                {
                    for (h2 = 0; h2 < pooling_dim[1]; h2++)
                    {
                        iw(&vector1[pooling_idx++],
                           ir(&input_addr[input_idx++]));
                    }
                    input_idx += input_dim[0] - pooling_dim[0];
                }
                iw(&output_addr[output_idx++],
                   q_max(vector1, pooling_vector_len));
            }
        }
    }

    // sanity check
    if (output_idx != output_dim[0] * output_dim[1] * output_dim[2])
    {
        printf("output_idx != output_dim[0]*output_dim[1]*output_dim[2]\n");
        printf("%ld != %ld\n", output_idx,
               output_dim[0] * output_dim[1] * output_dim[2]);
        exit(1);
    }
}

void fc(_q *input_addr, int32_t input_dim[MAX_INPUT_DIM], _q *weight_addr,
        _q *bias_addr, uint16_t layer_dim[MAX_NUM_OF_LAYER_DIM],
        _q *output_addr, int32_t output_dim[MAX_OUTPUT_DIM])
{
    uint32_t input_len = 1;
    int i;

    for (i = 0; i < MAX_INPUT_DIM; i++)
    {
        if (input_dim[i] > 0)
        {
            input_len *= input_dim[i];
        }
    }

    uint32_t output_idx = 0;
    uint32_t weight_vector_idx = 0;
    uint32_t bias_vector_idx = 0;

    for (i = 0; i < layer_dim[0]; i++)
    {
        _q bias = ir(&bias_addr[bias_vector_idx++]);
        iw(&output_addr[output_idx++],
           vector_dot_product_lea(input_addr, weight_addr + weight_vector_idx,
                                  input_len) + bias);
        weight_vector_idx += (input_len);
    }

    // sanity check
    if (layer_dim[0] != output_idx)
    {
        printf("layer_dim[0] != output_idx\n");
        exit(1);
    }

    output_dim[0] = output_idx;
    output_dim[1] = 0;
    output_dim[2] = 0;
}

int16_t feed_forward(_q *input_addr, int32_t input_dim[MAX_INPUT_DIM],
                     _q *weight_addr, _q *bias_addr,
                     uint16_t layer[][MAX_NUM_OF_LAYER_DIM], int num_of_layer,
                     int accel)
{
    int i = 0, l = 0;

    output_addr = neuron[neuron_toggle];

    for (l = 1; l < num_of_layer; l++)
    {
        int num_of_layer_item = 0;
        for (i = 0; i < MAX_NUM_OF_LAYER_DIM; i++)
        {
            if (layer[l][i] != 0)
            {
                num_of_layer_item += 1;
            }
        }

        if (num_of_layer_item <= 3) // fc layer
        {
            if (accel)
            {
                int num_of_prev_layer_item = 0;

                for (i = 0; i < MAX_NUM_OF_LAYER_DIM; i++)
                {
                    if (layer[l - 1][i] != 0)
                    {
                        num_of_prev_layer_item += 1;
                    }
                }

                if (num_of_prev_layer_item >= 4)
                {
                    int delay_count = 1000;
                    while (LIVE_IN & LIVE_PIN)
                    {
                        __delay_cycles(1600);
                        if (delay_count-- <= 0)
                            break;
                    }

                    if (delay_count > 0)
                    {
                        uint8_t send[4] = { 0x03, 0, 0, INTERIM_LEN_ADDR };
                        uint16_t interim_len = 0;
                        spi_transfer(send, (uint8_t *) &interim_len, 4, 2, 0,
                                     0);
                        send[0] = 0x03;
                        send[1] = 0x00;
                        send[2] = 0x00;
                        send[3] = INTERIM_ADDR;
                        spi_transfer(send, (uint8_t *) interim, 4, interim_len,
                                     1, 1);

                        int input_len = input_dim[0] * input_dim[1]
                                * input_dim[2];
                        for (i = 0; i < interim_len; i++)
                        {
                            input_addr[input_len + i] = ir(&interim[i]);
                        }
                        input_dim[0] = input_len + interim_len;
                        input_dim[1] = 1;
                        input_dim[2] = 1;

                        uint32_t weight_offset = 0;
                        uint32_t bias_offset = 0;
                        int num_of_layer2 = sizeof(extended_layer)
                                / sizeof(extended_layer[0]);
                        int l2;
                        layer = extended_layer;

                        for (l2 = 1; l2 < num_of_layer2; l2++)
                        {
                            int num_of_layer_item2 = 0;
                            for (i = 0; i < MAX_NUM_OF_LAYER_DIM; i++)
                            {
                                if (layer[l2][i] != 0)
                                {
                                    num_of_layer_item2 += 1;
                                }
                            }

                            if (num_of_layer_item2 >= 4)
                            {
                                uint32_t offset = 1;
                                int j;
                                for (j = 0; j < num_of_layer_item2; j++)
                                {
                                    offset *= layer[l2][j];
                                }
                                bias_offset += layer[l2][3];
                                weight_offset += offset;
                            }
                        }

                        weight_addr = extended_weights + weight_offset;
                        bias_addr = extended_biases + bias_offset;
                    }
                }
            }

            fc(input_addr, input_dim, weight_addr, bias_addr, layer[l],
               output_addr, output_dim);
            int input_len = 1;
            for (i = 0; i < MAX_INPUT_DIM; i++)
                if (input_dim[i] > 0)
                    input_len *= input_dim[i];
            weight_addr += (input_len * output_dim[0]);
            bias_addr += 1;

            input_addr = output_addr;
            for (i = 0; i < MAX_INPUT_DIM; i++)
                input_dim[i] = output_dim[i];
            neuron_toggle = !neuron_toggle;
            output_addr = neuron[neuron_toggle];

            if (l != num_of_layer - 1) // hidden layer
            {
                // activation (relu)
                activation(input_addr, input_dim, output_addr, output_dim);
                input_addr = output_addr;
                for (i = 0; i < MAX_INPUT_DIM; i++)
                    input_dim[i] = output_dim[i];
                neuron_toggle = !neuron_toggle;
                output_addr = neuron[neuron_toggle];

            }
            else // output layer
            {
                softmax(input_addr, output, layer[l][0]);
                _q max;
                int max_class;

                for (i = 0; i < layer[l][0]; i++)
                {
                    if (i == 0)
                    {
                        max = output[i];
                        max_class = i;
                    }
                    else
                    {
                        if (output[i] > max)
                        {
                            max = output[i];
                            max_class = i;
                        }
                    }

                    //printf("%f ", _QtoF(output[i]));
                }
                //printf("\n");

                return max_class;
            }
        }
        else if (num_of_layer_item >= 4) // conv layer
        {
            // conv
            conv2d(input_addr, input_dim, weight_addr, bias_addr, layer[l],
                   output_addr, output_dim);
            input_addr = output_addr;
            for (i = 0; i < MAX_INPUT_DIM; i++)
                input_dim[i] = output_dim[i];
            neuron_toggle = !neuron_toggle;
            output_addr = neuron[neuron_toggle];
            weight_addr += (layer[l][0] * layer[l][1] * layer[l][2]
                    * layer[l][3]);
            bias_addr += layer[l][3];

            // activation (relu)
            activation(input_addr, input_dim, output_addr, output_dim);
            input_addr = output_addr;
            for (i = 0; i < MAX_INPUT_DIM; i++)
                input_dim[i] = output_dim[i];
            neuron_toggle = !neuron_toggle;
            output_addr = neuron[neuron_toggle];

            // max pool
            max_pooling(input_addr, input_dim, pooling_dim, output_addr,
                        output_dim);
            input_addr = output_addr;
            for (i = 0; i < MAX_INPUT_DIM; i++)
                input_dim[i] = output_dim[i];
            neuron_toggle = !neuron_toggle;
            output_addr = neuron[neuron_toggle];
        }
    }

    return -1;
}

void init_clock()
{
    WDTCTL = WDTPW | WDTHOLD;               // Stop WDT

    // Configure GPIO
    P1OUT &= ~BIT0;      // Clear P1.0 output latch for a defined power-on state
    P1DIR |= BIT0;                          // Set P1.0 to output direction

    P2DIR |= BIT0;
    P2SEL0 |= BIT0;                         // Output ACLK
    P2SEL1 |= BIT0;

    P3DIR |= BIT4;
    P3SEL1 |= BIT4;                         // Output SMCLK
    P3SEL0 |= BIT4;

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
    CSCTL3 = DIVA__4 | DIVS__4 | DIVM__4; // Set all corresponding clk sources to divide by 4 for errata
    CSCTL1 = DCOFSEL_4 | DCORSEL;           // Set DCO to 16MHz
    // Delay by ~10us to let DCO settle. 60 cycles = 20 cycles buffer + (10us / (1/4MHz))
    __delay_cycles(60);
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1; // Set all dividers to 1 for 16MHz operation
    CSCTL0_H = 0; // Lock CS registers                      // Lock CS registers
}

void sense(_q *data)
{
    int idx = 0;
    int input_len = baseline_layer[0][0] * baseline_layer[0][1]
            * baseline_layer[0][2];

    for (idx = 0; idx < input_len; idx++)
    {
        float random = rand() / (float) RAND_MAX;
        iw(&data[idx], _Q(random));
    }
}

int main(void)
{
    WDTCTL = WDTPW | WDTHOLD;
    PMM_unlockLPM5();

    ACC_DIR |= ACC_PIN;
    LIVE_DIR &= (~LIVE_PIN);

    initSPI();
    initADC();
    init_clock();

    printf("System ready!\n");

    while (1)
    {
        sense(input);

        int prediction;
        int i;
        int adc_val = AnalogRead(ADC12INCH_15);
        int accel = 0;

        for (i = 0; i < MAX_INPUT_DIM; i++)
        {
            input_dim[i] = baseline_layer[0][i];
        }

        if (adc_val >= ADC_THRESHOLD)
        {
            accel = 1;
        }

        if (accel)
        {
            int input_len = baseline_layer[0][0] * baseline_layer[0][1]
                    * baseline_layer[0][2];

            uint8_t send[4] = { 0, }, recv = 0;
            send[0] = 0x02;
            send[1] = 0x00;
            send[2] = 0x00;
            send[3] = INTERIM_LEN_ADDR;
            spi_transfer(send, &recv, 4, 0, 0, 0);
            spi_transfer((uint8_t *) input_len, &recv, 2, 0, 0, 0);
            send[0] = 0x02;
            send[1] = 0x00;
            send[2] = 0x00;
            send[3] = INTERIM_ADDR;
            spi_transfer(send, &recv, 4, 0, 0, 0);
            spi_transfer((uint8_t *) input, &recv, input_len * 2, 0, 1, 0);
            ACC_OUT |= ACC_PIN;
        }

        prediction = feed_forward(
                input, input_dim, baseline_weights, baseline_biases,
                baseline_layer,
                sizeof(baseline_layer) / sizeof(baseline_layer[0]), accel);

        ACC_OUT &= ~(ACC_PIN);

        printf("prediction = %d\n", prediction);
    }
}
