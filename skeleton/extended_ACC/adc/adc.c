#include <msp430.h>
#include <driverlib.h>

void initADC()
{
#if 1
    PJSEL0 |= BIT4 | BIT5;                  // For XT1

    // XT1 Setup
    CSCTL0_H = CSKEY >> 8;                  // Unlock CS registers
    CSCTL1 = DCOFSEL_6;                     // Set DCO to 8MHz
    CSCTL2 = SELA__LFXTCLK | SELS__DCOCLK | SELM__DCOCLK;
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1;   // set all dividers
    CSCTL4 &= ~LFXTOFF;

    do
    {
        CSCTL5 &= ~LFXTOFFG;                // Clear XT1 fault flag
        SFRIFG1 &= ~OFIFG;
    }
    while(SFRIFG1 & OFIFG);                 // Test oscillator fault flag
    CSCTL0_H = 0;                           // Lock CS registers
#endif


    P3OUT = 0x00;
    P3DIR = 0xFF;

    ADC12_B_initParam adcConfig;
    ADC12_B_configureMemoryParam adcMemConfig;
    Timer_A_initUpModeParam timerUpConfig;
    Timer_A_initCompareModeParam timerCompareConfig;

    // Configure TA1.1 for ADC sample interval
    timerUpConfig.clockSource = TIMER_A_CLOCKSOURCE_SMCLK;
    timerUpConfig.clockSourceDivider = TIMER_A_CLOCKSOURCE_DIVIDER_1;
    //timerUpConfig.timerPeriod = (16000000 / 8000) - 1;
    timerUpConfig.timerPeriod = (16000000 / 1) - 1;
    timerUpConfig.timerInterruptEnable_TAIE = TIMER_A_TAIE_INTERRUPT_DISABLE;
    timerUpConfig.captureCompareInterruptEnable_CCR0_CCIE = TIMER_A_CCIE_CCR0_INTERRUPT_DISABLE;
    timerUpConfig.timerClear = TIMER_A_DO_CLEAR;
    timerUpConfig.startTimer = false;
    Timer_A_initUpMode(TIMER_A0_BASE, &timerUpConfig);

    // Initialize TA0CCR1 to generate trigger clock output, reset/set mode
    timerCompareConfig.compareRegister = TIMER_A_CAPTURECOMPARE_REGISTER_1;
    timerCompareConfig.compareInterruptEnable = TIMER_A_CAPTURECOMPARE_INTERRUPT_DISABLE;
    timerCompareConfig.compareOutputMode = TIMER_A_OUTPUTMODE_SET_RESET;
    //timerCompareConfig.compareValue = ((16000000 / 8000) / 2) - 1;
    timerCompareConfig.compareValue = ((16000000 / 1) / 2) - 1;
    Timer_A_initCompareMode(TIMER_A0_BASE, &timerCompareConfig);

    GPIO_setAsPeripheralModuleFunctionOutputPin(GPIO_PORT_P3, GPIO_PIN0, GPIO_TERNARY_MODULE_FUNCTION);
    adcConfig.sampleHoldSignalSourceSelect = ADC12_B_SAMPLEHOLDSOURCE_1;
    adcConfig.clockSourceSelect = ADC12_B_CLOCKSOURCE_ADC12OSC;
    adcConfig.clockSourceDivider = ADC12_B_CLOCKDIVIDER_1;
    adcConfig.clockSourcePredivider = ADC12_B_CLOCKPREDIVIDER__1;
    adcConfig.internalChannelMap = ADC12_B_NOINTCH;
    ADC12_B_init(ADC12_B_BASE, &adcConfig);
    ADC12_B_enable(ADC12_B_BASE);
    ADC12_B_setupSamplingTimer(ADC12_B_BASE,
                               ADC12_B_CYCLEHOLD_16_CYCLES,
                               ADC12_B_CYCLEHOLD_16_CYCLES,
                               ADC12_B_MULTIPLESAMPLESDISABLE);
    ADC12_B_setResolution(ADC12_B_BASE,
                          ADC12_B_RESOLUTION_12BIT);
    ADC12_B_setDataReadBackFormat(ADC12_B_BASE,
                                  ADC12_B_SIGNED_2SCOMPLEMENT);
    adcMemConfig.differentialModeSelect = ADC12_B_DIFFERENTIAL_MODE_DISABLE;
    adcMemConfig.endOfSequence = ADC12_B_NOTENDOFSEQUENCE;
    adcMemConfig.inputSourceSelect = ADC12_B_INPUT_A15;
    adcMemConfig.memoryBufferControlIndex = ADC12_B_MEMORY_0;
    adcMemConfig.refVoltageSourceSelect = ADC12_B_VREFPOS_AVCC_VREFNEG_VSS;
    adcMemConfig.windowComparatorSelect = ADC12_B_WINDOW_COMPARATOR_DISABLE;
    ADC12_B_configureMemory(ADC12_B_BASE, &adcMemConfig);
    ADC12_B_startConversion(ADC12_B_BASE,
                            ADC12_B_MEMORY_0,
                            ADC12_B_REPEATED_SINGLECHANNEL);


    // Start TA0 timer to begin audio data collection
    Timer_A_clear(TIMER_A0_BASE);
    Timer_A_startCounter(TIMER_A0_BASE, TIMER_A_UP_MODE);
}

int AnalogRead(uint8_t channel)
{
    return ADC12MEM0;
}
