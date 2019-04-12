#include "spi.h"

uint8_t *ReceiveBuffer;
int16_t RXByteCtr;
int16_t ReceiveIndex;

uint8_t *TransmitBuffer;
int16_t TXByteCtr;
int16_t TransmitIndex;

uint8_t is_send_fram;
uint8_t is_recv_fram;

static void SendUCB1Data(uint8_t val)
{
    while (!(UCB1IFG & UCTXIFG))
        ;              // USCI_B1 TX buffer ready?
    UCB1TXBUF = val;
}

void initSPI()
{
    SLAVE_CS_DIR |= SLAVE_CS_PIN;
    SLAVE_CS_OUT |= SLAVE_CS_PIN;

    // Configure SPI
    P5SEL0 |= BIT0 | BIT1 | BIT2;

    //Clock Polarity: The inactive state is high
    //MSB First, 8-bit, Master, 3-pin mode, Synchronous
    UCB1CTLW0 = UCSWRST;                       // **Put state machine in reset**
    UCB1CTLW0 |= UCCKPL | UCMSB | UCSYNC
                | UCMST | UCSSEL__SMCLK;      // 3-pin, 8-bit SPI Slave
    UCB1BRW = 0x01;
    //UCB1MCTLW = 0;
    UCB1CTLW0 &= ~UCSWRST;                     // **Initialize USCI state machine**
    UCB1IE |= UCRXIE;                          // Enable USCI0 RX interrupt
}

void spi_transfer(uint8_t *send, uint8_t *recv, uint16_t send_len,
                  uint16_t recv_len, uint8_t send_fram, uint8_t recv_fram)
{
    TXByteCtr = send_len - 1;
    RXByteCtr = recv_len;
    ReceiveIndex = -1;
    TransmitIndex = 1;
    TransmitBuffer = send;
    ReceiveBuffer = recv;

    is_send_fram = send_fram;
    is_recv_fram = recv_fram;

    SLAVE_CS_OUT &= ~(SLAVE_CS_PIN);
    SendUCB1Data(send[0]);

    __bis_SR_register(CPUOFF + GIE);              // Enter LPM0 w/ interrupts

    SLAVE_CS_OUT |= SLAVE_CS_PIN;
}

#pragma vector=USCI_B1_VECTOR
__interrupt void USCI_B1_ISR(void)
{
    uint8_t ucb1_rx_val = 0;
    uint8_t data;

    switch (__even_in_range(UCB1IV, USCI_SPI_UCTXIFG))
    {
    case USCI_NONE:
        break;
    case USCI_SPI_UCRXIFG:
        ucb1_rx_val = UCB1RXBUF;
        UCB1IFG &= ~UCRXIFG;

        if (TXByteCtr)
        {
            if (is_send_fram) {
                data = __data20_read_char((unsigned long int)&TransmitBuffer[TransmitIndex++]);

            } else {
                data = TransmitBuffer[TransmitIndex++];
            }
            SendUCB1Data(data);
            TXByteCtr--;
        }
        else if (RXByteCtr)
        {
            if (ReceiveIndex == -1)
            {
                ReceiveIndex = 0;
                SendUCB1Data(DUMMY);
            }
            else
            {
                if (is_recv_fram) {
                    __data20_write_char((unsigned long int)&ReceiveBuffer[ReceiveIndex++], ucb1_rx_val);
                } else {
                    ReceiveBuffer[ReceiveIndex++] = ucb1_rx_val;
                }
                if (--RXByteCtr > 0)
                {
                    SendUCB1Data(DUMMY);
                }
                else
                {
                    __bic_SR_register_on_exit(CPUOFF);
                }
            }
        }
        else
        {
            __bic_SR_register_on_exit(CPUOFF);      // Exit LPM0
        }
        __delay_cycles(10);
        break;
    case USCI_SPI_UCTXIFG:
        break;
    default:
        break;
    }
}
