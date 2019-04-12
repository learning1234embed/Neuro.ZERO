#include <msp430.h>
#include <stdint.h>

#define SLAVE_CS_OUT    P5OUT
#define SLAVE_CS_DIR    P5DIR
#define SLAVE_CS_PIN    BIT3

#define SLAVE_CS_OUT2    P4OUT
#define SLAVE_CS_DIR2    P4DIR
#define SLAVE_CS_PIN2    BIT4
#define DUMMY   0xFF

void initSPI();
void spi_transfer(uint8_t *send, uint8_t *recv, uint16_t send_len,
                  uint16_t recv_len, uint8_t send_fram, uint8_t recv_fram);
