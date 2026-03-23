# Sample makefile
CC = mpicc
CFLAGS = -O2 -Wall -Wextra -std=c11

all: producer_consumer workpool

producer_consumer: producer_consumer.c
	$(CC) $(CFLAGS) -o $@ $<

workpool: workpool.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f producer_consumer workpool

