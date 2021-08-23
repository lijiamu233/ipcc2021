CC=gcc
CXX=g++
CFLAGS=-I. -O3 -g -march=native -mtune=native -fopenmp
DEPS = SLIC.h Makefile
OBJ = SLIC.o

%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS)

SLIC: $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS)

clean:
	rm SLIC SLIC.o