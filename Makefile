CC=icc
CXX=icpc
CFLAGS=-I. -O1 -fomit-frame-pointer  -fopenmp -Wall -ffast-math -march=core-avx2 -mfma -ipo
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