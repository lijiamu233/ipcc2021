CC=gcc
CXX=mpicxx
CFLAGS=-I. -O3 -march=znver1 -mtune=znver1 -mfma -mavx2 -m3dnow -fomit-frame-pointer -g  -fopenmp -Wall -ffast-math -ftree-loop-vectorize -lm # -fsanitize=address
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