CC=gcc
CXX=g++
CFLAGS=-I. -O2
DEPS = SLIC.h
OBJ = SLIC.o

%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS)

SLIC: $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS)

clean:
	rm SLIC SLIC.o