CXX = g++

SOURCE_FILES := $(wildcard Source/*.cpp)
OBJECT_FILES := $(patsubst Source/%.cpp,%.o,$(wildcard Source/*.cpp))

all: main

%.o: Source/%.cpp
	$(CXX) -std=c++0x -c Source/$*.cpp -lpthread

main: $(OBJECT_FILES)
	$(CXX)  $(OBJECT_FILES) -o main -lpthread

valgrind:
	valgrind ./main

clean:
	rm -f *.o main vgcore.*