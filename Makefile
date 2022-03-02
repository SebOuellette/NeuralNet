CXX = g++

SOURCE_FILES := $(wildcard Source/*.cpp)
OBJECT_FILES := $(patsubst Source/%.cpp,%.o,$(wildcard Source/*.cpp))

all: main

%.o: Source/%.cpp
	$(CXX) -c Source/$*.cpp

main: $(OBJECT_FILES)
	$(CXX) $(OBJECT_FILES) -o main

valgrind:
	valgrind ./main

clean:
	rm -f *.o main vgcore.*