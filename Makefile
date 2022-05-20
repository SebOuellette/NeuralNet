CXX := g++
EXE := main
FOLDER := Source
LINKER_OPTS := -lpthread
COMPILE_OPTS := #-g

SOURCE_FILES := $(wildcard $(FOLDER)/*.cpp)
OBJECT_FILES := $(patsubst $(FOLDER)/%.cpp,%.o,$(SOURCE_FILES))

all: $(EXE)

$(EXE): $(OBJECT_FILES)
	$(CXX) $(OBJECT_FILES) -o $(EXE) $(LINKER_OPTS)

%.o: $(FOLDER)/%.cpp
	$(CXX) -c $(COMPILE_OPTS) $(FOLDER)/$*.cpp

valgrind:
	valgrind ./$(EXE)

clean:
	rm -f *.o $(EXE) vgcore.*