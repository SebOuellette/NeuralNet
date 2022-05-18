CXX := g++
EXE := main
FOLDER := Source
#OPTIONS := 

SOURCE_FILES := $(wildcard $(FOLDER)/*.cpp)
OBJECT_FILES := $(patsubst $(FOLDER)/%.cpp,%.o,$(SOURCE_FILES))

all: $(EXE)

$(EXE): $(OBJECT_FILES)
	$(CXX) $(OBJECT_FILES) -o $(EXE) $(OPTIONS)

%.o: $(FOLDER)/%.cpp
	$(CXX) -c $(FOLDER)/$*.cpp

valgrind:
	valgrind ./$(EXE)

clean:
	rm -f *.o $(EXE) vgcore.*