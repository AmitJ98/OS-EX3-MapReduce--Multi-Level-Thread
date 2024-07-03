CXX = g++

CXXFLAGS = -Wall -std=c++11

SRC = Barrier.cpp MapReduceFramework.cpp
    
OBJ = $(SRC:.cpp=.o)

LIB = libMapReduceFramework.a

# Default target
all: $(EXEC)

# Default target
all: $(LIB)

# Create static library from object files
$(LIB): $(OBJ)
	ar rcs $@ $(OBJ)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create tar archive of the project
tar:
	tar -cvf project.tar $(SRC) Barrier.h Makefile README

# Clean up object files and library
clean:
	rm -f $(OBJ) $(LIB) 