CC = mpicxx 

CFLAGS += -g -Wall -std=c++11  -mavx2 
INC += -I. `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv`

TARGET = main 
OBJS += main.o  
	     

all:$(TARGET)
$(TARGET):$(OBJS)
	$(CC) $(INC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LIBS)
$(OBJS):%.o:%.cpp
	$(CC) $(INC) $(CFLAGS) -c $< -o $@

.PHONY:clean
clean:
	-rm  *.o $(TARGET) result.bmp
