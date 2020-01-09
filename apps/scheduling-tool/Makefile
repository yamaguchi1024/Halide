BIN=bin
default:
	@mkdir -p $(BIN)
	g++ -shared -rdynamic -fPIC -O3 -std=c++11 -I /home/yuka/Halide/src -I /home/yuka/Halide/distrib/include/ -I /home/yuka/Halide/distrib/tools/ -I /home/yuka/Halide/apps/autoscheduler -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -g -O3 -I ../autoscheduler/bin/cost_model ./SchedulingTool.cpp ./PrintLoopNest.cpp ../autoscheduler/ASLog.cpp ../autoscheduler/DefaultCostModel.cpp ../autoscheduler/Weights.cpp ../autoscheduler/bin/baseline_weights.o ../autoscheduler/bin/cost_model/cost_model.a ../autoscheduler/bin/cost_model/train_cost_model.a /home/yuka/Halide/distrib/bin/libHalide.so /home/yuka/Halide/distrib/tools/GenGen.cpp ../autoscheduler/bin/auto_schedule_runtime.a -o ./bin/libscheduling_tool.so -lz -lrt -ldl -ltinfo -lpthread -lm -lxml2
	g++ -O3 -std=c++11 -I /home/yuka/Halide/distrib/include/ -I /home/yuka/Halide/distrib/tools/ -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -rdynamic /home/yuka/Halide/apps/scheduling-tool/bin/libscheduling_tool.so -ldl -lpthread -lz /home/yuka/Halide/distrib/bin/libHalide.so -lz -lrt -ldl -ltinfo -lpthread -lm -lxml2 `libpng-config --cflags --ldflags` -ljpeg test/simple_test.cpp -o bin/simple_test

clean:
	rm -rf $(BIN)
