BIN=bin
default:
	@mkdir -p $(BIN)
	g++ -shared -rdynamic -fPIC -O3 -std=c++11 -I /home/yuka/Halide/distrib/include/ -I /home/yuka/Halide/distrib/tools/ -I /home/yuka/Halide/apps/autoscheduler -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -g -O3 -I ../autoscheduler/bin/cost_model ./SchedulingTool.cpp ../autoscheduler/ASLog.cpp ../autoscheduler/DefaultCostModel.cpp ../autoscheduler/Weights.cpp ../autoscheduler/bin/baseline_weights.o ../autoscheduler/bin/cost_model/cost_model.a ../autoscheduler/bin/cost_model/train_cost_model.a /home/yuka/Halide/distrib/bin/libHalide.so /home/yuka/Halide/distrib/tools/GenGen.cpp ../autoscheduler/bin/auto_schedule_runtime.a -o ./bin/libscheduling_tool.so -lz -lrt -ldl -ltinfo -lpthread -lm -lxml2
	g++ test/simple_test.cpp -I bin/host -I ~/Halide/distrib/include -L ~/Halide/distrib/lib -I /home/yuka/Halide/distrib/tools/ -lHalide -lpthread -ldl -lz -lrt -ltinfo -rdynamic -o bin/simple_test

clean:
	rm -rf $(BIN)
