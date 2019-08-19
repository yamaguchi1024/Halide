BIN=bin
default:
	@mkdir -p $(BIN)
	g++ -shared -rdynamic -fPIC -O3 -std=c++11 -I /home/yuka/Halide/distrib/include/ -I /home/yuka/Halide/distrib/tools/ -I /home/yuka/Halide/apps/autoscheduler -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -g -I ../autoscheduler/bin/cost_model ./SchedulingTool.cpp ../autoscheduler/DefaultCostModel.cpp ../autoscheduler/bin/weights_head1_conv1_weight.o ../autoscheduler/bin/weights_head1_conv1_bias.o ../autoscheduler/bin/weights_head2_conv1_weight.o ../autoscheduler/bin/weights_head2_conv1_bias.o ../autoscheduler/bin/weights_trunk_conv1_weight.o ../autoscheduler/bin/weights_trunk_conv1_bias.o ../autoscheduler/bin/cost_model/cost_model.a ../autoscheduler/bin/cost_model/train_cost_model.a  ../autoscheduler/bin/auto_schedule_runtime.a -O3 -o ./bin/libscheduling_tool.so -lz -lrt -ldl -ltinfo -lpthread -lm -lxml2

clean:
	rm -rf $(BIN)
