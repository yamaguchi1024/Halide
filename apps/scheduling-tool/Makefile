BIN=bin
default:
	@mkdir -p $(BIN)
	g++ -shared -fPIC -O3 -std=c++11 -I /home/yuka/Halide/src -I /home/yuka/Halide/distrib/include/ -I /home/yuka/Halide/distrib/tools/ -I /home/yuka/Halide/src/autoschedulers/adams2019/ -I /home/yuka/Halide/src/autoschedulers/common/ -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -g -O3 -I ../../src/autoschedulers/adams2019/bin/cost_model ./SchedulingTool.cpp ../../src/autoschedulers/adams2019/ASLog.cpp ../../src/autoschedulers/adams2019/FunctionDAG.cpp ../../src/autoschedulers/adams2019/DefaultCostModel.cpp ../../src/autoschedulers/adams2019/Weights.cpp ../../src/autoschedulers/adams2019/bin/baseline_weights.o ../../src/autoschedulers/adams2019/bin/cost_model/cost_model.a ../../src/autoschedulers/adams2019/bin/cost_model/train_cost_model.a /home/yuka/Halide/distrib/lib/libHalide.so /home/yuka/Halide/distrib/tools/GenGen.cpp ../../src/autoschedulers/adams2019/bin/auto_schedule_runtime.a -o ./bin/libscheduling_tool.so -lz -lrt -ldl -ltinfo -lpthread -lm -lxml2
	g++ -O3 -std=c++11 -I /home/yuka/Halide/distrib/include/ -I /home/yuka/Halide/distrib/tools/ -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -rdynamic /home/yuka/Halide/apps/scheduling-tool/bin/libscheduling_tool.so -ldl -lpthread -lz /home/yuka/Halide/distrib/lib/libHalide.so -lz -lrt -ldl -ltinfo -lpthread -lm -lxml2 `libpng-config --cflags --ldflags` -ljpeg test/simple_test.cpp -o bin/simple_test

unsharp-manual:
	g++ -O3 -std=c++11 -I /home/yuka/Halide/distrib/include/ -I /home/yuka/Halide/distrib/tools/ -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -rdynamic /home/yuka/Halide/apps/scheduling-tool/bin/libscheduling_tool.so -ldl -lpthread -lz /home/yuka/Halide/distrib/lib/libHalide.so -lz -lrt -ldl -ltinfo -lpthread -lm -lxml2 `libpng-config --cflags --ldflags` -ljpeg test/unsharp_mask_manual.cpp -o bin/unsharp_mask_manual
	./bin/unsharp_mask_manual
unsharp:
	g++ -O3 -std=c++11 -I /home/yuka/Halide/distrib/include/ -I /home/yuka/Halide/distrib/tools/ -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -rdynamic /home/yuka/Halide/apps/scheduling-tool/bin/libscheduling_tool.so -ldl -lpthread -lz /home/yuka/Halide/distrib/lib/libHalide.so -lz -lrt -ldl -ltinfo -lpthread -lm -lxml2 `libpng-config --cflags --ldflags` -ljpeg test/unsharp_mask.cpp -o bin/unsharp_mask
	./bin/unsharp_mask

clean:
	rm -rf $(BIN)
