BUILD := build

.PHONY: build bench run clean

build:
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release -Wno-dev
	cmake --build $(BUILD) --parallel

bench: build
	./$(BUILD)/benchmark

run: build
	./$(BUILD)/rtos_scheduler

clean:
	rm -rf $(BUILD)
