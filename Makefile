all: cmake


.PHONY: all cmake clean


cmake: .build
	cmake --build .build

.build:
	mkdir -p .build && cd .build && cmake .. -DCMAKE_BUILD_TYPE=Release

clean:
	rm -rf .build
