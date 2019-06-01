all: cmake


.PHONY: all cmake clean


cmake: .build
	cmake --build .build

.build:
	cmake . -B .build -DCMAKE_BUILD_TYPE=Release

clean:
	rm -rf .build
