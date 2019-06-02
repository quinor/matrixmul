all: cmake


.PHONY: all cmake clean


cmake: .build
	cmake --build .build

.build:
	mkdir -p .build && cd .build && cmake .. -DCMAKE_BUILD_TYPE=Release

clean:
	rm -rf .build

package:
	rm -rf wj359634 && mkdir -p wj359634 && cp -r src densematgen* Readme.txt wj359634/ && cp CMakeLists_remote.txt wj359634/CMakeLists.txt && cd wj359634 && zip -r ../wj359634.zip .
