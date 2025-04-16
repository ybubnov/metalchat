set(metalama_TARGET "air64-apple-macos14.0")
set(metalama_KERNEL_NAME "metalama")
set(metalama_KERNEL_LIBRARY "${metalama_KERNEL_NAME}.metallib")

string(STRIP metalama_TARGET ${metalama_TARGET})

include_directories()

file(GLOB metalama_KERNEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.metal)


add_custom_command(
    OUTPUT ${metalama_KERNEL_NAME}.air
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${metalama_KERNEL_NAME}.metal ${COMMON_DEPENDENCIES}
    COMMAND xcrun -v -sdk macosx metal
    ${METAL_FLAGS}
    -I .
    -O3
    -ffast-math
    -c
    -Wno-unused-variable
    -target ${metalama_TARGET}
    ${CMAKE_CURRENT_SOURCE_DIR}/${metalama_KERNEL_NAME}.metal
    -o ${metalama_KERNEL_NAME}.air
    VERBATIM
)


add_custom_target(
    ${metalama_KERNEL_LIBRARY}
    DEPENDS ${metalama_KERNEL_SOURCES}
    ${metalama_KERNEL_NAME}.air
    COMMAND xcrun -sdk macosx metallib
    *.air
    -o ${CMAKE_BINARY_DIR}/${metalama_KERNEL_LIBRARY}
)


set_target_properties(${metalama_KERNEL_LIBRARY} PROPERTIES BUNDLE TRUE)
set_target_properties(${metalama_KERNEL_LIBRARY} PROPERTIES PREFIX "")
set_target_properties(${metalama_KERNEL_LIBRARY} PROPERTIES SUFFIX ".metallib")
