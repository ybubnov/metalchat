set(metalchat_TARGET "air64-apple-macos14.0")
set(metalchat_KERNEL_NAME "metalchat")
set(metalchat_KERNEL_LIBRARY "${metalchat_KERNEL_NAME}.metallib")

string(STRIP metalchat_TARGET ${metalchat_TARGET})

file(GLOB metalchat_KERNEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.metal)


add_custom_command(
    OUTPUT ${metalchat_KERNEL_NAME}.air
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${metalchat_KERNEL_NAME}.metal ${COMMON_DEPENDENCIES}
    COMMAND xcrun -v -sdk macosx metal
    ${METAL_FLAGS}
    -I .
    -I ${PROJECT_SOURCE_DIR}/include
    -O3
    -ffast-math
    -c
    -Wno-unused-variable
    -target ${metalchat_TARGET}
    ${CMAKE_CURRENT_SOURCE_DIR}/${metalchat_KERNEL_NAME}.metal
    -o ${metalchat_KERNEL_NAME}.air
    VERBATIM
)


add_custom_target(
    ${metalchat_KERNEL_LIBRARY}
    DEPENDS ${metalchat_KERNEL_SOURCES}
    ${metalchat_KERNEL_NAME}.air
    COMMAND xcrun -sdk macosx metallib
    *.air
    -o ${CMAKE_BINARY_DIR}/${metalchat_KERNEL_LIBRARY}
)


set_target_properties(${metalchat_KERNEL_LIBRARY} PROPERTIES BUNDLE TRUE)
set_target_properties(${metalchat_KERNEL_LIBRARY} PROPERTIES PREFIX "")
set_target_properties(${metalchat_KERNEL_LIBRARY} PROPERTIES SUFFIX ".metallib")
