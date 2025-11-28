set(metalchat_SDK macosx)
set(metalchat_TARGET "air64-apple-macos14.0")
set(metalchat_KERNEL_LIBRARY "metalchat.metallib")

# When the target system is iOS, choose a different target for the xrun command.
if (CMAKE_SYSTEM_NAME EQUAL iOS)
    set(metalchat_SDK iphoneos)
    set(metalchat_TARGET "air64-apple-ios18.0")
endif()


string(STRIP metalchat_TARGET ${metalchat_TARGET})

file(GLOB metalchat_KERNEL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB metalchat_KERNEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.metal)


foreach(metalchat_KERNEL_SOURCE ${metalchat_KERNEL_SOURCES})
    get_filename_component(metalchat_KERNEL_NAME ${metalchat_KERNEL_SOURCE} NAME_WE)

    add_custom_command(
        OUTPUT ${metalchat_KERNEL_NAME}.air
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/${metalchat_KERNEL_NAME}.metal
            ${COMMON_DEPENDENCIES}
            ${metalchat_KERNEL_HEADERS}
        COMMAND xcrun -sdk ${metalchat_SDK} metal
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

    list(APPEND metalchat_KERNELS ${metalchat_KERNEL_NAME}.air)
endforeach(metalchat_KERNEL_SOURCE)


add_custom_target(
    ${metalchat_KERNEL_LIBRARY}
    DEPENDS ${metalchat_KERNELS}
    ${metalchat_KERNEL_NAME}.air
    COMMAND xcrun -sdk ${metalchat_SDK} metallib
    *.air
    -o ${CMAKE_BINARY_DIR}/${metalchat_KERNEL_LIBRARY}
)


set_target_properties(${metalchat_KERNEL_LIBRARY} PROPERTIES BUNDLE TRUE)
set_target_properties(${metalchat_KERNEL_LIBRARY} PROPERTIES PREFIX "")
set_target_properties(${metalchat_KERNEL_LIBRARY} PROPERTIES SUFFIX ".metallib")
