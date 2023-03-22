#
# MultiSense - Makefile
#

#
# Recurse into each of these directories.
#

add_subdirectory(LibMultiSense)
add_subdirectory(vr-interface)

if (${MULTISENSE_BUILD_UTILITIES})
    add_subdirectory(Utilities)
endif ()
