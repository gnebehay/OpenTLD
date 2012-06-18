set(MAIN_SOURCES config.cpp gui.cpp Main.cpp settings.cpp qtconfiggui.cpp)

#Compile
add_library(main ${MAIN_SOURCES})

include_directories(../imacq ../tld ../config ${OpenCV_INCLUDE_DIRS})
