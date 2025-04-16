#include <metalchat/device.h>


int
some()
{
    metalchat::device gpu0("metalchat.metallib");
    std::cout << "device = " << gpu0.name() << std::endl;
    return 0;
}
