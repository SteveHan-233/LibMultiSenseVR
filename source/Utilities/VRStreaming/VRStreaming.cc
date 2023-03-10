/* Example code to extract 8 bit image data from a image header and display it
using OpenCV (header.bitsPerPixel = 8)
\code{.cpp} */

#include "MultiSense/MultiSenseTypes.hh"
#include <iostream>
#include <stdexcept>
#include <signal.h>
#include <unistd.h>
#include <eigen3/Eigen/Core>

#include <MultiSense/MultiSenseChannel.hh>

// Note this example has only been tested under Linux
#include <opencv2/opencv.hpp>

volatile bool doneG = false;

void signalHandler(int sig)
{
    (void) sig;
    doneG = true;
}

template <typename T>
class BufferWrapper
{
public:
    BufferWrapper(crl::multisense::Channel* driver,
                 const T &data):
        driver_(driver),
        callback_buffer_(driver->reserveCallbackBuffer()),
        data_(data)
    {
    }

    ~BufferWrapper()
    {
        driver_->releaseCallbackBuffer(callback_buffer_);
    }

    const T &data() const noexcept
    {
        return data_;
    }

private:

    BufferWrapper(const BufferWrapper&) = delete;
    BufferWrapper operator=(const BufferWrapper&) = delete;

    crl::multisense::Channel * driver_;
    void* callback_buffer_;
    const T data_;

};

template <typename T>
constexpr Eigen::Matrix<T, 3, 1> ycbcrToBgr(const crl::multisense::image::Header &luma,
                                            const crl::multisense::image::Header &chroma,
                                            size_t u,
                                            size_t v)
{
    const uint8_t *lumaP = reinterpret_cast<const uint8_t*>(luma.imageDataP);
    const uint8_t *chromaP = reinterpret_cast<const uint8_t*>(chroma.imageDataP);

    const size_t luma_offset = (v * luma.width) + u;
    const size_t chroma_offset = 2 * (((v/2) * (luma.width/2)) + (u/2));

    const float px_y = static_cast<float>(lumaP[luma_offset]);
    const float px_cb = static_cast<float>(chromaP[chroma_offset+0]) - 128.0f;
    const float px_cr = static_cast<float>(chromaP[chroma_offset+1]) - 128.0f;

    float px_r  = px_y + 1.402f   * px_cr;
    float px_g  = px_y - 0.34414f * px_cb - 0.71414f * px_cr;
    float px_b  = px_y + 1.772f   * px_cb;

    if (px_r < 0.0f)        px_r = 0.0f;
    else if (px_r > 255.0f) px_r = 255.0f;
    if (px_g < 0.0f)        px_g = 0.0f;
    else if (px_g > 255.0f) px_g = 255.0f;
    if (px_b < 0.0f)        px_b = 0.0f;
    else if (px_b > 255.0f) px_b = 255.0f;

    return Eigen::Matrix<T, 3, 1>{static_cast<T>(px_b), static_cast<T>(px_g), static_cast<T>(px_r)};
}

void ycbcrToBgr(const crl::multisense::image::Header &luma,
                const crl::multisense::image::Header &chroma,
                uint8_t *output)
{
    const size_t rgb_stride = luma.width * 3;

    for(uint32_t y=0; y< luma.height; ++y)
    {
        const size_t row_offset = y * rgb_stride;

        for(uint32_t x=0; x< luma.width; ++x)
        {
            memcpy(output + row_offset + (3 * x), ycbcrToBgr<uint8_t>(luma, chroma, x, y).data(), 3);
        }
    }
}


class Camera
{
    public:
        Camera(crl::multisense::Channel* channel);
        ~Camera();
        void imageCallback(const crl::multisense::image::Header& header);

    private:
        crl::multisense::Channel* m_channel;
};

namespace {
    //
    // Shim for the C-style callbacks accepted by
    // crl::mulisense::Channel::addIsolatedCallback
    void monoCallback(const crl::multisense::image::Header& header, void* userDataP)
    { reinterpret_cast<Camera*>(userDataP)->imageCallback(header); }
}


Camera::Camera(crl::multisense::Channel* channel):
    m_channel(channel)
{
    crl::multisense::Status status;

    //
    // Attach our monoCallback to our Channel instance. It will get
    // called every time there is new Left Luma or Right luma image
    // data.
    status = m_channel->addIsolatedCallback(monoCallback,
                                           crl::multisense::Source_Rgb_Left, 
                                           this);

    //
    // Check to see if the callback was successfully attached
    if(crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to attach isolated callback");
    }

    //
    // Start streaming luma images for the left and right cameras.
    m_channel->startStreams(crl::multisense::Source_Rgb_Left);

    //
    // Check to see if the streams were sucessfully started
    if(crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to start image streams");
    }
}

Camera::~Camera()
{
    crl::multisense::Status status;

    //
    // Remove our isolated callback.
    status = m_channel->removeIsolatedCallback(monoCallback);

    //
    // Check to see if the callback was successfully removed
    if(crl::multisense::Status_Ok != status) {
        std::cout << "Unable to remove isolated callback" << std::endl;
    }

    //
    // Stop streaming luma images for the left and right cameras
    m_channel->stopStreams(crl::multisense::Source_Luma_Left | crl::multisense::Source_Luma_Right);

    //
    // Check to see if the image streams were successfully stopped
    if(crl::multisense::Status_Ok != status) {
        std::cout << "Unable to stop streams" << std::endl;
    }
}

void Camera::imageCallback(const crl::multisense::image::Header& header)
{
    std::cout << "hey" << std::endl;
    //
    // Create a container for the image data
    std::vector<uint8_t> imageData;
    imageData.resize(header.imageLength);

    //
    // Copy image data from the header's image data pointer to our
    // image container
    memcpy(&(imageData[0]), header.imageDataP, header.imageLength);

    //
    // Create a OpenCV matrix using our image container
    cv::Mat_<uint8_t> imageMat(header.height, header.width, &(imageData[0]));

    //
    // Display the image using OpenCV
    cv::namedWindow("Example");
    cv::imshow("Example", imageMat);
    cv::waitKey(1000./header.framesPerSecond/2);
}


int main()
{
    //
    // Setup a signal handler to kill the application
    signal(SIGINT, signalHandler);

    //
    // Instantiate a channel connecting to a sensor at the factory default
    // IP address
    crl::multisense::Channel* channel;
    channel = crl::multisense::Channel::Create("10.66.171.21");

    channel->setMtu(7200);

    try
    {
        Camera camera(channel);
        while(!doneG)
        {
            usleep(100000);
        }
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    //
    // Destroy the channel instance
    crl::multisense::Channel::Destroy(channel);
}

