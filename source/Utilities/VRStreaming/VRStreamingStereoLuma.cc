/* Example code to extract 8 bit image data from a image header and display it
using OpenCV (header.bitsPerPixel = 8)
\code{.cpp} */

#include "MultiSense/MultiSenseTypes.hh"
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <signal.h>
#include <string>
#include <unistd.h>
#include <eigen3/Eigen/Core>
#include <chrono>
#include <zmq.hpp>

#include <MultiSense/MultiSenseChannel.hh>

// Note this example has only been tested under Linux
#include <opencv2/opencv.hpp>
#include <unordered_map>

volatile bool doneG = false;

zmq::context_t context(1);
zmq::socket_t rgb_streaming_socket(context, ZMQ_PUSH);
zmq::socket_t stereo_streaming_socket(context, ZMQ_PUSH);

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
        void lumaCallback(const crl::multisense::image::Header& header);

    private:
        crl::multisense::Channel* m_channel;

        // store images from different callbacks to synchronize luma and rbg, left and right.
        std::unordered_map<crl::multisense::DataSource, std::shared_ptr<BufferWrapper<crl::multisense::image::Header>>> image_buffers_;
};

namespace {
    // Shim for the C-style callbacks accepted by
    // crl::mulisense::Channel::addIsolatedCallback
    void lumaCB(const crl::multisense::image::Header& header, void* userDataP)
    { reinterpret_cast<Camera*>(userDataP)->lumaCallback(header); }
}


Camera::Camera(crl::multisense::Channel* channel):
    m_channel(channel)
{
    crl::multisense::Status status;

    std::cout << "attaching callbacks" <<  std::endl;

    //
    // Attach our monoCallback to our Channel instance. It will get
    // called every time there is new Left Luma or Right luma image
    // data.
    status = m_channel->addIsolatedCallback(lumaCB,
                                           crl::multisense::Source_Luma_Left | crl::multisense::Source_Luma_Right, 
                                           this);
    // Check to see if the callback was successfully attached
    if(crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to attach isolated callback");
    }

    std::cout << "starting streams" <<  std::endl;
    // Start streaming luma images for the left and right cameras.
    m_channel->startStreams(crl::multisense::Source_Luma_Left | crl::multisense::Source_Luma_Right);

    // Check to see if the streams were sucessfully started
    if(crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to start image streams");
    }
}

Camera::~Camera()
{
    crl::multisense::Status status;

    // Remove our isolated callback.
    status = m_channel->removeIsolatedCallback(lumaCB);
    // Check to see if the callback was successfully removed
    if(crl::multisense::Status_Ok != status) {
        std::cout << "Unable to remove isolated callback" << std::endl;
    }

    // Stop streaming luma images for the left and right cameras
    m_channel->stopStreams(crl::multisense::Source_Luma_Left | crl::multisense::Source_Chroma_Left);

    //
    // Check to see if the image streams were successfully stopped
    if(crl::multisense::Status_Ok != status) {
        std::cout << "Unable to stop streams" << std::endl;
    }
}

// Saves the new luma image in the buffer and replaces the previous luma image. 
// The previous luma image will not be freed until there are no pointers pointing to it.
void Camera::lumaCallback(const crl::multisense::image::Header& header)
{
    //std::cout << "lumaCallback" << std::endl;
    if (crl::multisense::Source_Luma_Left != header.source &&
        crl::multisense::Source_Luma_Right != header.source)
    {
        std::cout << "Unexpected source: " << header.source << std::endl;
        return;
    }

    image_buffers_[header.source] = std::make_shared<BufferWrapper<crl::multisense::image::Header>>(m_channel, header);

    bool isLeft = header.source == crl::multisense::Source_Luma_Left;
    const auto luma = image_buffers_.find(isLeft ? crl::multisense::Source_Luma_Right : crl::multisense::Source_Luma_Left);
    if (luma == image_buffers_.end())
    {
        std::cout << "No luma image" << std::endl;
        return;
    }
    const auto luma_ptr = luma->second;
    if (header.frameId == luma_ptr->data().frameId)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat thisIm(header.height, header.width, CV_8UC1, (void*)header.imageDataP);
        cv::Mat otherIm(header.height, header.width, CV_8UC1, (void*)luma_ptr->data().imageDataP);
        cv::Mat combined;
        cv::hconcat(thisIm, otherIm, combined);
        cv::namedWindow("luma");
        cv::imshow("luma", combined);
        cv::Mat smaller;
        cv::resize(combined, smaller, cv::Size(800, 200));
        cv::imwrite("../images/" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) + ".png", smaller);
        auto start_wait = std::chrono::high_resolution_clock::now();
        cv::waitKey(1);
        auto end_wait = std::chrono::high_resolution_clock::now();
        auto img_duration = std::chrono::duration_cast<std::chrono::milliseconds>(start_wait - start);
        auto wait_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_wait - start_wait);
    std::cout << "Time taken by image processing: " << img_duration.count() << " ms, time taken by waiting is " << wait_duration.count() << " ms." << std::endl;
    }

}

int main()
{
    //
    // Setup a signal handler to kill the application
    signal(SIGINT, signalHandler);

    rgb_streaming_socket.bind("tcp://*:5557");
    rgb_streaming_socket.set(zmq::sockopt::conflate, 1);

    stereo_streaming_socket.bind("tcp://*:5558");
    stereo_streaming_socket.set(zmq::sockopt::conflate, 1);

    //
    // Instantiate a channel connecting to a sensor at the factory default
    // IP address
    crl::multisense::Channel* channel;
    channel = crl::multisense::Channel::Create("192.168.1.7");

    channel->setMtu(1500);

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

