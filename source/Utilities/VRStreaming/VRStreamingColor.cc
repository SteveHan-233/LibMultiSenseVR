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
#include <unistd.h>
#include <eigen3/Eigen/Core>

#include <MultiSense/MultiSenseChannel.hh>

// Note this example has only been tested under Linux
#include <opencv2/opencv.hpp>
#include <unordered_map>

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
        void chromaCallback(const crl::multisense::image::Header& header);
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
    void chromaCB(const crl::multisense::image::Header& header, void* userDataP)
    { reinterpret_cast<Camera*>(userDataP)->chromaCallback(header); }
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
    status = m_channel->addIsolatedCallback(chromaCB,
                                           crl::multisense::Source_Chroma_Left | crl::multisense::Source_Chroma_Right, 
                                           this);
    // Check to see if the callback was successfully attached
    if(crl::multisense::Status_Ok != status) {
        throw std::runtime_error("Unable to attach isolated callback");
    }

    std::cout << "starting streams" <<  std::endl;
    // Start streaming luma images for the left and right cameras.
    m_channel->startStreams(crl::multisense::Source_Luma_Left | crl::multisense::Source_Chroma_Left | crl::multisense::Source_Luma_Right | crl::multisense::Source_Chroma_Right);

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

    status = m_channel->removeIsolatedCallback(chromaCB);
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
    std::cout << "lumaCallback" << std::endl;
    if (crl::multisense::Source_Luma_Left != header.source &&
        crl::multisense::Source_Luma_Right != header.source)
    {
        std::cout << "Unexpected source: " << header.source << std::endl;
        return;
    }

    if (header.source == crl::multisense::Source_Luma_Right) {
        std::cout << "Right Luma" << std::endl;
    }

    image_buffers_[header.source] = std::make_shared<BufferWrapper<crl::multisense::image::Header>>(m_channel, header);
}

// Finds the corresponding Luma image and if it is the same frame, converts the chroma image to BGR and displays it.
void Camera::chromaCallback(const crl::multisense::image::Header& header)
{
    // The left-luma image is currently published before the matching chroma image so this can just trigger on that

    std::cout << "chromaCallback" << std::endl;
    if (crl::multisense::Source_Chroma_Left != header.source &&
        crl::multisense::Source_Chroma_Right != header.source)
    {
        std::cout << "Unexpected source: " << header.source << std::endl;
        return;
    }

    if (header.source == crl::multisense::Source_Chroma_Right) {
        std::cout << "Right Chroma" << std::endl;
    }

    bool isLeft = header.source == crl::multisense::Source_Chroma_Left;
    if (!isLeft) {
        std::cout << "Right image" << std::endl;
    }
    const auto luma = image_buffers_.find(isLeft ? crl::multisense::Source_Luma_Left : crl::multisense::Source_Luma_Right);
    if (luma == image_buffers_.end())
    {
        std::cout << "No luma image" << std::endl;
        return;
    }

    const auto luma_ptr = luma->second;

    if (header.frameId == luma_ptr->data().frameId)
    {
        const uint32_t height = luma_ptr->data().height;
        const uint32_t width = luma_ptr->data().width;
        std::cout << "luma resolution " << luma_ptr->data().width << "x" <<  luma_ptr->data().height << " and chroma resolution " << header.width << "x" << header.height << std::endl;
        std::cout << "luma length " << luma_ptr->data().imageLength << " and chroma length " << header.imageLength << std::endl;
        //std::vector<uint8_t> lumaImage(width * height); // for some reason it's different from header.imageLength
        //memcpy(&(lumaImage[0]), luma_ptr->data().imageDataP, luma_ptr->data().imageLength);
        //cv::Mat lumaMat(luma_ptr->data().height, luma_ptr->data().width, CV_8UC1, &(lumaImage[0]));
        //cv::namedWindow("luma");
        //cv::imshow("luma", lumaMat);
        //cv::waitKey(1);
        
        // Create a container for the image data
        std::vector<uint8_t> imageData(width * height * 3); // for some reason it's different from header.imageLength

        // Convert YCbCr 4:2:0 to RGB
        ycbcrToBgr(luma_ptr->data(), header, &imageData[0]);
        // Create a OpenCV matrix using our image container
        cv::Mat imageMat(height, width, CV_8UC3, &(imageData[0]));
        //cv::Mat converted;
        //cvtColor(imageMat, converted, cv::COLOR_BGR2RGB);

        // Display the image using OpenCV
        cv::namedWindow(isLeft ? "left" : "right");
        cv::imshow((isLeft ? "left" : "right"), imageMat);
        cv::waitKey(1);
    } 
    else
    {
        std::cout << "Chroma image is not from the same frame as the luma image" << std::endl;
    }

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
    channel = crl::multisense::Channel::Create("192.168.1.7");

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
