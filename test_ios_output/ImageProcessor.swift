
// Image Processing Utilities for PicoTuri-EditJudge

import CoreML
import UIKit
import Accelerate

@available(iOS 15.0, *)
class ImageProcessor {

    static func preprocessImage(_ image: UIImage,
                              targetSize: CGSize = CGSize(width: 512, height: 512)) -> MLMultiArray? {
        guard let cgImage = image.cgImage else { return nil }

        // Resize image using Accelerate framework
        let resizedImage = resizeCGImage(cgImage, to: targetSize)

        // Convert to RGB pixel buffer
        guard let pixelBuffer = pixelBufferFromCGImage(resizedImage) else { return nil }

        // Create MLMultiArray
        guard let multiArray = try? MLMultiArray(shape: [1, 3, 512, 512], dataType: .float32) else {
            return nil
        }

        // Copy pixel data (RGB format: channels x height x width)
        copyPixelBufferToMultiArray(pixelBuffer, multiArray: multiArray)

        return multiArray
    }

    private static func resizeCGImage(_ cgImage: CGImage, to size: CGSize) -> CGImage {
        let context = CGContext(data: nil, width: Int(size.width), height: Int(size.height),
                               bitsPerComponent: 8, bytesPerRow: 0,
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.draw(cgImage, in: CGRect(origin: .zero, size: size))
        return context!.makeImage()!
    }

    private static func pixelBufferFromCGImage(_ cgImage: CGImage) -> CVPixelBuffer? {
        let options = [kCVPixelBufferCGImageCompatibilityKey: true,
                      kCVPixelBufferCGBitmapContextCompatibilityKey: true]

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, cgImage.width, cgImage.height,
                                       kCVPixelFormatType_32ARGB, options as CFDictionary, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                               width: cgImage.width, height: cgImage.height,
                               bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))

        return buffer
    }

    private static func copyPixelBufferToMultiArray(_ pixelBuffer: CVPixelBuffer, multiArray: MLMultiArray) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        let pointer = baseAddress.assumingMemoryBound(to: UInt8.self)

        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = y * bytesPerRow + x * 4
                let blue = Float(pointer[pixelIndex]) / 255.0
                let green = Float(pointer[pixelIndex + 1]) / 255.0
                let red = Float(pointer[pixelIndex + 2]) / 255.0

                // RGB to tensor format [1, 3, H, W]
                let redIndex = [0, 0, y, x] as [NSNumber]
                let greenIndex = [0, 1, y, x] as [NSNumber]
                let blueIndex = [0, 2, y, x] as [NSNumber]

                multiArray[redIndex] = NSNumber(value: red)
                multiArray[greenIndex] = NSNumber(value: green)
                multiArray[blueIndex] = NSNumber(value: blue)
            }
        }
    }
}
