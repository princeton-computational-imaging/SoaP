/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 An object that connects the CameraController and the views.
 */

import Foundation
import SwiftUI
import Combine
import simd
import AVFoundation

final class MetalTextureContent {
    var texture: MTLTexture?
}

extension CVPixelBuffer {
    
    func texture(withFormat pixelFormat: MTLPixelFormat, planeIndex: Int, addToCache cache: CVMetalTextureCache) -> MTLTexture? {
        
        let width = CVPixelBufferGetWidthOfPlane(self, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(self, planeIndex)
        
        var cvtexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, self, nil, pixelFormat, width, height, planeIndex, &cvtexture)
        guard let texture = cvtexture else { return nil }
        return CVMetalTextureGetTexture(texture)
    }
    
}


class CameraManager: ObservableObject, CaptureDataReceiver {
    
    var capturedData: CameraCapturedData
    @Published var isFilteringDepth: Bool {
        didSet {
            controller.isFilteringEnabled = isFilteringDepth
        }
    }
    @Published var orientation = UIDevice.current.orientation
    
    var fpsArray = Array(repeating: 0.0, count: 30)
    var fpsCount = 0
    var timePrev = 0.0
    
    var controller: CameraController
    var cancellables = Set<AnyCancellable>()
    var session: AVCaptureSession { controller.captureSession }
    
    @Published var iso : Float = 0
    @Published var exposureTime : Double = 0
    @Published var frameCount = 99999
    @Published var savingState = 0
    
    init() {
        // Create an object to store the captured data for the views to present.
        capturedData = CameraCapturedData(depth: nil, color: nil, timestamp: 0)
        controller = CameraController()
        controller.isFilteringEnabled = true
        controller.startStream()
        isFilteringDepth = controller.isFilteringEnabled
        
        NotificationCenter.default.publisher(for: UIDevice.orientationDidChangeNotification).sink { _ in
            self.orientation = UIDevice.current.orientation
        }.store(in: &cancellables)
        controller.delegate = self
    }
    
    func resumeStream() {
        controller.startStream()
    }
    
    func onNewPhotoData(capturedData: CameraCapturedData) {
        // Because the views hold a reference to `capturedData`, the app updates each texture separately.
        self.capturedData.depthContent.texture = capturedData.depth
        self.capturedData.colorRGBContent.texture = capturedData.color
        self.capturedData.timestamp = capturedData.timestamp
        
        if capturedData.timestamp != nil && 1.0/(capturedData.timestamp! - self.timePrev) < 10000 { // skip double-frames
            self.fpsCount += 1
            self.fpsArray[self.fpsCount % self.fpsArray.count] = 1.0/(capturedData.timestamp! - self.timePrev)
//            print("Current FPS: ", self.fpsArray.reduce(0.0, +)/(Double(self.fpsArray.count)))
            self.timePrev = capturedData.timestamp!
        }
        
        DispatchQueue.main.async { // Hacky, for printing to UI
            self.iso = self.controller.device.iso
            self.exposureTime = self.controller.device.exposureDuration.seconds
            self.frameCount = self.controller.frameCount
            self.savingState = self.controller.savingState
            
        }
    }
    
    func onNewData(capturedData: CameraCapturedData) {
        // do nothing
    }
    
}

class CameraCapturedData {
    
    var depth: MTLTexture?
    var depthContent: MetalTextureContent
    var color: MTLTexture?
    var colorRGBContent: MetalTextureContent
    var timestamp: Double?
    
    init(depth: MTLTexture?,
         color: MTLTexture?,
         timestamp: Double?) {
        
        self.depth = depth
        self.depthContent = MetalTextureContent()
        self.depthContent.texture = depth
        self.color = color
        self.colorRGBContent = MetalTextureContent()
        self.colorRGBContent.texture = color
        self.timestamp = timestamp
    }
}
