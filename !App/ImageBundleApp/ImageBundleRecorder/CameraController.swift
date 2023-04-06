/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 An object that configures and manages the capture pipeline to stream video and LiDAR depth data.
 */

import Foundation
import AVFoundation
import CoreImage
import CoreMotion

protocol CaptureDataReceiver: AnyObject {
    func onNewData(capturedData: CameraCapturedData)
    func onNewPhotoData(capturedData: CameraCapturedData)
}

class CameraController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    enum ConfigurationError: Error {
        case lidarDeviceUnavailable
        case requiredFormatUnavailable
    }
    
    private let preferredWidthResolution = 4032
    
    private(set) var captureSession: AVCaptureSession!
    
    private let videoQueue = DispatchQueue(label: "com.example.apple-samplecode.VideoQueue", qos: .userInteractive)
    
    private var photoOutput: AVCapturePhotoOutput!
    private var depthDataOutput: AVCaptureDepthDataOutput!
    private var videoDataOutput: AVCaptureVideoDataOutput!
    private var outputVideoSync: AVCaptureDataOutputSynchronizer!
    private let metalDevice: MTLDevice?
    private var timer: Timer?
    private var motion: CMMotionManager!
    public var device: AVCaptureDevice!
    public var savingState = 0 // 0 - not saving, 1 - saving, 2 - error
    public var frameCount = 99999
    public var bundleSize = 42
    public var convertedDepth: AVDepthData!
    public var recordScene = false
    
    public var saveSuffix: String!
    public var rawFrameTimes: [Double] = []
    public var rgbFrameTimes: [Double] = []
    public var motionURL: URL!
    public var motionData: Data!
    public var imageRGBData: Data!
    public var imageRGBURL: URL!
    public var depthData: Data!
    public var depthURL: URL!
    public var imageRAWData: Data!
    public var imageRAWURL: URL!
    
    @Published var bundleFolder : URL?
    
    
    private var textureCache: CVMetalTextureCache!
    
    weak var delegate: CaptureDataReceiver?
    
    var isFilteringEnabled = true
    
    override init() {
        
        // create a texture cache to hold sample buffer textures
        metalDevice = MTLCreateSystemDefaultDevice()
        CVMetalTextureCacheCreate(nil,
                                  nil,
                                  metalDevice!,
                                  nil,
                                  &textureCache)
        
        super.init()
        
        do {
            try setupSession()
        } catch {
            fatalError("Unable to configure the capture session.")
        }
        
        
    }
    
    private func setupSession() throws {
        captureSession = AVCaptureSession()
        
        // configure the capture session
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .photo
        
        try setupCaptureInput()
        setupCaptureOutputs()
        
        // finalize capture session configuration
        captureSession.commitConfiguration()
    }
    
    // MARK: Init Bundle
    private func initBundleFolder(suffix: String = "") {
        let currDate = Date()
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let currDateString = dateFormatter.string(from : currDate)
        
        let DocumentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let DirPath = DocumentDirectory.appendingPathComponent("bundle-" + currDateString + suffix + "/")
        
        do {
            try FileManager.default.createDirectory(atPath: DirPath.path, withIntermediateDirectories: true, attributes: nil)
        } catch let error as NSError {
            print("Unable to create directory \(error.debugDescription)")
        }
        
        bundleFolder = URL(fileURLWithPath: DirPath.path)
    }
    
    public func recordMotionBundle(saveSuffix: String = ""){
        self.saveSuffix = saveSuffix
        
        recordScene = false
        
        motionData = Data.init()
        rawFrameTimes = []
        rgbFrameTimes = []
        frameCount = 0
        
        capturePhoto()
    }
    
    public func recordBundle(saveSuffix: String = ""){
        self.saveSuffix = saveSuffix
        
        recordScene = true
        
        motionData = Data.init()
        imageRGBData = Data.init()
        imageRAWData = Data.init()
        depthData = Data.init()
        rawFrameTimes = []
        rgbFrameTimes = []
        frameCount = 0
        
        capturePhoto()
    }
    
    
    // MARK: Start Motion Capture
    private func startMotionCapture() {
        self.motion = CMMotionManager()
        
        if self.motion.isDeviceMotionAvailable { self.motion!.deviceMotionUpdateInterval = 1.0 / 200.0 // ask for 200Hz but max frequency is 100Hz for 14pro
            self.motion.showsDeviceMovementDisplay = true
            // get the attitude relative to the magnetic north reference frame
            self.motion.startDeviceMotionUpdates(using: .xArbitraryZVertical,
                                                 to: OperationQueue(), withHandler: { (data, error) in
                // make sure the data is valid before accessing it
                if let validData = data {
                    
                    let timestamp = validData.timestamp
                    
                    let attitude = validData.attitude
                    let quaternion = validData.attitude.quaternion
                    let rotationRate = validData.rotationRate
                    let userAcceleration = validData.userAcceleration
                    let gravity = validData.gravity
                    
                    // generate header information to parse later in python
                    var header = """
                    <BEGINHEADER>
                    frameCount:\(String(describing: self.frameCount)),timestamp:\(String(describing: timestamp)),
                    quaternionX:\(String(describing: quaternion.x)),quaternionY:\(String(describing: quaternion.y)),
                    quaternionZ:\(String(describing: quaternion.z)),quaternionW:\(String(describing: quaternion.w)),
                    rotationRateX:\(String(describing: rotationRate.x)),rotationRateY:\(String(describing: rotationRate.y)),
                    rotationRateZ:\(String(describing: rotationRate.z)),roll:\(String(describing: attitude.roll)),
                    pitch:\(String(describing: attitude.pitch)),yaw:\(String(describing: attitude.yaw)),
                    userAccelerationX:\(String(describing: userAcceleration.x)),userAccelerationY:\(String(describing: userAcceleration.y)),
                    userAccelerationZ:\(String(describing: userAcceleration.z)),gravityX:\(String(describing: gravity.x)),
                    gravityY:\(String(describing: gravity.y)),gravityZ:\(String(describing: gravity.z))
                    <ENDHEADER>
                    """
                    header = header.components(separatedBy: .whitespacesAndNewlines).joined() // remove newlines
                    let encodedHeader = [UInt8](header.utf8)
                    
                    if self.motionData != nil && self.frameCount != 99999 {
                        self.motionData.append(encodedHeader, count: header.utf8.count)
                    }
                }
            })
        }
    }
    
    // MARK: Set Up Capture
    private func setupCaptureInput() throws {
        
        self.startMotionCapture()
        
        // LiDAR + main wide lens
        self.device = AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)
        
        guard let format = (self.device.formats.last { format in
            format.formatDescription.dimensions.width == preferredWidthResolution &&
            format.formatDescription.mediaSubType.rawValue == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange &&
            !format.isVideoBinned &&
            !format.supportedDepthDataFormats.isEmpty
        }) else {
            print("No such image format.")
            throw ConfigurationError.requiredFormatUnavailable
        }
        
        guard let depthFormat = (format.supportedDepthDataFormats.last { depthFormat in
            depthFormat.formatDescription.mediaSubType.rawValue == kCVPixelFormatType_DepthFloat16
        }) else {
            print("No such depth format.")
            throw ConfigurationError.requiredFormatUnavailable
        }
        
        // begin the device configuration
        try self.device.lockForConfiguration()
        
        // configure the device and depth formats
        self.device.activeFormat = format
        self.device.activeDepthDataFormat = depthFormat
        self.device.focusMode = .continuousAutoFocus
        self.device.activeVideoMaxFrameDuration = CMTimeMake(value: 1, timescale: 30) // 30 fps
        self.device.activeVideoMinFrameDuration = CMTimeMake(value: 1, timescale: 30) // 30 fps
        self.device.activeDepthDataMinFrameDuration = CMTimeMake(value: 1, timescale: 30) // 30 fps
        
        // finish the device configuration
        self.device.unlockForConfiguration()
        
        print("Selected video format: \(self.device.activeFormat)")
        print("Selected depth format: \(String(describing: self.device.activeDepthDataFormat))")
        
        // add a device input to the capture session
        let deviceInput = try AVCaptureDeviceInput(device: self.device)
        captureSession.addInput(deviceInput)
    }
    
    private func setupCaptureOutputs() {
        // create an object to output video sample buffers
        videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as String): NSNumber(value: 1111970369), // BGRA stream
                                         (kCVPixelBufferWidthKey as String): NSNumber(value: 1920),
                                         (kCVPixelBufferHeightKey as String): NSNumber(value: 1440)]
        captureSession.addOutput(videoDataOutput)
        
        // create an object to output depth data.
        depthDataOutput = AVCaptureDepthDataOutput()
        depthDataOutput.isFilteringEnabled = true
        captureSession.addOutput(depthDataOutput)
        
        
        // create an object to synchronize the delivery of depth and video data
        outputVideoSync = AVCaptureDataOutputSynchronizer(dataOutputs: [depthDataOutput, videoDataOutput])
        outputVideoSync.setDelegate(self, queue: videoQueue)
        
        // enable camera intrinsics matrix delivery
        guard let outputConnection = videoDataOutput.connection(with: .video) else { return }
        if outputConnection.isCameraIntrinsicMatrixDeliverySupported {
            outputConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
        }
        
        // create an object to output photos
        photoOutput = AVCapturePhotoOutput()
        captureSession.addOutput(photoOutput)
        photoOutput.maxPhotoQualityPrioritization = .speed
        photoOutput.isAppleProRAWEnabled = false // if true, captures are extremely slow as they stitch/process images
        photoOutput.maxPhotoDimensions = .init(width: 8064, height: 6048) // only gives 4k even if you ask for 8k unless you set proraw true
        
        // enable delivery of depth data after adding the output to the capture session
        photoOutput.isDepthDataDeliveryEnabled = true
    }
    
    func startStream() {
        captureSession.startRunning()
    }
    
    func stopStream() {
        captureSession.stopRunning()
    }
}

// MARK: Synchronized RGB and Depth
extension CameraController: AVCaptureDataOutputSynchronizerDelegate {
    
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        
        // retrieve the synchronized depth and sample buffer container objects
        guard let syncedDepthData = synchronizedDataCollection.synchronizedData(for: depthDataOutput) as? AVCaptureSynchronizedDepthData,
              let syncedVideoData = synchronizedDataCollection.synchronizedData(for: videoDataOutput) as? AVCaptureSynchronizedSampleBufferData else { return }
        
        guard let pixelBuffer = syncedVideoData.sampleBuffer.imageBuffer else { return }
        
        let timestamp = syncedDepthData.timestamp.seconds
        self.convertedDepth = syncedDepthData.depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat16)
        var data: CameraCapturedData!
        
        if (self.frameCount != 99999 && self.recordScene) || (self.recordScene && self.rawFrameTimes.contains(timestamp)){
            // if long-burst being recorded, write data
            self.writeImageBGRA(sampleBuffer: syncedVideoData.sampleBuffer, timestamp: timestamp, frameCount: self.frameCount)
            self.writeDepth(depthData: syncedDepthData.depthData, timestamp: timestamp, frameCount: self.frameCount)
            self.rgbFrameTimes.append(round(timestamp * 1000) / 1000.0)
        }
        data = CameraCapturedData(depth: self.convertedDepth.depthDataMap.texture(withFormat: .r16Float, planeIndex: 0, addToCache: textureCache),
                                  color: pixelBuffer.texture(withFormat: .bgra8Unorm, planeIndex: 0, addToCache: textureCache),
                                  timestamp: timestamp)
        
        
        delegate?.onNewPhotoData(capturedData: data)
    }
}


extension CameraController: AVCapturePhotoCaptureDelegate {
    
    // MARK: Capture Photo
    func capturePhoto() {
        var photoSettings: AVCapturePhotoSettings
        
        
        // MARK: Terminate Recording
        if self.frameCount == self.bundleSize {
            
            // delay so we catch last RGB/depth pair if it's delayed
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                self.frameCount = 99999
                print("Resetting camera back to autoexposure.")
                
                do{
                    try self.device.lockForConfiguration()
                } catch {
                    fatalError("Device could not be locked.")
                }
                
                self.device.exposureMode = .continuousAutoExposure
                self.device.focusMode = .continuousAutoFocus
                self.device.unlockForConfiguration()
                
                print("Writing to disk.")
                self.savingState = 1
            }
            
            // delay more so UI catches the 'savingData' change
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [self] in
                
                var missingTimes : [Double] = []
                for elem in self.rawFrameTimes {
                    if !self.rgbFrameTimes.contains(elem){
                        missingTimes.append(elem)
                    }
                }
                
                if missingTimes.count > 0 || self.rgbFrameTimes.count < self.bundleSize {
                    // something broke, missing synced frames
                    print("Missing times: ", missingTimes)
                    
                    self.motionData = nil
                    self.imageRGBData = nil
                    self.imageRAWData = nil
                    self.depthData = nil
                    
                    self.savingState = 2 // error
                    DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                        self.savingState = 0 // clear error in 5 seconds
                    }
                    
                    return
                }
                
                // make folders to save files to
                if self.recordScene {
                    if self.saveSuffix != "" {
                        self.initBundleFolder(suffix: "-" + self.saveSuffix)
                    } else {
                        self.initBundleFolder()
                    }
                    
                    print("Recording bundle into \(String(describing: self.bundleFolder!.path))")
                    
                    self.motionURL = URL(fileURLWithPath: "motion", relativeTo: self.bundleFolder).appendingPathExtension("bin")
                    self.imageRGBURL = URL(fileURLWithPath: "imageRGB", relativeTo: self.bundleFolder).appendingPathExtension("bin")
                    self.imageRAWURL = URL(fileURLWithPath: "imageRAW", relativeTo: self.bundleFolder).appendingPathExtension("bin")
                    self.depthURL = URL(fileURLWithPath: "depth", relativeTo: self.bundleFolder).appendingPathExtension("bin")
                    
                } else { // motion bundle
                    if self.saveSuffix != "" {
                        self.initBundleFolder(suffix: "-" + self.saveSuffix + "-motion")
                    } else {
                        self.initBundleFolder(suffix: "-motion")
                    }
                    
                    print("Recording motion into \(String(describing: self.bundleFolder!.path))")
                    
                    self.motionURL = URL(fileURLWithPath: "motion", relativeTo: self.bundleFolder).appendingPathExtension("bin")
                }
                
                
                try? self.motionData.write(to: self.motionURL)
                
                // record to disk
                if self.recordScene {
                    try? self.imageRGBData.write(to: self.imageRGBURL)
                    try? self.imageRAWData.write(to: self.imageRAWURL)
                    try? self.depthData.write(to: self.depthURL)
                }
                
                self.recordScene = false
                self.savingState = 0
                
                // clear memory
                self.motionData = nil
                self.imageRGBData = nil
                self.imageRAWData = nil
                self.depthData = nil
                
                print("Done recording bundle.")
            }
            return
            
        } else if self.frameCount >= self.bundleSize {
            self.frameCount = 99999
            return // don't record past bundle size
        }
        
        if  photoOutput.availableRawPhotoPixelFormatTypes.count > 0 {
            
            for format in photoOutput.availableRawPhotoPixelFormatTypes {
                print(format)
            }
            
            let rawType = photoOutput.availableRawPhotoPixelFormatTypes.first!
            
            // set ISO and Exposure Time
            do{
                try self.device.lockForConfiguration()
            } catch {
                fatalError("Device could not be locked.")
            }
            
            let deviceISO = device.iso
            let deviceExposureDuration = device.exposureDuration.seconds
            
            let iso = deviceISO
            let maxExposureDuration: CMTime = CMTime(seconds: 0.041, preferredTimescale: CMTimeScale(1000000))
            let exposureDuration: CMTime = min(device.exposureDuration, maxExposureDuration) // Don't drop under 21fps
            
            self.device.setExposureModeCustom(duration: exposureDuration, iso: iso)
            self.device.focusMode = .locked
            
            self.device.unlockForConfiguration()
            
            if frameCount == 0 { // sleep for 200 milliseconds to let exposure catch up for first frame
                usleep(200000)
            }
            
            photoSettings = AVCapturePhotoSettings(rawPixelFormatType: rawType, processedFormat: nil)
            photoSettings.isDepthDataDeliveryEnabled = false
        } else {
            fatalError("No RAW format found.")
        }
        
        photoOutput.capturePhoto(with: photoSettings, delegate: self)
    }
    
    // MARK: Photo Output
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        
        // Retrieve the image and depth data.
        guard let pixelBuffer = photo.pixelBuffer else {return}
        
        if self.frameCount >= self.bundleSize {
            self.frameCount = 99999
            return // don't record past bundle size
        }
        
        
        if self.recordScene {
            self.writeImageRAW(photo: photo, timestamp: photo.timestamp.seconds, frameCount: self.frameCount)
            self.rawFrameTimes.append(round(photo.timestamp.seconds * 1000) / 1000.0)
        }
        
        self.frameCount += 1
        self.capturePhoto()
        
    }
    
    // MARK: Write Depth
    func convertLensDistortionLookupTable(lookupTable: Data) -> [Float] {
        let tableLength = lookupTable.count / MemoryLayout<Float>.size
        var floatArray: [Float] = Array(repeating: 0, count: tableLength)
        _ = floatArray.withUnsafeMutableBytes{lookupTable.copyBytes(to: $0)}
        return floatArray
    }
    
    func convertIntrinsicMatrix(intrinsicMatrix: simd_float3x3) -> [[Float]]{
        return (0 ..< 3).map{ x in
            (0 ..< 3).map{ y in intrinsicMatrix[x][y]}
        }
    }
    
    func writeDepth(depthData: AVDepthData, timestamp: Double, frameCount: Int) {
        let pixelBuffer = depthData.depthDataMap
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == noErr else { return }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let srcPtr = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("Failed to retrieve depth pointer.")
            return
        }
        
        let rowBytes : Int = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = Int(CVPixelBufferGetWidth(pixelBuffer))
        let height = Int(CVPixelBufferGetHeight(pixelBuffer))
        let capacity = CVPixelBufferGetDataSize(pixelBuffer)
        let uint8Pointer = srcPtr.bindMemory(to: UInt8.self, capacity: capacity)
        
        let intrinsicWidth = depthData.cameraCalibrationData!.intrinsicMatrixReferenceDimensions.width
        let intrinsicHeight = depthData.cameraCalibrationData!.intrinsicMatrixReferenceDimensions.height
        let intrinsicMatrix = depthData.cameraCalibrationData!.intrinsicMatrix
        let lensDistortion = depthData.cameraCalibrationData!.lensDistortionLookupTable!
        let lensInverseDistortion = depthData.cameraCalibrationData!.inverseLensDistortionLookupTable!
        let depthAccuracy = depthData.depthDataAccuracy.rawValue
        
        var header = """
        <BEGINHEADER>
        description:depthmap,
        frameCount:\(String(describing: frameCount)),
        timestamp:\(String(describing: timestamp)),
        height:\(String(describing: height)),
        width:\(String(describing: width)),
        rowBytes:\(String(describing: rowBytes)),
        intrinsicWidth:\(String(describing: intrinsicWidth)),
        intrinsicHeight:\(String(describing: intrinsicHeight)),
        intrinsicMatrix:\(String(describing: convertIntrinsicMatrix(intrinsicMatrix: intrinsicMatrix))),
        lensDistortion:\(String(describing: convertLensDistortionLookupTable(lookupTable: lensDistortion))),
        lensInverseDistortion:\(String(describing: convertLensDistortionLookupTable(lookupTable: lensInverseDistortion))),
        depthAccuracy:\(String(describing: depthAccuracy))
        <ENDHEADER>
        """
        
        header = header.components(separatedBy: .whitespacesAndNewlines).joined() // remove newlines
        let encodedHeader = [UInt8](header.utf8)
        self.depthData.append(encodedHeader, count: header.utf8.count)
        self.depthData.append(uint8Pointer, count: Int(rowBytes * height))
    }
    
    // MARK: Write RAW
    func writeImageRAW(photo: AVCapturePhoto, timestamp: Double, frameCount: Int) {
        guard let pixelBuffer = photo.pixelBuffer else { return }
        
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == noErr else {
            print("Failed to retrieve readonly base address for RAW.")
            return
        }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        
        guard let srcPtr = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("Failed to retrieve RAW pointer.")
            return
        }
        
        let rowBytes : Int = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = Int(CVPixelBufferGetWidth(pixelBuffer))
        let height = Int(CVPixelBufferGetHeight(pixelBuffer))
        let capacity = CVPixelBufferGetDataSize(pixelBuffer)
        let uint8Pointer = srcPtr.bindMemory(to: UInt8.self, capacity: capacity)
        
        let exifdata = photo.metadata["{Exif}"] as! NSDictionary
        let DNGdata = photo.metadata["{DNG}"] as! NSDictionary
        let brightnessValue = exifdata["BrightnessValue"] != nil ? exifdata["BrightnessValue"]! : -1.0
        
        var header = """
        <BEGINHEADER>
        description:imageRAW,
        frameCount:\(String(describing: frameCount)),
        timestamp:\(String(describing: timestamp)),
        height:\(String(describing: height)),
        width:\(String(describing: width)),
        rowBytes:\(String(describing: rowBytes)),
        ISO:\(String(describing: (exifdata["ISOSpeedRatings"] as! NSArray)[0])),
        exposureTime:\(String(describing: exifdata["ExposureTime"]!)),
        apertureValue:\(String(describing: exifdata["ApertureValue"]!)),
        brightnessValue:\(String(describing: brightnessValue)),
        shutterSpeedValue:\(String(describing: exifdata["ShutterSpeedValue"]!)),
        pixelXDimension:\(String(describing: exifdata["PixelXDimension"]!)),
        pixelYDimension:\(String(describing: exifdata["PixelYDimension"]!)),
        blackLevel:\(String(describing: DNGdata["BlackLevel"]!)),
        whiteLevel:\(String(describing: DNGdata["WhiteLevel"]!))
        <ENDHEADER>
        """
        
        header = header.components(separatedBy: .whitespacesAndNewlines).joined() // remove newlines
        let encodedHeader = [UInt8](header.utf8)
        self.imageRAWData.append(encodedHeader, count: header.utf8.count)
        self.imageRAWData.append(uint8Pointer, count: Int(rowBytes * height))
    }
    
    
    // MARK: Write BGRA
    func writeImageBGRA(sampleBuffer: CMSampleBuffer, timestamp: Double, frameCount: Int) {
        
        var intrinsicMatrix: simd_float3x3?
        
        if let camData = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) as? Data {
            intrinsicMatrix = camData.withUnsafeBytes { $0.pointee }
        }
        
        guard let pixelBuffer = sampleBuffer.imageBuffer else { return }
        
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == noErr else { return }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        
        guard let srcPtr = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("Failed to retrieve BGRA pointer.")
            return
        }
        
        let rowBytes : Int = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = Int(CVPixelBufferGetWidth(pixelBuffer))
        let height = Int(CVPixelBufferGetHeight(pixelBuffer))
        let capacity = CVPixelBufferGetDataSize(pixelBuffer)
        let uint8Pointer = srcPtr.bindMemory(to: UInt8.self, capacity: capacity)
        
        
        var header = """
        <BEGINHEADER>
        description:imageBGRA,
        frameCount:\(String(describing: frameCount)),
        timestamp:\(String(describing: timestamp)),
        height:\(String(describing: height)),
        width:\(String(describing: width)),
        rowBytes:\(String(describing: rowBytes)),
        intrinsicMatrix:\(String(describing: convertIntrinsicMatrix(intrinsicMatrix: intrinsicMatrix!)))
        <ENDHEADER>
        """
        
        header = header.components(separatedBy: .whitespacesAndNewlines).joined() // remove newlines
        let encodedHeader = [UInt8](header.utf8)
        self.imageRGBData.append(encodedHeader, count: header.utf8.count)
        self.imageRGBData.append(uint8Pointer, count: Int(rowBytes * height))
        
    }
}

