/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 The app's main user interface.
 */

import Foundation
import SwiftUI
import MetalKit

// Add a title to a view that enlarges the view to full screen on tap.
struct Texture<T: View>: ViewModifier {
    let height: CGFloat
    let width: CGFloat
    let title: String
    let view: T
    func body(content: Content) -> some View {
        VStack {
            Text(title).foregroundColor(Color.red)
            // To display the same view in the navigation, reference the view
            // directly versus using the view's `content` property.
            NavigationLink(destination: view.aspectRatio(CGSize(width: width, height: height), contentMode: .fill)) {
                view.frame(maxWidth: width, maxHeight: height, alignment: .center)
                    .aspectRatio(CGSize(width: width, height: height), contentMode: .fill)
            }
        }
    }
}

extension View {
    // Apply `zoomOnTapModifier` with a `self` reference to show the same view
    // on tap.
    func zoomOnTapModifier(height: CGFloat, width: CGFloat, title: String) -> some View {
        modifier(Texture(height: height, width: width, title: title, view: self))
    }
}
extension Image {
    init(_ texture: MTLTexture, ciContext: CIContext, scale: CGFloat, orientation: Image.Orientation, label: Text) {
        let ciimage = CIImage(mtlTexture: texture)!
        let cgimage = ciContext.createCGImage(ciimage, from: ciimage.extent)
        self.init(cgimage!, scale: 1.0, orientation: .leftMirrored, label: label)
    }
}


struct MetalDepthView: View {
    @ObservedObject var manager = CameraManager()
    
    // Set the default sizes for the texture views.
    let sizeH: CGFloat = 320
    let sizeW: CGFloat = 240
    
    // Manage the AR session and AR data processing.
    //- Tag: ARProvider
    let ciContext: CIContext = CIContext()
    
    // Save the user's confidence selection.
    @State var isPaused = false
    @State var selectedConfidence = 0
    @State private var scaleMovement: Float = 1.5
    @State var saveSuffix: String = ""
    @State var numRecordedSceneBundles = 0
    @State var numRecordedPoseBundles = 0
    let screenWidth = UIScreen.main.bounds.size.width
    let fontSize : CGFloat = 22
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
        
            
            // depth and image display
            HStack(alignment: .top) {
                if manager.savingState == 1 {
                    Spacer(minLength: 10)
                    Image(systemName: "square.and.arrow.down.fill").font(.system(size: fontSize + 2)); Text("SAVING DATA TO DISK").font(.system(size: fontSize + 2))
                    Spacer(minLength: 10)
                } else if manager.savingState == 2 {
                    Spacer(minLength: 10)
                    Image(systemName: "exclamationmark.triangle.fill").font(.system(size: fontSize + 2)); Text("SOMETHING WENT WRONG,\nWAIT A MOMENT AND TRY AGAIN").font(.system(size: fontSize + 2))
                    Spacer(minLength: 10)
                } else {
                    MetalTextureViewDepth(mtkView: MTKView(), content: manager.capturedData.depthContent, confSelection: $selectedConfidence)
                    MetalTextureView(mtkView: MTKView(), content: manager.capturedData.colorRGBContent)
                }
            }.frame(width: screenWidth, height:400)
            
            HStack() {
                if manager.savingState == 0 {
                    Text("Exposure: \(manager.exposureTime) ISO: \(manager.iso)").font(.system(size: fontSize))
                }
            }.frame(width: 400, height: 30)
            
            // input field
            HStack() {
                Spacer(minLength: 10)
                TextField("Save File Suffix", text: $saveSuffix)
                    .disableAutocorrection(true)
                    .border(Color(UIColor.separator))
                    .autocapitalization(.none)
                    .font(.system(size: fontSize))
                Spacer(minLength: 10)
            }.frame(width: screenWidth, height: 50)
            
            // input field
            HStack() {
                Spacer(minLength: 10)
                Text("Recorded \(numRecordedPoseBundles) Motion, \(numRecordedSceneBundles) Scene Bundles").font(.system(size: fontSize))
                Spacer(minLength: 10)
            }.frame(width: screenWidth, height: 30)
            
            // bundle size selector
            HStack {
                Spacer(minLength: 10)
                Text("Bundle Size:").font(.system(size: fontSize))
                Picker(selection: $manager.controller.bundleSize, label: Text("Bundle Size:")) {
                    Text("1").tag(1)
                    Text("15").tag(15)
                    Text("30").tag(30)
                    Text("42").tag(42)
                }.pickerStyle(SegmentedPickerStyle())
                Spacer(minLength: 10)
            }.frame(width: screenWidth, height:50)
            
            // buttons for stream interaction
            HStack() {
                Spacer(minLength: 20)
                Button(action: {
                    manager.controller.frameCount = 99999
                    manager.controller.stopStream()
                    usleep(100000)
                    manager.controller.startStream()
                }) {
                    Image(systemName: "exclamationmark.arrow.circlepath").font(.system(size: 40))
                }
                Spacer(minLength: 80)
                Button(action: {
                    if manager.controller.frameCount == 99999 {
                        manager.controller.recordBundle(saveSuffix: saveSuffix)
                        numRecordedSceneBundles += 1
                    }
                }) {
                    Image(systemName: (manager.frameCount == 99999) ? "record.circle.fill" : "" ).font(.system(size: 40))
                }
                Spacer(minLength: 80)
                Button(action: {
                    if manager.controller.frameCount == 99999 {
                        manager.controller.recordMotionBundle(saveSuffix: saveSuffix)
                        numRecordedPoseBundles += 1
                    }
                }) {
                    Image(systemName: (manager.frameCount == 99999) ? "move.3d" : "" ).font(.system(size: 40))
                }
                Spacer(minLength: 20)
            }.frame(width: screenWidth, height: 90)
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(CGColor(red: 0, green: 0, blue: 0, alpha: 0.3)))
            .ignoresSafeArea()
    }
    
}
