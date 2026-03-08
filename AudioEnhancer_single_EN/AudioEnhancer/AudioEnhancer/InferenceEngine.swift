import Foundation
import CoreML
import AVFoundation
import Combine
import Accelerate

class InferenceEngine: ObservableObject {
    @Published var status = "Please select files"
    @Published var isProcessing = false
    @Published var processingTime = ""
    
    func runPipeline(wavURL: URL, accURL: URL, gyroURL: URL) {
        self.isProcessing = true
        self.status = "Initializing..."
        self.processingTime = ""
        
        let startTime = Date()
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                DispatchQueue.main.async { self.status = "Loading models..." }
                let config = MLModelConfiguration()
                config.computeUnits = .all
                
                let genModel = try Generator(configuration: config)
                let denoiseModel = try DenoiseNet(configuration: config)
                
                // ==========================================
                // 2. Process IMU Data
                // ==========================================
                DispatchQueue.main.async { self.status = "Processing IMU Data..." }
                guard let imuInput = DataProcessor.processIMUFiles(accURL: accURL, gyroURL: gyroURL) else {
                    throw NSError(domain: "App", code: -1, userInfo: [NSLocalizedDescriptionKey: "IMU data processing failed"])
                }
                
                // ==========================================
                // 3. Run Generator
                // ==========================================
                DispatchQueue.main.async { self.status = "Running Generator..." }
                let genOutput = try genModel.prediction(imu_input: imuInput)
                
                guard let videoDisp = self.processGeneratorOutputFast(genOutput.video_disp_output) else {
                    throw NSError(domain: "App", code: -3, userInfo: [NSLocalizedDescriptionKey: "Generator output processing failed"])
                }
                
                // ==========================================
                // 4. Audio STFT
                // ==========================================
                DispatchQueue.main.async { self.status = "Audio STFT transform..." }
                let (audioData, sampleRate) = self.readWav(url: wavURL)
                if audioData.isEmpty {
                    throw NSError(domain: "App", code: -2, userInfo: [NSLocalizedDescriptionKey: "Audio reading failed or is empty"])
                }
                
                let stftTool = AudioSpectrogram(n_fft: 2048, hop_length: 512, sampleRate: Float(sampleRate))
                let (rawMagnitudes, rawPhases) = stftTool.stft(audioData: audioData)
                let normSpec = stftTool.processToModelInput(magnitudes: rawMagnitudes)
                
                let targetTimeSteps = 431
                let freqBins = 1025
                let fixedSpec = self.fixSpectrogramDimensions(normSpec, targetTime: targetTimeSteps, targetFreq: freqBins)
                
                // ==========================================
                // 5. Build CoreML Input (Fix stride issue)
                // ==========================================
                // Shape: [1, 1, 431, 1025]
                let audioInputML = try MLMultiArray(shape: [1, 1, NSNumber(value: targetTimeSteps), NSNumber(value: freqBins)], dataType: .float32)
                
                // Get actual strides
                let strides = audioInputML.strides.map { $0.intValue }
                let tStride = strides[2]
                let fStride = strides[3]
                
                // ✅ Mutable pointer: keep 2 parameters (ptr, _)
                try audioInputML.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
                    for t in 0..<targetTimeSteps {
                        // Use actual stride to calculate row start address
                        let rowStart = t * tStride
                        let specRow = fixedSpec[t]
                        
                        for f in 0..<freqBins {
                            // Keep frequency flipping logic
                            // Pointer address = row start + col * col stride
                            ptr[rowStart + f * fStride] = specRow[freqBins - 1 - f]
                        }
                    }
                }
                
                // ==========================================
                // 6. Run Denoise Model
                // ==========================================
                DispatchQueue.main.async { self.status = "Running Denoise Model..." }
                let denoiseInput = DenoiseNetInput(audio_spectrogram: audioInputML, video_displacement: videoDisp)
                let denoiseOutput = try denoiseModel.prediction(input: denoiseInput)
                
                // ==========================================
                // 7. Post-processing (Fix stride issue)
                // ==========================================
                DispatchQueue.main.async { self.status = "Reconstructing audio..." }
                let cleanedML = denoiseOutput.clean_spectrogram
                
                // CoreML might return Float16, type checking is necessary here
                let cleanedSpec2D = try self.safeReadOutput(cleanedML, timeSteps: targetTimeSteps, freqBins: freqBins)
                
                let linearMag = stftTool.denormalize(modelOutput: cleanedSpec2D, minDb: -90.0, maxDb: 0.0)
                let alignedPhases = self.alignPhases(rawPhases, targetTime: targetTimeSteps, freqBins: freqBins)
                var finalAudio = stftTool.istft(magnitudes: linearMag, phases: alignedPhases)
                
                // Automatic Gain Control
                var maxAmp: Float = 0
                vDSP_maxmgv(finalAudio, 1, &maxAmp, vDSP_Length(finalAudio.count))
                if maxAmp > 0 {
                    let targetPeak: Float = 0.95
                    var scale = targetPeak / maxAmp
                    vDSP_vsmul(finalAudio, 1, &scale, &finalAudio, 1, vDSP_Length(finalAudio.count))
                }
                
                // ==========================================
                // 8. Save Results
                // ==========================================
                DispatchQueue.main.async { self.status = "Saving file..." }
                let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                let outputURL = docsURL.appendingPathComponent("enhanced_result.wav")
                self.saveWav(audioData: finalAudio, sampleRate: sampleRate, url: outputURL)
                
                DispatchQueue.main.async {
                    let duration = Date().timeIntervalSince(startTime)
                    self.processingTime = String(format: "⏱️ Total time: %.3f seconds", duration)
                    self.status = "✅ Processing complete"
                    self.isProcessing = false
                    print("Result generated: \(outputURL.path)")
                }
                
            } catch {
                print("Pipeline Error: \(error)")
                DispatchQueue.main.async {
                    self.status = "Error: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }
    
    // MARK: - 🔥 Core fix: Safe and fast reading (supports stride and type)
    private func safeReadOutput(_ mlArray: MLMultiArray, timeSteps: Int, freqBins: Int) throws -> [[Float]] {
        var result = [[Float]](repeating: [Float](repeating: 0, count: freqBins), count: timeSteps)
        
        let strides = mlArray.strides.map { $0.intValue }
        let tStride = strides[2]
        let fStride = strides[3]
        
        // 1. If Float32, use pointer + stride calculation (ultra-fast)
        if mlArray.dataType == .float32 {
            // 🛠️ Fix: Immutable pointer, only accepts 1 parameter `ptr`
            try mlArray.withUnsafeBufferPointer(ofType: Float.self) { ptr in
                for t in 0..<timeSteps {
                    let rowStart = t * tStride
                    for f in 0..<freqBins {
                        // Use real physical address
                        let val = ptr[rowStart + f * fStride]
                        result[t][freqBins - 1 - f] = val
                    }
                }
            }
        }
        // 2. If Float16/Double, or non-standard memory returned by GPU, fallback to safe subscript (Safe)
        else {
            for t in 0..<timeSteps {
                for f in 0..<freqBins {
                    let idx = [0, 0, NSNumber(value: t), NSNumber(value: f)] as [NSNumber]
                    result[t][freqBins - 1 - f] = mlArray[idx].floatValue
                }
            }
        }
        return result
    }
    
    // MARK: - Optimized Generator Output Processing (Supports Stride)
    private func processGeneratorOutputFast(_ rawOutput: MLMultiArray) -> MLMultiArray? {
        let srcH = 152; let srcW = 48
        let dstH = 150; let dstW = 40
        
        do {
            let result = try MLMultiArray(shape: [1, 1, 150, 40], dataType: .float32)
            
            // Get input and output strides
            let rawStrides = rawOutput.strides.map { $0.intValue }
            let resStrides = result.strides.map { $0.intValue }
            
            let rawYStride = rawStrides[2]
            let rawXStride = rawStrides[3]
            let resYStride = resStrides[2]
            let resXStride = resStrides[3]
            
            // Only use pointer optimization when input is Float32
            if rawOutput.dataType == .float32 {
                // 🛠️ Fix: Immutable pointer, only accepts 1 parameter `rawPtr`
                try rawOutput.withUnsafeBufferPointer(ofType: Float.self) { rawPtr in
                    
                    // ✅ Mutable pointer: keep 2 parameters `resPtr, _`
                    try result.withUnsafeMutableBufferPointer(ofType: Float.self) { resPtr, _ in
                        
                        for y in 0..<dstH {
                            let invertedY = dstH - 1 - y
                            let srcY = Double(invertedY) * Double(srcH) / Double(dstH)
                            
                            for x in 0..<dstW {
                                let srcX = Double(x) * Double(srcW) / Double(dstW)
                                
                                let y1 = Int(floor(srcY)); let y2 = min(y1 + 1, srcH - 1)
                                let x1 = Int(floor(srcX)); let x2 = min(x1 + 1, srcW - 1)
                                
                                let dy = Float(srcY - Double(y1))
                                let dx = Float(srcX - Double(x1))
                                
                                // 🔥 Use stride to calculate physical address
                                let v11 = rawPtr[y1 * rawYStride + x1 * rawXStride]
                                let v12 = rawPtr[y1 * rawYStride + x2 * rawXStride]
                                let v21 = rawPtr[y2 * rawYStride + x1 * rawXStride]
                                let v22 = rawPtr[y2 * rawYStride + x2 * rawXStride]
                                
                                let val = (v11 * (1 - dx) + v12 * dx) * (1 - dy) + (v21 * (1 - dx) + v22 * dx) * dy
                                
                                resPtr[y * resYStride + x * resXStride] = (val + 1.0) / 2.0
                            }
                        }
                    }
                }
            } else {
                // Slow path (Compatibility)
                for y in 0..<dstH {
                    let invertedY = dstH - 1 - y
                    let srcY = Double(invertedY) * Double(srcH) / Double(dstH)
                    for x in 0..<dstW {
                        let srcX = Double(x) * Double(srcW) / Double(dstW)
                        let y1 = Int(floor(srcY)); let y2 = min(y1 + 1, srcH - 1)
                        let x1 = Int(floor(srcX)); let x2 = min(x1 + 1, srcW - 1)
                        let dy = srcY - Double(y1); let dx = srcX - Double(x1)
                        
                        let v11 = rawOutput[[0, 0, NSNumber(value: y1), NSNumber(value: x1)]].doubleValue
                        let v12 = rawOutput[[0, 0, NSNumber(value: y1), NSNumber(value: x2)]].doubleValue
                        let v21 = rawOutput[[0, 0, NSNumber(value: y2), NSNumber(value: x1)]].doubleValue
                        let v22 = rawOutput[[0, 0, NSNumber(value: y2), NSNumber(value: x2)]].doubleValue
                        
                        let val = (v11 * (1 - dx) + v12 * dx) * (1 - dy) + (v21 * (1 - dx) + v22 * dx) * dy
                        result[[0, 0, NSNumber(value: y), NSNumber(value: x)]] = NSNumber(value: (val + 1.0) / 2.0)
                    }
                }
            }
            return result
        } catch { return nil }
    }
    
    // --- Helper Functions ---
    
    private func fixSpectrogramDimensions(_ spec: [[Float]], targetTime: Int, targetFreq: Int) -> [[Float]] {
        var result = spec
        if result.count > targetTime { result = Array(result.prefix(targetTime)) }
        else if result.count < targetTime {
            let padding = targetTime - result.count
            let empty = [Float](repeating: 0.0, count: targetFreq)
            for _ in 0..<padding { result.append(empty) }
        }
        return result
    }
    
    private func alignPhases(_ phases: [[Float]], targetTime: Int, freqBins: Int) -> [[Float]] {
        var result = phases
        if result.count > targetTime { result = Array(result.prefix(targetTime)) }
        else if result.count < targetTime {
            let padding = targetTime - result.count
            let empty = [Float](repeating: 0.0, count: freqBins)
            for _ in 0..<padding { result.append(empty) }
        }
        return result
    }
    
    func readWav(url: URL) -> ([Float], Int) {
        let secured = url.startAccessingSecurityScopedResource()
        defer { if secured { url.stopAccessingSecurityScopedResource() } }
        guard let file = try? AVAudioFile(forReading: url),
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: file.fileFormat.sampleRate, channels: 1, interleaved: false),
              let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else { return ([], 44100) }
        try? file.read(into: buf)
        let data = Array(UnsafeBufferPointer(start: buf.floatChannelData?[0], count: Int(buf.frameLength)))
        return (data, Int(file.fileFormat.sampleRate))
    }
    
    func saveWav(audioData: [Float], sampleRate: Int, url: URL) {
        if FileManager.default.fileExists(atPath: url.path) { try? FileManager.default.removeItem(at: url) }
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
        guard let file = try? AVAudioFile(forWriting: url, settings: format.settings),
              let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audioData.count)) else { return }
        buf.frameLength = AVAudioFrameCount(audioData.count)
        if let ptr = buf.floatChannelData?[0] {
            for (i, val) in audioData.enumerated() { ptr[i] = val }
        }
        try? file.write(from: buf)
    }
}
