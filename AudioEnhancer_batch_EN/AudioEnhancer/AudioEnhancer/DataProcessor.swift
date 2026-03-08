import Foundation
import CoreML
import Accelerate

class DataProcessor {
    
    // MARK: - Core IMU Data Processing Logic
    static func processIMUFiles(accURL: URL, gyroURL: URL) -> MLMultiArray? {
        guard let accRaw = parseTxtFile(url: accURL),
              let gyroRaw = parseTxtFile(url: gyroURL) else {
            return nil
        }
        
        let acc125 = fixLength(data: accRaw, targetLen: 125)
        let gyro125 = fixLength(data: gyroRaw, targetLen: 125)
        
        let acc150 = interpolateTime(data: acc125, targetLen: 150)
        let gyro150 = interpolateTime(data: gyro125, targetLen: 150)
        
        // 4. Data concatenation and flattening (for min/max calculation)
        var combinedData: [[Double]] = []
        var allValues: [Double] = []
        
        for i in 0..<150 {
            let rowAcc = acc150[i]
            let rowGyro = gyro150[i]
            let rowCombined = rowAcc + rowGyro
            combinedData.append(rowCombined)
            allValues.append(contentsOf: rowCombined)
        }
        
        // Time axis reversal
        let reversedData = Array(combinedData.reversed())
        
        guard let minVal = allValues.min(), let maxVal = allValues.max() else { return nil }
        let range = maxVal - minVal
        
        do {
            // 🔥 Performance optimization key: direct memory write
            let mlArray = try MLMultiArray(shape: [1, 1, 150, 6], dataType: .float32)
            
            // Get underlying pointer (MLMultiArray is usually Float32)
            let ptr = UnsafeMutablePointer<Float>(OpaquePointer(mlArray.dataPointer))
            let stride = 6 // Size of the last dimension
            
            for i in 0..<150 {
                let row = reversedData[i]
                for c in 0..<6 {
                    let rawVal = row[c]
                    var normalizedVal: Double = 0.0
                    if range > 1e-6 {
                        let norm01 = (rawVal - minVal) / range
                        normalizedVal = (norm01 * 2.0) - 1.0
                    }
                    
                    // Direct memory offset calculation, avoiding NSNumber and subscript checks
                    // Shape is [1, 1, 150, 6], flattened index = i * 6 + c
                    ptr[i * stride + c] = Float(normalizedVal)
                }
            }
            return mlArray
            
        } catch {
            print("❌ Failed to create MLArray: \(error)")
            return nil
        }
    }
    
    // --- Helper functions remain unchanged ---
    // Replace parseTxtFile function in DataProcessor class

        private static func parseTxtFile(url: URL) -> [[Double]]? {
            do {
                // Fix: Do not force guard, as child files might not need independent authorization (parent folder is authorized)
                let isSecured = url.startAccessingSecurityScopedResource()
                defer { if isSecured { url.stopAccessingSecurityScopedResource() } }
                
                let content = try String(contentsOf: url, encoding: .utf8)
                
                var result: [[Double]] = []
                let lines = content.components(separatedBy: .newlines)
                for line in lines {
                    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                    if trimmed.isEmpty { continue }
                    let parts = trimmed.split(separator: " ").compactMap { Double($0) }
                    if parts.count >= 3 { result.append(Array(parts.prefix(3))) }
                }
                return result
            } catch {
                print("Failed to read file: \(url.lastPathComponent), Error: \(error)")
                return nil
            }
        }
   
    private static func fixLength(data: [[Double]], targetLen: Int) -> [[Double]] {
        if data.count >= targetLen { return Array(data.prefix(targetLen)) }
        var padded = data
        let diff = targetLen - data.count
        for _ in 0..<diff { padded.append([0.0, 0.0, 0.0]) }
        return padded
    }
    
    private static func interpolateTime(data: [[Double]], targetLen: Int) -> [[Double]] {
        let sourceLen = data.count
        if sourceLen == 0 { return [] }
        var result: [[Double]] = []
        result.reserveCapacity(targetLen) // Pre-allocate
        for i in 0..<targetLen {
            let position = Double(i) * Double(sourceLen) / Double(targetLen)
            let index = Int(position)
            let remainder = position - Double(index)
            if index >= sourceLen - 1 {
                result.append(data.last!)
            } else {
                let p1 = data[index]
                let p2 = data[index + 1]
                let newX = p1[0] + (p2[0] - p1[0]) * remainder
                let newY = p1[1] + (p2[1] - p1[1]) * remainder
                let newZ = p1[2] + (p2[2] - p1[2]) * remainder
                result.append([newX, newY, newZ])
            }
        }
        return result
    }
}
