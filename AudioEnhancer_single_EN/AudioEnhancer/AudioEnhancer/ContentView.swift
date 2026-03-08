import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject var engine = InferenceEngine()
    
    @State private var wavURL: URL?
    @State private var accURL: URL?
    @State private var gyroURL: URL?
    
    @State private var showWavPicker = false
    @State private var showAccPicker = false
    @State private var showGyroPicker = false
    
    var body: some View {
        VStack(spacing: 30) {
            Text("Audio Enhancer App")
                .font(.title)
                .bold()
            
            // --- File Selection Area ---
            Group {
                FileSelectButton(title: "Select Audio (.wav)", selectedURL: wavURL) {
                    showWavPicker = true
                }
                .fileImporter(isPresented: $showWavPicker, allowedContentTypes: [.audio]) { result in
                    if let url = try? result.get() { self.wavURL = url }
                }
                
                FileSelectButton(title: "Select Accelerometer (.txt)", selectedURL: accURL) {
                    showAccPicker = true
                }
                .fileImporter(isPresented: $showAccPicker, allowedContentTypes: [.text]) { result in
                    if let url = try? result.get() { self.accURL = url }
                }
                
                FileSelectButton(title: "Select Gyroscope (.txt)", selectedURL: gyroURL) {
                    showGyroPicker = true
                }
                .fileImporter(isPresented: $showGyroPicker, allowedContentTypes: [.text]) { result in
                    if let url = try? result.get() { self.gyroURL = url }
                }
            }
            
            Divider()
            
            // --- Run and Status ---
            VStack(spacing: 8) {
                Text(engine.status)
                    .foregroundColor(.gray)
                
                // ✨ New: Show processing time
                if !engine.processingTime.isEmpty {
                    Text(engine.processingTime)
                        .font(.headline)
                        .foregroundColor(.green)
                        .transition(.opacity)
                }
            }
            .padding()
            
            if engine.isProcessing {
                ProgressView()
            } else {
                Button(action: {
                    if let w = wavURL, let a = accURL, let g = gyroURL {
                        engine.runPipeline(wavURL: w, accURL: a, gyroURL: g)
                    } else {
                        engine.status = "Please select all files first"
                    }
                }) {
                    Text("Start Processing")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(width: 200, height: 50)
                        .background((wavURL != nil && accURL != nil && gyroURL != nil) ? Color.blue : Color.gray)
                        .cornerRadius(10)
                }
                .disabled(wavURL == nil || accURL == nil || gyroURL == nil)
            }
            
            // Share button (only show when file exists)
            if let docsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                let resultURL = docsURL.appendingPathComponent("enhanced_result.wav")
                if FileManager.default.fileExists(atPath: resultURL.path) && !engine.isProcessing {
                    ShareLink(item: resultURL) {
                        Label("Export Result File", systemImage: "square.and.arrow.up")
                            .font(.headline)
                            .padding()
                            .background(Color.orange)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                }
            }
        }
        .padding()
        .animation(.easeInOut, value: engine.processingTime) // Add a simple fade-in animation
    }
}

// Helper View: File selection button component
struct FileSelectButton: View {
    var title: String
    var selectedURL: URL?
    var action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: selectedURL == nil ? "doc.badge.plus" : "checkmark.circle.fill")
                    .foregroundColor(selectedURL == nil ? .blue : .green)
                VStack(alignment: .leading) {
                    Text(title)
                        .foregroundColor(.primary)
                    if let url = selectedURL {
                        Text(url.lastPathComponent)
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                }
                Spacer()
            }
            .padding()
            .background(Color(UIColor.secondarySystemBackground))
            .cornerRadius(8)
        }
    }
}
