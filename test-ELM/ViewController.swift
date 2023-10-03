//
//  ViewController.swift
//  test-ELM
//
//  Created by Anton Akusok on 27/07/2018.
//  Copyright © 2018 Anton Akusok. All rights reserved.
//

import UIKit
import MetalPerformanceShaders

protocol Logger {
    func log(_ message: String)
}

class ViewController: UIViewController, Logger {

    @IBOutlet weak var textField: UITextView!
    
    @IBAction func buttonClicked(_ sender: Any) {
        textField.text = ""
        run()
    }
    
    func run() {
        let mainBundle = Bundle.main
        let device = MTLCreateSystemDefaultDevice()!

        self.log("All files:")
        for url in mainBundle.urls(forResourcesWithExtension: "npy", subdirectory: nil)! {
            self.log(url.relativePath)
        }

        // use uploaded files
        let docDir = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let allFiles = try! FileManager.default.contentsOfDirectory(at: docDir, includingPropertiesForKeys: nil)
        let myFiles = Dictionary(uniqueKeysWithValues: allFiles.map { ($0.lastPathComponent, $0) })
        
        let bK = 35
        let bL = 1024
        let c = 3


        self.log("Data file:")
        let fileX = mainBundle.url(forResource: "x1k", withExtension: "npy")!
        let fileXs = mainBundle.url(forResource: "xs", withExtension: "npy")!
        let fileY = mainBundle.url(forResource: "y1k", withExtension: "npy")!
        
        let filesW = Array(0 ..< bK).map { myFiles["w_\($0).npy"]! }
        let filesBias = Array(0 ..< bK).map { myFiles["bias_\($0).npy"]! }
        _ = filesW.map { self.log($0.absoluteString) }
        self.log("")

        let X: MPSMatrix = loadFromNpy(contentsOf: fileX, device: device)
        let Xs: MPSMatrix = loadFromNpy(contentsOf: fileXs, device: device)
        let Y: MPSMatrix = loadFromNpy(contentsOf: fileY, device: device)

        let t0 = CFAbsoluteTimeGetCurrent()
            let model = ELM(device: device, bK: bK, bL: bL, alpha: 1E3, W: filesW, bias: filesBias)!
            model.fit(X: X, Y: Y)
            let Yh = model.predict(X: Xs)!
        let t = CFAbsoluteTimeGetCurrent() - t0
        self.log(String(format: "Runtime: %.3f", t))
        self.log("")


        self.log("Results")
        let res = Yh.data.contents().bindMemory(to: Float.self, capacity: Yh.rows * c)
        for i in 0 ..< Yh.rows {
            self.log(Array(0 ..< c).map { res[c*i + $0] }.map { String(format: "%.3f", $0) }.joined(separator: "\t"))
        }
    }
    
    func batchProcessing() {
        let mainBundle = Bundle.main
        let device = MTLCreateSystemDefaultDevice()!
        
        self.log("All files:")
        for url in mainBundle.urls(forResourcesWithExtension: "npy", subdirectory: nil)! {
            self.log(url.relativePath)
        }
        
        // use uploaded files
        let docDir = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let allFiles = try! FileManager.default.contentsOfDirectory(at: docDir, includingPropertiesForKeys: nil)
        let myFiles = Dictionary(uniqueKeysWithValues: allFiles.map { ($0.lastPathComponent, $0) })
        
        let bK = 3
        let bL = 1024
        let c = 3
        
        
        self.log("Data file:")
        let fileX = mainBundle.url(forResource: "x1k", withExtension: "npy")!
        let fileXs = mainBundle.url(forResource: "xs", withExtension: "npy")!
        let fileY = mainBundle.url(forResource: "y1k", withExtension: "npy")!
        
        let filesW = Array(0 ..< bK).map { myFiles["w_\($0).npy"]! }
        let filesBias = Array(0 ..< bK).map { myFiles["bias_\($0).npy"]! }
        _ = filesW.map { self.log($0.absoluteString) }
        self.log("")
        
        let X: MPSMatrix = loadFromNpy(contentsOf: fileX, device: device)
        let Xs: MPSMatrix = loadFromNpy(contentsOf: fileXs, device: device)
        let Y: MPSMatrix = loadFromNpy(contentsOf: fileY, device: device)
        
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = ELM(device: device, bK: bK, bL: bL, alpha: 1E3, W: filesW, bias: filesBias)!
        model.fit(X: X, Y: Y)
        let Yh = model.predict(X: Xs)!
        let t = CFAbsoluteTimeGetCurrent() - t0
        self.log(String(format: "Runtime: %.3f", t))
        self.log("")
        
        
        self.log("Results")
        let res = Yh.data.contents().bindMemory(to: Float.self, capacity: Yh.rows * c)
        for i in 0 ..< Yh.rows {
            self.log(Array(0 ..< c).map { res[c*i + $0] }.map { String(format: "%.3f", $0) }.joined(separator: "\t"))
        }

    }
    
    func customELM(data: Data) {
        let mainBundle = Bundle.main
        let device = MTLCreateSystemDefaultDevice()!
        
        // use uploaded files
        let docDir = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let allFiles = try! FileManager.default.contentsOfDirectory(at: docDir, includingPropertiesForKeys: nil)
        let myFiles = Dictionary(uniqueKeysWithValues: allFiles.map { ($0.lastPathComponent, $0) })
        
        let bK = 1  // weight batches
        let bL = 150
        let c = 1
        
        self.log("Data file:")
        let fileX = mainBundle.url(forResource: "hX", withExtension: "npy")!
        let fileXs = mainBundle.url(forResource: "hX", withExtension: "npy")!
        let fileY = mainBundle.url(forResource: "hY", withExtension: "npy")!
        
        let fileW = myFiles["hW_150.npy"]!
        let fileBias = myFiles["hbias_150.npy"]!
        self.log("")
        
        let X: MPSMatrix = loadFromNpy(contentsOf: fileX, device: device)
        let Y: MPSMatrix = loadFromNpy(contentsOf: fileY, device: device)
        
        // load Xs from data
        let npy = try! Npy(data: data)
        let rows = npy.shape[0]
        let columns = npy.shape[1]
        let ptr = npy.elementsData.withUnsafeBytes { UnsafeRawPointer($0) }
        let buffer = device.makeBuffer(bytes: ptr, length: rows * columns * fp32stride, options: [])!
        let descr = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * fp32stride, dataType: .float32)
        let Xs = MPSMatrix(buffer: buffer, descriptor: descr)
        
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = ELM(device: device, bK: bK, bL: bL, alpha: 1E1, W: [fileW], bias: [fileBias])!
        model.fit(X: X, Y: Y)
        let t = CFAbsoluteTimeGetCurrent() - t0
        self.log(String(format: "Training time: %.3f", t))

        let t1 = CFAbsoluteTimeGetCurrent()
        _ = model.predict(X: Xs)
        let t2 = CFAbsoluteTimeGetCurrent() - t1
        self.log(String(format: "Predict time: %.3f", t2))
        self.log("")
    }
    
    func buildELM(device: MTLDevice) -> ELM {
        let mainBundle = Bundle.main
        
        // use uploaded files
        let docDir = try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let allFiles = try! FileManager.default.contentsOfDirectory(at: docDir, includingPropertiesForKeys: nil)
        let myFiles = Dictionary(uniqueKeysWithValues: allFiles.map { ($0.lastPathComponent, $0) })
        
        let bK = 1  // weight batches
        let bL = 150
        let c = 1
        
        self.log("Data file:")
        let fileX = mainBundle.url(forResource: "hX", withExtension: "npy")!
        let fileY = mainBundle.url(forResource: "hY", withExtension: "npy")!
        
        let fileW = myFiles["hW_150.npy"]!
        let fileBias = myFiles["hbias_150.npy"]!
        self.log("")
        
        let X: MPSMatrix = loadFromNpy(contentsOf: fileX, device: device)
        let Y: MPSMatrix = loadFromNpy(contentsOf: fileY, device: device)
        
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = ELM(device: device, bK: bK, bL: bL, alpha: 1E1, W: [fileW], bias: [fileBias])!
        model.fit(X: X, Y: Y)
        let t = CFAbsoluteTimeGetCurrent() - t0
        self.log(String(format: "Training time: %.3f", t))
        
        return model
    }
    
    func predictWithModel(device: MTLDevice, model: ELM, data: Data, y: Int) {
        // load Xs from data
        let t0 = CFAbsoluteTimeGetCurrent()
        let npy = try! Npy(data: data)
        let rows = npy.shape[0]
        let columns = npy.shape[1]
        
        let buffer = device.makeBuffer(bytes: Array(npy.elementsData), length: rows * columns * fp32stride, options: [])!
        let descr = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * fp32stride, dataType: .float32)
        let Xs = MPSMatrix(buffer: buffer, descriptor: descr)
        
        let t1 = CFAbsoluteTimeGetCurrent()
        _ = model.predict(X: Xs)
        let t2 = CFAbsoluteTimeGetCurrent() - t1
        let t3 = CFAbsoluteTimeGetCurrent() - t0
        
        self.log(String(format: "Predict time of y=\(y): %.0f ms; total time: %.0f ms", t2*1000, t3*1000))
    }

    
    func bar(_ z: Int, _ x: Int, _ y: Int, device: MTLDevice, model: ELM) async {
//        self.log("sync starts: <\(z) \(x) \(y)>")
        
        let url = URL(string: "http://akusok.asuscomm.com:9000/elevation/combined_data/\(z)/\(x)/\(y).npy")!
        
        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                print("error")
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
                print("bad return code")
                return
            }

            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200, let data = data {
                DispatchQueue.main.async { self.predictWithModel(device: device, model: model, data: data, y: y) }
            }
        }
        task.resume()
//        self.log("sync ends: <\(z) \(x) \(y)>")
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        textField.text = nil
        // run()
        // batchProcessing()
        // customELM()
//        Task { await foo() }
        
        let device = MTLCreateSystemDefaultDevice()!
        let model = buildELM(device: device)
        
        Task {
            await bar(6, 36, 15, device: device, model: model)
            await bar(6, 36, 16, device: device, model: model)
            await bar(6, 36, 17, device: device, model: model)
            await bar(6, 36, 18, device: device, model: model)
        }
        
        let t1 = CFAbsoluteTimeGetCurrent()
        let loadTasks = Array(1103...1200).map { y in
            Task {
                await bar(12, 2287, y, device: device, model: model)
            }
        }
        let t2 = CFAbsoluteTimeGetCurrent() - t1
        self.log(String(format: "Loading tasks took: %.1f seconds", t2))
        print(String(format: "Loading tasks took: %.1f seconds", t2))
    }

    func log(_ message: String) {
        let currentText = textField.text ?? ""
        textField.text = currentText + message + "\n"
    }

}
