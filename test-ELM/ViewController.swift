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
        
        let bK = 19
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
        
        let bK = 19
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
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        textField.text = nil
        // run()
        batchProcessing()
    }

    func log(_ message: String) {
        let currentText = textField.text ?? ""
        textField.text = currentText + message + "\n"
    }

}



//        self.log("All files:")
//        for url in mainBundle.urls(forResourcesWithExtension: "npy", subdirectory: nil)! {
//            self.log(url.relativePath)
//        }
