//
//  RandomNumbers.swift
//  test-ELM
//
//  Created by Anton Akusok on 29/07/2018.
//  Copyright © 2018 Anton Akusok. All rights reserved.
//

import Foundation


private var nextNextGaussian: Double? = {
    srand48(Int(arc4random())) //initialize drand48 buffer at most once
    return nil
}()

func nextGaussian() -> Double {
    if let gaussian = nextNextGaussian {
        nextNextGaussian = nil
        return gaussian
    } else {
        var v1, v2, s: Double
        
        repeat {
            v1 = 2 * drand48() - 1
            v2 = 2 * drand48() - 1
            s = v1 * v1 + v2 * v2
        } while s >= 1 || s == 0
        
        let multiplier = sqrt(-2 * log(s)/s)
        nextNextGaussian = v2 * multiplier
        return v1 * multiplier
    }
}
