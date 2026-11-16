// kadr-matte: frames in, mattes out — the person (or the main subject) of each frame as a gray picture, white where it
// is. Uses what macOS has built in (the Vision framework): nothing to download, nothing leaves the machine.
//   kadr-matte W H person|subject|soft [x y]
//     person   the people of the frame, with the clean edge of the subject finder: the things macOS lifts out of a
//              picture are taken where they are people (a soft person matte alone left a dark jacket half transparent)
//     subject  every thing that stands out; with x y (frame px) — the one under that point, followed from frame to frame
//     soft     the person matte as it is (hair and motion blur keep their softness; each frame knows the ones before)
//   kadr-matte W H boxes person|subject
//     no pictures out: for every frame a line of numbers — where the people (or the subject) are: the middle of the
//     matte, its box, the share of the frame it takes, and the largest face's box (−1s when none): kadr reframe
//   kadr-matte W H pose
//     no pictures out: for every frame the joints of the person seen best and the points of the largest face (its
//     middle, eyes, mouth, chin, the top of its box) — a line «name x y confidence» each (frame px), then a line
//     «end»: kadr puppet --pose
//   stdin: frames W×H, 4 bytes a pixel (BGRA);  stdout: mattes W×H, 1 byte a pixel;  stderr: a line of progress
import Foundation
import Vision
import CoreVideo
import Accelerate

let args = CommandLine.arguments
guard args.count >= 4, let w = Int(args[1]), let h = Int(args[2]), w > 0, h > 0 else {
    fputs("usage: kadr-matte W H person|subject|soft [x y]\n", stderr)
    exit(2)
}
let boxes = args[3] == "boxes"
let mode = boxes ? (args.count > 4 ? args[4] : "person") : args[3]
let point: (Int, Int)? = (!boxes && args.count > 5) ? (Int(args[4]) ?? 0, Int(args[5]) ?? 0) : nil
let faces = VNDetectFaceRectanglesRequest()

var pbOpt: CVPixelBuffer?
CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary, &pbOpt)
guard let pb = pbOpt else { fputs("kadr-matte: no pixel buffer\n", stderr); exit(1) }

let sequence = VNSequenceRequestHandler()            // person mattes of a clip: each frame knows the ones before it
let person = VNGeneratePersonSegmentationRequest()
person.qualityLevel = .accurate
person.outputPixelFormat = kCVPixelFormatType_OneComponent8

var frame = [UInt8](repeating: 0, count: w * h * 4)
var out = [UInt8](repeating: 0, count: w * h)
var n = 0, empty = 0
var cover = 0.0                                   // the share of the frame the mattes take, added up
var last = [UInt8]()                              // subject x y: the thing chosen in the last frame

func scale8(_ m: CVPixelBuffer) {                    // a one-byte matte of any size → `out`, W×H
    CVPixelBufferLockBaseAddress(m, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(m, .readOnly) }
    var src = vImage_Buffer(data: CVPixelBufferGetBaseAddress(m), height: vImagePixelCount(CVPixelBufferGetHeight(m)),
                            width: vImagePixelCount(CVPixelBufferGetWidth(m)), rowBytes: CVPixelBufferGetBytesPerRow(m))
    out.withUnsafeMutableBytes { p in
        var dst = vImage_Buffer(data: p.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
        vImageScale_Planar8(&src, &dst, nil, vImage_Flags(kvImageHighQualityResampling))
    }
}

func scaleF(_ m: CVPixelBuffer) {                    // a float matte 0…1 of any size → `out`, W×H
    CVPixelBufferLockBaseAddress(m, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(m, .readOnly) }
    let mw = CVPixelBufferGetWidth(m), mh = CVPixelBufferGetHeight(m)
    var src = vImage_Buffer(data: CVPixelBufferGetBaseAddress(m), height: vImagePixelCount(mh), width: vImagePixelCount(mw), rowBytes: CVPixelBufferGetBytesPerRow(m))
    var tmp = [UInt8](repeating: 0, count: mw * mh)
    tmp.withUnsafeMutableBytes { t in
        var mid = vImage_Buffer(data: t.baseAddress, height: vImagePixelCount(mh), width: vImagePixelCount(mw), rowBytes: mw)
        vImageConvert_PlanarFtoPlanar8(&src, &mid, 1.0, 0.0, vImage_Flags(kvImageNoFlags))
        out.withUnsafeMutableBytes { p in
            var dst = vImage_Buffer(data: p.baseAddress, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w)
            vImageScale_Planar8(&mid, &dst, nil, vImage_Flags(kvImageHighQualityResampling))
        }
    }
}

while true {
    let got = frame.withUnsafeMutableBytes { fread($0.baseAddress, 1, w * h * 4, stdin) }
    if got < w * h * 4 { break }
    CVPixelBufferLockBaseAddress(pb, [])
    let base = CVPixelBufferGetBaseAddress(pb)!, row = CVPixelBufferGetBytesPerRow(pb)
    frame.withUnsafeBytes { f in
        for y in 0..<h { memcpy(base + y * row, f.baseAddress! + y * w * 4, w * 4) }
    }
    CVPixelBufferUnlockBaseAddress(pb, [])
    if args[3] == "pose" {
        let req = VNDetectHumanBodyPoseRequest()
        try? VNImageRequestHandler(cvPixelBuffer: pb, options: [:]).perform([req])
        let names: [(VNHumanBodyPoseObservation.JointName, String)] = [
            (.nose, "nose"), (.neck, "neck"), (.root, "root"), (.leftEye, "eye_a"), (.rightEye, "eye_b"), (.leftEar, "ear_a"), (.rightEar, "ear_b"),
            (.leftShoulder, "shoulder_a"), (.rightShoulder, "shoulder_b"), (.leftElbow, "elbow_a"), (.rightElbow, "elbow_b"),
            (.leftWrist, "wrist_a"), (.rightWrist, "wrist_b"), (.leftHip, "hip_a"), (.rightHip, "hip_b"),
            (.leftKnee, "knee_a"), (.rightKnee, "knee_b"), (.leftAnkle, "ankle_a"), (.rightAnkle, "ankle_b")]
        var best: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint] = [:]
        var bestScore: Float = -1
        for o in req.results ?? [] {                       // the person whose joints are seen best
            guard let pts = try? o.recognizedPoints(.all) else { continue }
            let score = pts.values.reduce(Float(0)) { $0 + $1.confidence }
            if score > bestScore { bestScore = score; best = pts }
        }
        for (j, name) in names {
            if let p = best[j], p.confidence > 0.05 {
                print(String(format: "%@ %.1f %.1f %.3f", name, Double(p.location.x) * Double(w), (1 - Double(p.location.y)) * Double(h), p.confidence))
            }
        }
        // the largest face's own points (a portrait shows too little of the body for its joints to be found)
        let lm = VNDetectFaceLandmarksRequest()
        try? VNImageRequestHandler(cvPixelBuffer: pb, options: [:]).perform([lm])
        if let f = lm.results?.max(by: { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height }), let marks = f.landmarks {
            let size = CGSize(width: w, height: h)
            func mid(_ r: VNFaceLandmarkRegion2D?, _ name: String) {
                guard let r = r, r.pointCount > 0 else { return }
                let pts = r.pointsInImage(imageSize: size)
                let x = pts.reduce(0.0) { $0 + Double($1.x) } / Double(pts.count), y = pts.reduce(0.0) { $0 + Double($1.y) } / Double(pts.count)
                print(String(format: "%@ %.1f %.1f %.3f", name, x, Double(h) - y, f.confidence))
            }
            mid(marks.nose, "face"); mid(marks.leftEye, "eye_a"); mid(marks.rightEye, "eye_b"); mid(marks.outerLips, "mouth")
            if let c = marks.faceContour, c.pointCount > 0, let low = c.pointsInImage(imageSize: size).min(by: { $0.y < $1.y }) {
                print(String(format: "chin %.1f %.1f %.3f", Double(low.x), Double(h) - Double(low.y), f.confidence))
            }
            let b = f.boundingBox
            print(String(format: "crown %.1f %.1f %.3f", (Double(b.minX) + Double(b.width) / 2) * Double(w), (1 - Double(b.maxY)) * Double(h), f.confidence))
        }
        print("end")
        n += 1
        continue
    }
    for i in 0..<(w * h) { out[i] = 0 }
    var found = false
    do {
        if mode == "soft" {
            try sequence.perform([person], on: pb)
            if let m = person.results?.first?.pixelBuffer {
                scale8(m)
                found = true
            }
        } else {
            let handler = VNImageRequestHandler(cvPixelBuffer: pb, options: [:])
            let req = VNGenerateForegroundInstanceMaskRequest()
            try handler.perform([req])
            var soft: CVPixelBuffer? = nil
            if mode == "person" {
                try sequence.perform([person], on: pb)
                soft = person.results?.first?.pixelBuffer
            }
            var chosen = IndexSet()
            if let obs = req.results?.first, !obs.allInstances.isEmpty {
                let lab = obs.instanceMask                 // one byte a pixel: 0 — nothing, 1… — which thing
                CVPixelBufferLockBaseAddress(lab, .readOnly)
                let lw = CVPixelBufferGetWidth(lab), lh = CVPixelBufferGetHeight(lab), lrow = CVPixelBufferGetBytesPerRow(lab)
                let lp = CVPixelBufferGetBaseAddress(lab)!.assumingMemoryBound(to: UInt8.self)
                // what decides: the person matte (person), the last frame's choice or the point (subject x y), else everything
                var guide: [UInt8]? = nil
                if mode == "person", let sm = soft {
                    var g = [UInt8](repeating: 0, count: lw * lh)
                    CVPixelBufferLockBaseAddress(sm, .readOnly)
                    var src = vImage_Buffer(data: CVPixelBufferGetBaseAddress(sm), height: vImagePixelCount(CVPixelBufferGetHeight(sm)),
                                            width: vImagePixelCount(CVPixelBufferGetWidth(sm)), rowBytes: CVPixelBufferGetBytesPerRow(sm))
                    g.withUnsafeMutableBytes { p in
                        var dst = vImage_Buffer(data: p.baseAddress, height: vImagePixelCount(lh), width: vImagePixelCount(lw), rowBytes: lw)
                        vImageScale_Planar8(&src, &dst, nil, vImage_Flags(kvImageNoFlags))
                    }
                    CVPixelBufferUnlockBaseAddress(sm, .readOnly)
                    guide = g
                } else if mode == "subject", point != nil {
                    if last.count == lw * lh {
                        guide = last
                    } else {
                        let px = min(lw - 1, max(0, point!.0 * lw / w)), py = min(lh - 1, max(0, point!.1 * lh / h))
                        let at = Int(lp[py * lrow + px])
                        if at > 0 { chosen.insert(at) }
                    }
                }
                if let g = guide {
                    // a thing is taken if a third of it is person, or if most of it is what was taken in the last
                    // frame: on two frames of 144 the person matte went pale over a dark jacket and the man was dropped
                    let was: [UInt8]? = last.count == lw * lh ? last : nil
                    var sum = [Int](repeating: 0, count: 256), cnt = [Int](repeating: 0, count: 256), old = [Int](repeating: 0, count: 256)
                    for y in 0..<lh { for x in 0..<lw {
                        let v = Int(lp[y * lrow + x])
                        if v > 0 { cnt[v] += 1; sum[v] += Int(g[y * lw + x]); if let p = was { old[v] += Int(p[y * lw + x]) } }
                    } }
                    for i in obs.allInstances where cnt[i] > 0 && (sum[i] > cnt[i] * 85 || old[i] > cnt[i] * 150) { chosen.insert(i) }
                } else if !(mode == "subject" && point != nil) {
                    chosen = obs.allInstances
                }
                if (mode == "person" || point != nil), !chosen.isEmpty {     // remembered for the next frame
                    last = [UInt8](repeating: 0, count: lw * lh)
                    for y in 0..<lh { for x in 0..<lw where chosen.contains(Int(lp[y * lrow + x])) { last[y * lw + x] = 255 } }
                }
                CVPixelBufferUnlockBaseAddress(lab, .readOnly)
                if !chosen.isEmpty {
                    scaleF(try obs.generateScaledMaskForImage(forInstances: chosen, from: handler))
                    found = true
                }
            }
            if !found, mode == "person", let sm = soft {     // people the subject finder did not lift out: the soft matte
                scale8(sm)
                found = out.contains { $0 > 127 }
            }
        }
    } catch {
        fputs("kadr-matte: frame \(n): \(error.localizedDescription)\n", stderr)
    }
    if !found { empty += 1 }
    cover += Double(out.reduce(0) { $0 + ($1 > 127 ? 1 : 0) }) / Double(w * h)
    if boxes {
        var sx = 0.0, sy = 0.0, cnt = 0.0
        var x0 = w, y0 = h, x1 = -1, y1 = -1
        for y in 0..<h { for x in 0..<w where out[y * w + x] > 127 {
            sx += Double(x); sy += Double(y); cnt += 1
            if x < x0 { x0 = x }; if x > x1 { x1 = x }; if y < y0 { y0 = y }; if y > y1 { y1 = y }
        } }
        var fx = -1.0, fy = -1.0, fw = -1.0, fh = -1.0
        if mode == "person" {
            try? VNImageRequestHandler(cvPixelBuffer: pb, options: [:]).perform([faces])
            if let f = faces.results?.max(by: { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height }) {
                let b = f.boundingBox                       // 0…1, counted from the bottom left
                fx = Double(b.minX) * Double(w); fw = Double(b.width) * Double(w)
                fy = (1 - Double(b.maxY)) * Double(h); fh = Double(b.height) * Double(h)
            }
        }
        if cnt > 0 {
            print(String(format: "%.1f %.1f %d %d %d %d %.5f %.1f %.1f %.1f %.1f", sx / cnt, sy / cnt, x0, y0, x1, y1, cnt / Double(w * h), fx, fy, fw, fh))
        } else {
            print(String(format: "-1 -1 -1 -1 -1 -1 0 %.1f %.1f %.1f %.1f", fx, fy, fw, fh))
        }
        n += 1
        continue
    }
    _ = out.withUnsafeBytes { fwrite($0.baseAddress, 1, w * h, stdout) }
    n += 1
    if n % 25 == 0 { fputs("frames \(n)\n", stderr) }
}
fflush(stdout)
fputs("done \(n) empty \(empty) cover \(n > 0 ? cover / Double(n) : 0)\n", stderr)
