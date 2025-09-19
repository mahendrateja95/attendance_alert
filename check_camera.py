#!/usr/bin/env python3
"""
Camera troubleshooting script for the Attendance Recognition System
Run this to check if the camera is accessible
"""
import cv2
import sys
import os

def check_camera_availability():
    """Check if camera devices are available"""
    print("🔍 Checking camera availability...")
    
    # Check for video devices in /dev
    video_devices = []
    for i in range(10):  # Check video0 to video9
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            video_devices.append(device_path)
    
    if video_devices:
        print(f"✅ Found video devices: {video_devices}")
    else:
        print("❌ No video devices found in /dev/")
        return False
    
    return True

def test_opencv_camera():
    """Test OpenCV camera access"""
    print("\n📷 Testing OpenCV camera access...")
    
    # Test local camera devices first
    for camera_id in range(3):  # Test camera IDs 0, 1, 2
        print(f"  Testing camera ID {camera_id}...")
        
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                print(f"    ❌ Camera {camera_id}: Cannot open")
                continue
            
            # Try to read a frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"    ✅ Camera {camera_id}: Working! Resolution: {width}x{height}")
                cap.release()
                return camera_id
            else:
                print(f"    ❌ Camera {camera_id}: Cannot read frame")
            
            cap.release()
            
        except Exception as e:
            print(f"    ❌ Camera {camera_id}: Error - {str(e)}")
    
    print("    ❌ No working local cameras found")
    return None

def test_network_camera_source():
    """Test network camera source from environment variable"""
    camera_source = os.getenv("CAMERA_SOURCE", "0")
    
    if camera_source.isdigit():
        print(f"\n🌐 CAMERA_SOURCE is set to local device: {camera_source}")
        return None
    
    print(f"\n🌐 Testing network camera source: {camera_source}")
    
    try:
        # For RTSP streams, use FFMPEG backend
        if camera_source.startswith('rtsp://'):
            cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
        else:
            # For HTTP/MJPEG streams, use default backend
            cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print(f"    ❌ Network camera: Cannot open stream")
            return None
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"    ✅ Network camera: Working! Resolution: {width}x{height}")
            cap.release()
            return camera_source
        else:
            print(f"    ❌ Network camera: Cannot read frame")
        
        cap.release()
        
    except Exception as e:
        print(f"    ❌ Network camera: Error - {str(e)}")
    
    return None

def check_permissions():
    """Check camera device permissions"""
    print("\n🔐 Checking camera permissions...")
    
    video_devices = [f"/dev/video{i}" for i in range(3)]
    
    for device in video_devices:
        if os.path.exists(device):
            try:
                # Check if we can read the device
                with open(device, 'rb') as f:
                    pass
                print(f"    ✅ {device}: Read permission OK")
            except PermissionError:
                print(f"    ❌ {device}: Permission denied")
            except Exception as e:
                print(f"    ⚠️  {device}: {str(e)}")

def main():
    print("🎯 Attendance Recognition System - Camera Diagnostic Tool")
    print("=" * 60)
    
    # Check if running in Docker
    if os.path.exists('/.dockerenv'):
        print("🐳 Running inside Docker container")
    else:
        print("💻 Running on host system")
    
    # Run checks
    camera_available = check_camera_availability()
    check_permissions()
    working_camera = test_opencv_camera()
    network_camera = test_network_camera_source()
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if working_camera is not None or network_camera is not None:
        if working_camera is not None:
            print(f"✅ SUCCESS: Local camera {working_camera} is working!")
        if network_camera is not None:
            print(f"✅ SUCCESS: Network camera stream is working!")
        print("🎉 Your attendance system should work properly.")
    else:
        print("❌ FAILURE: No working cameras found.")
        print("\n🔧 TROUBLESHOOTING STEPS:")
        
        if os.path.exists('/.dockerenv'):
            print("   Docker-specific fixes:")
            print("   1. For network cameras: Set CAMERA_SOURCE environment variable")
            print("   2. For local cameras: Ensure devices are mapped: --device /dev/video0:/dev/video0")
            print("   3. Run with privileged mode: --privileged (for local cameras)")
            print("   4. Check if camera is being used by another process")
            print("   5. Restart Docker container")
        else:
            print("   Host system fixes:")
            print("   1. For network cameras: Set CAMERA_SOURCE environment variable")
            print("   2. For local cameras: Check if camera is connected and recognized by OS")
            print("   3. Try: sudo usermod -a -G video $USER")
            print("   4. Check camera permissions: ls -la /dev/video*")
            print("   5. Install camera drivers if needed")
        
        print("\n📋 NETWORK CAMERA EXAMPLES:")
        print("   export CAMERA_SOURCE='rtsp://user:pass@192.168.1.100:554/stream1'")
        print("   export CAMERA_SOURCE='http://192.168.1.101:8080/video'  # IP Webcam")
    
    return 0 if (working_camera is not None or network_camera is not None) else 1

if __name__ == "__main__":
    sys.exit(main())
