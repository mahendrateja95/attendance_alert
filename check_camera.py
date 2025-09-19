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
    
    print("    ❌ No working cameras found")
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
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if working_camera is not None:
        print(f"✅ SUCCESS: Camera {working_camera} is working!")
        print("🎉 Your attendance system should work properly.")
    else:
        print("❌ FAILURE: No working cameras found.")
        print("\n🔧 TROUBLESHOOTING STEPS:")
        
        if os.path.exists('/.dockerenv'):
            print("   Docker-specific fixes:")
            print("   1. Ensure camera devices are mapped: --device /dev/video0:/dev/video0")
            print("   2. Run with privileged mode: --privileged")
            print("   3. Check if camera is being used by another process")
            print("   4. Restart Docker container")
        else:
            print("   Host system fixes:")
            print("   1. Check if camera is connected and recognized by OS")
            print("   2. Try: sudo usermod -a -G video $USER")
            print("   3. Check camera permissions: ls -la /dev/video*")
            print("   4. Install camera drivers if needed")
    
    return 0 if working_camera is not None else 1

if __name__ == "__main__":
    sys.exit(main())
