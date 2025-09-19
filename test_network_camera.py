#!/usr/bin/env python3
"""
Network Camera Test Script
Quick test to validate IP camera streams inside Docker container

Usage:
  # Test with environment variable
  python test_network_camera.py
  
  # Test with specific URL
  python test_network_camera.py rtsp://user:pass@192.168.1.100:554/stream1
  python test_network_camera.py http://192.168.1.101:8080/video
"""

import cv2
import sys
import os

def test_camera_url(url):
    """Test a specific camera URL"""
    print(f"🔍 Testing camera URL: {url}")
    
    try:
        # Choose backend based on URL type
        if url.startswith('rtsp://'):
            print("  Using FFMPEG backend for RTSP stream...")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        else:
            print("  Using default backend for HTTP/MJPEG stream...")
            cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            print("  ❌ Cannot open camera stream")
            return False
        
        print("  ✅ Camera stream opened successfully")
        
        # Try to read a frame
        print("  📸 Attempting to read frame...")
        ret, frame = cap.read()
        
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"  ✅ Frame read successful! Resolution: {width}x{height}")
            
            # Try to read a few more frames to test stability
            for i in range(5):
                ret, frame = cap.read()
                if not ret:
                    print(f"  ⚠️  Frame {i+2} failed to read")
                    break
            else:
                print("  ✅ Stream appears stable (read 5 frames successfully)")
            
            cap.release()
            return True
        else:
            print("  ❌ Cannot read frame from stream")
            cap.release()
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🎯 Network Camera Test Script")
    print("=" * 50)
    
    # Check if URL provided as argument
    if len(sys.argv) > 1:
        camera_url = sys.argv[1]
    else:
        # Get from environment variable
        camera_url = os.getenv("CAMERA_SOURCE", "0")
    
    if camera_url.isdigit():
        print(f"📱 Testing local camera device: {camera_url}")
        success = test_camera_url(int(camera_url))
    else:
        print(f"🌐 Testing network camera stream")
        success = test_camera_url(camera_url)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ SUCCESS: Camera test passed!")
        print("🎉 Your camera should work with the attendance system.")
    else:
        print("❌ FAILURE: Camera test failed!")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check if the camera URL is correct")
        print("2. Ensure the camera is accessible from this network")
        print("3. Verify camera credentials (user/password)")
        print("4. Check firewall settings")
        print("5. Try a different camera app or settings")
        
        print("\n📱 PHONE CAMERA APPS:")
        print("Android: 'IP Webcam' - http://PHONE_IP:8080/video")
        print("iOS: 'RTSP Camera' or 'Larix Broadcaster'")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
