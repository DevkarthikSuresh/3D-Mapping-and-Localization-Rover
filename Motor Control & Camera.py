import RPi.GPIO as GPIO
from time import sleep
import subprocess
import os
from pynput import keyboard

# --- 1. Motor Hardware Setup ---
in1, in2, ena = 24, 23, 25
in3, in4, enb = 17, 27, 18

GPIO.setmode(GPIO.BCM)
GPIO.setup([in1, in2, in3, in4, ena, enb], GPIO.OUT)
GPIO.output([in1, in2, in3, in4], GPIO.LOW)

pwm_a = GPIO.PWM(ena, 1000)
pwm_b = GPIO.PWM(enb, 1000)
pwm_a.start(45) 
pwm_b.start(45)

# --- 2. Start the FPV Preview ---
print("Waking up Camera Module v3...")
# This opens the live window using the system-level rpicam-vid
preview_process = subprocess.Popen([
    'rpicam-vid', '-t', '0', '--inline', '--width', '1280', '--height', '720', '--preview', '0,0,1280,720'
])

# --- 3. Control Logic ---
img_count = 0

def stop_motors():
    GPIO.output([in1, in2, in3, in4], GPIO.LOW)

def on_press(key):
    global img_count
    try:
        # Check for WASD or Arrow Keys
        if key == keyboard.Key.up or (hasattr(key, 'char') and key.char == 'w'):
            GPIO.output([in1, in3], GPIO.HIGH); GPIO.output([in2, in4], GPIO.LOW)
        elif key == keyboard.Key.down or (hasattr(key, 'char') and key.char == 's'):
            GPIO.output([in1, in3], GPIO.LOW); GPIO.output([in2, in4], GPIO.HIGH)
        elif key == keyboard.Key.left or (hasattr(key, 'char') and key.char == 'a'):
            GPIO.output([in1, in4], GPIO.LOW); GPIO.output([in2, in3], GPIO.HIGH)
        elif key == keyboard.Key.right or (hasattr(key, 'char') and key.char == 'd'):
            GPIO.output([in1, in4], GPIO.HIGH); GPIO.output([in2, in3], GPIO.LOW)
            
        # Capture Frame for Gaussian Splatting
        elif key == keyboard.Key.space:
            stop_motors() # Critical: avoid motion blur
            sleep(0.3)
            fname = f"frame_{img_count:04d}.jpg"
            # Direct system call for the photo
            subprocess.run(['rpicam-still', '-o', fname, '--immediate', '--nopreview'])
            print(f"📸 CAPTURED: {fname}")
            img_count += 1
            
    except Exception as e:
        pass # Ignore weird key combos

def on_release(key):
    stop_motors() # Stop as soon as you let go
    if key == keyboard.Key.esc:
        return False

# --- 4. Print Instructions & Run ---
print("\n" + "="*40)
print("       MOBILE GS ROVER PILOT")
print("="*40)
print(" DRIVE CONTROLS:")
print("  [W] or [UP]    - Move Forward")
print("  [S] or [DOWN]  - Move Backward")
print("  [A] or [LEFT]  - Rotate Left")
print("  [D] or [RIGHT] - Rotate Right")
print("\n CAPTURE CONTROL:")
print("  [SPACEBAR]     - Take Photo for GS")
print("\n SYSTEM:")
print("  [ESC]          - Emergency Stop & Exit")
print("="*40)
print("\nKEEP THIS TERMINAL FOCUSED TO DRIVE!\n")

try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
finally:
    # Cleanup
    stop_motors()
    pwm_a.stop(); pwm_b.stop()
    GPIO.cleanup()
    preview_process.terminate() # Close the camera window
    print("\n[!] Hardware released. Pilot session ended.")
