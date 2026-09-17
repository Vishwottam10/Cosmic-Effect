"""Quick test to verify Taichi GUI window appears on screen."""
import taichi as ti

ti.init(arch=ti.cpu)

# Test 1: Try GGUI (modern API)
print("Attempting to open Taichi GGUI window...", flush=True)
try:
    window = ti.ui.Window("Test GGUI", (640, 480), vsync=True)
    canvas = window.get_canvas()
    for frame in range(120):  # ~2 seconds at 60fps
        canvas.set_background_color((0.1, 0.2, 0.5))
        window.show()
        if not window.running:
            break
    print("GGUI window test PASSED", flush=True)
    # window closes when script ends
except Exception as e:
    print(f"GGUI failed: {e}", flush=True)
    print("Falling back to legacy ti.GUI...", flush=True)
    
    # Test 2: Try legacy GUI
    gui = ti.GUI("Test Legacy GUI", (640, 480))
    for frame in range(120):
        gui.clear(0x112244)
        gui.text("Cosmic Butterfly Effect", pos=(0.2, 0.5), color=0xFFFFFF, font_size=30)
        gui.show()
        if gui.running is False:
            break
    print("Legacy GUI test PASSED", flush=True)
