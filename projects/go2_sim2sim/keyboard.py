"""MuJoCo 窗口的按住控制；X11 查询补充原生回调缺少的松键事件。"""

import ctypes as C
from ctypes.util import find_library
import threading

import glfw
import numpy as np


class X11Keys:
    """只在绑定的仿真窗口有焦点时读取键位；不截获或注入系统按键。"""
    def __init__(self):
        self.lib = C.CDLL(find_library("X11") or "libX11.so.6")
        self.lib.XOpenDisplay.argtypes, self.lib.XOpenDisplay.restype = [C.c_char_p], C.c_void_p
        self.lib.XCloseDisplay.argtypes = [C.c_void_p]
        self.lib.XGetInputFocus.argtypes = [C.c_void_p, C.POINTER(C.c_ulong), C.POINTER(C.c_int)]
        self.lib.XQueryKeymap.argtypes = [C.c_void_p, C.POINTER(C.c_char)]
        self.lib.XKeysymToKeycode.argtypes = [C.c_void_p, C.c_ulong]
        self.lib.XKeysymToKeycode.restype = C.c_ubyte
        self.display = self.lib.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("键盘按住控制需要 X11/XWayland，请从 Ubuntu 桌面终端启动")
        self.window = None
        self.codes = {key: self.lib.XKeysymToKeycode(self.display, symbol) for key, symbol in (
            (glfw.KEY_UP, 0xFF52), (glfw.KEY_DOWN, 0xFF54),
            (glfw.KEY_LEFT, 0xFF51), (glfw.KEY_RIGHT, 0xFF53), (glfw.KEY_SPACE, 0x20),
        )}

    def focus(self):
        window, revert = C.c_ulong(), C.c_int()
        self.lib.XGetInputFocus(self.display, C.byref(window), C.byref(revert))
        return window.value

    def bind_current_window(self):
        # 仅由 MuJoCo 自己的 key_callback 调用，事件来源即本次仿真窗口。
        if self.window is None:
            self.window = self.focus()

    def read(self):
        if self.window is None or self.focus() != self.window:
            return set(), False
        bits = C.create_string_buffer(32)
        self.lib.XQueryKeymap(self.display, bits)
        values = bits.raw
        held = {key for key, code in self.codes.items() if code and values[code // 8] & (1 << (code % 8))}
        return held, True

    def close(self):
        if self.display:
            self.lib.XCloseDisplay(self.display)
            self.display = None


class Keyboard:
    def __init__(self, command, command_cfg):
        self.command = np.array(command, dtype=float)
        self.cfg = command_cfg
        self.reset_requested = False
        self.lock = threading.Lock()
        self.sampler = None

    def enable_hold_control(self):
        self.sampler = X11Keys()
        self.command[:] = 0.

    def press(self, key):
        with self.lock:
            if self.sampler is not None:
                self.sampler.bind_current_window()
            if key == glfw.KEY_R:
                self.reset_requested = True
            elif key == glfw.KEY_SPACE:
                self.command[:] = 0.

    def read(self):
        with self.lock:
            if self.sampler is not None:
                held, focused = self.sampler.read()
                forward = int(glfw.KEY_UP in held) - int(glfw.KEY_DOWN in held)
                turn = int(glfw.KEY_LEFT in held) - int(glfw.KEY_RIGHT in held)
                if not focused or glfw.KEY_SPACE in held:
                    forward = turn = 0
                speed = self.cfg["keyboard_forward"] if forward >= 0 else self.cfg["keyboard_backward"]
                self.command[:] = [forward * speed, 0., turn * self.cfg["keyboard_turn"]]
            reset = self.reset_requested
            self.reset_requested = False
            return self.command.copy(), reset

    def close(self):
        with self.lock:
            self.command[:] = 0.
            if self.sampler is not None:
                self.sampler.close()
                self.sampler = None
