"""Read PID creation time and executable without extra process-monitor packages."""
import os
from pathlib import Path


def process_identity(pid):
    if os.name != 'nt':
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            return dict(pid=int(pid), create_time=int(stat[19]), executable=str(Path(f'/proc/{pid}/exe').resolve()))
        except (FileNotFoundError, ProcessLookupError):
            return None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        code = ctypes.get_last_error()
        if code in (87, 1168):
            return None
        raise OSError(code, f'Cannot inspect PID {pid}; refusing to assume it is dead')
    try:
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if exit_code.value != 259:
            return None
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(x) for x in (created, exited, system, user)]):
            raise ctypes.WinError(ctypes.get_last_error())
        size = wintypes.DWORD(32768); buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(pid=int(pid), create_time=(created.dwHighDateTime << 32)|created.dwLowDateTime, executable=buffer.value)
    finally:
        kernel.CloseHandle(handle)
