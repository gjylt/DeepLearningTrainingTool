from ctypes import *

# g++ -shared - Wl, -soname, post_phase - o post_phase.so - fPIC post_c/tools.cpp

predlist = [2, 2, 2, 2, 2, 3, 5, 4, 4, 4, 4, 4,
            4, 6, 6, 6, 1, 1, 2, 2, 2, 2, 2, 2, 6, 6, 6]
arr = (c_int*len(predlist))(*predlist)
phase_post_process_C = CDLL(
    '/root/Task/Algorithm/post_phase_c.so')

phase_post_process_C.runThis(arr, len(predlist))

print(list(arr))
