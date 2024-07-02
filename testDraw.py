import cv2 as cv
import numpy as np
from threading import Lock

cell_size = 1000
atomic_cnts = np.zeros((2, cell_size, cell_size), dtype=np.ulonglong)
lock = Lock()

def bresenham(r1, c1, r2, c2):
    atomic_cnts[1, r2, c2] += 1
    if c1 == c2:
        if r1 > r2:
            r1, r2 = r2, r1
        while r1 <= r2:
            with lock:
                atomic_cnts[0, r1, c1] += 1
            r1 += 1
    else:
        if c1 > c2:
            c1, c2 = c2, c1
            r1, r2 = r2, r1
        if r1 == r2:
            while c1 <= c2:
                with lock:
                    atomic_cnts[0, r1, c1] += 1
                c1 += 1
        else:
            if r1 > r2:
                r2 = r1 + (r1 - r2)
                dr = r2 - r1
                dc = c2 - c1
                if dr <= dc:
                    r0 = r1
                    p = 2 * dr - dc
                    while c1 <= c2:
                        with lock:
                            atomic_cnts[0, r0 - (r1 - r0), c1] += 1
                        c1 += 1
                        if p < 0:
                            p += 2 * dr
                        else:
                            p += 2 * dr - 2 * dc
                            r1 += 1
                else:
                    dr, dc = dc, dr
                    c1, r1 = r1, c1
                    c2, r2 = r2, c2
                    p = 2 * dr - dc
                    c0 = c1
                    while c1 <= c2:
                        with lock:
                            atomic_cnts[0, c0 - (c1 - c0), r1] += 1
                        c1 += 1
                        if p < 0:
                            p += 2 * dr
                        else:
                            p += 2 * dr - 2 * dc
                            r1 += 1
            else:
                dr = r2 - r1
                dc = c2 - c1
                if dc >= dr:
                    p = 2 * dr - dc
                    while c1 <= c2:
                        with lock:
                            atomic_cnts[0, r1, c1] += 1
                        c1 += 1
                        if p < 0:
                            p += 2 * dr
                        else:
                            p += 2 * dr - 2 * dc
                            r1 += 1
                else:
                    dr, dc = dc, dr
                    c1, r1 = r1, c1
                    c2, r2 = r2, c2
                    p = 2 * dr - dc
                    while c1 <= c2:
                        with lock:
                            atomic_cnts[0, c1, r1] += 1
                        c1 += 1
                        if p < 0:
                            p += 2 * dr
                        else:
                            p += 2 * dr - 2 * dc
                            r1 += 1

def drawOccupancyMap(canvas):
    for i in range(cell_size):
        for j in range(cell_size):
            if atomic_cnts[0][i][j] == 0:
                continue
            prob = 100 - (atomic_cnts[1][i][j] * 100) // atomic_cnts[0][i][j] # 비어있는 확률
            if prob < 20:
                cv.circle(canvas, (j, i), 0, (0, 0, 0), 5)
            elif prob > 80:
                cv.circle(canvas, (j, i), 0, (255, 255, 255), -1)

def main():
    canvas = np.full((cell_size, cell_size, 3), (120, 120, 120), dtype=np.uint8) # Creating a blank canvas
    res = 0.01 # m/cell

    c0 = cell_size // 2 + int(1.0 / res)  # x
    r0 = cell_size // 2 - int(-2.1 / res) # y

    arr = [[2, 3.5], [-1, 4], [-4, -3], [3, -2.2]]
    for data in arr:
        c = cell_size // 2 + int(data[0] / res)  # x
        r = cell_size // 2 - int(data[1] / res) # y

        bresenham(r0, c0, r, c)

        drawOccupancyMap(canvas)
        cv.imshow("Canvas", canvas)
        cv.waitKey(2000)

if __name__ == "__main__":
    main()
