# 3-frame token palette. Chosen so the two ADJACENT mixes are clean:
#   C1+C2 -> violet, C2+C3 -> coral.  C1 and C3 never mix directly on the ramp.
C1 = (0.216, 0.404, 0.839)   # f-1  indigo-blue   #376ED6
C2 = (0.788, 0.188, 0.557)   # f    magenta       #C9308E
C3 = (0.855, 0.510, 0.086)   # f+1  amber         #E8912B

INK      = '#12203A'
INK_SOFT = '#4A5A76'
RULE     = '#C8D2E2'
PAPER    = '#FFFFFF'
BAND_A   = '#F5F8FE'
BAND_B   = '#F8F6FD'
BOX_FILL = '#EAF0FC'
BOX_EDGE = '#5B7FC7'
LORA_FILL= '#E8F6EC'
LORA_EDGE= '#2E9E5B'
FREE_FILL= '#FFF3E2'
FREE_EDGE= '#D98319'

def hexc(c):
    return '#%02x%02x%02x' % tuple(max(0, min(255, int(round(v * 255)))) for v in c)

def mix(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))

WHITE = (1.0, 1.0, 1.0)
