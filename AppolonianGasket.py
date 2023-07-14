import sys
import heapq
import math
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Given an externally tangent circle with curvature k1 centered at the origin and a circle along the x-axis
# with curvature k2 and center (1/k1 + 1/k2), find the center of a circle tangent to both circles with curvature
# k3. These formulas were derived elsewhere.
def place_circle(k1, k2, k3):
    k_recip_sum = 1/k1 + 1/k2
    Y = 2 * math.sqrt((k1 + k2) * k3 + k1 * k2)
    X1 = 1/k1 - (8 * k1)/(4 * k1 * k1 + Y * Y)
    Y1 = (4 * Y) / (4 * k1 * k1 + Y * Y)
    X2 = 1/k1 + (8 * k2)/(4 * k2 * k2 + Y * Y)
    Y2 = (4 * Y) / (4 * k2 * k2 + Y * Y)
    
    if (X1 == 0):
        h = X1
        k = Y1 - 1/k3
    elif (Y1 == 0):
        h = (X1 + X2)/2
        k = 0
    else:    
        h = (X1 * Y2) * k_recip_sum / (X1 * Y2 - Y1 * X2 + Y1 * k_recip_sum)
        k = Y1 * (h - X1) / X1 + Y1
    return (h, k)

# Add two complex numbers, represented by tuples.
def add(z1, z2):
    return (z1[0] + z2[0], z1[1] + z2[1])

# Multiply two complex numbers, represented by tuples.
def mul(z1, z2):
    return (z1[0] * z2[0] - z1[1] * z2[1], z1[0] * z2[1] + z1[1] * z2[0])

# Use De Moivre's Theorem to find the square root of a complex number.
def sqrt(z):
    arg = math.atan2(z[1], z[0])/2
    mag = math.sqrt(math.sqrt(z[0] * z[0] + z[1] * z[1]))
    return (mag * math.cos(arg), mag * math.sin(arg))

# Use Descartes's Theorem to place a circle with curvature k4 tangent to circles
# k1, k2, and k3.
def find_last_circle_center(k1, k2, k3, k4, z1, z2, z3):
    z1 = (k1 * z1[0], k1 * z1[1])
    z2 = (k2 * z2[0], k2 * z2[1])
    z3 = (k3 * z3[0], k3 * z3[1])
    discrim = sqrt(add(add(mul(z1, z2), mul(z2, z3)), mul(z3, z1)))
    for d in [discrim, (-discrim[0], -discrim[1])]:
        z4 = add(add(z1, z2), add(z3, (2 * d[0], 2 * d[1])))
        
        a = add(add(z1, z2), add(z3, z4))
        a = mul(a, a)
        b = add(add(mul(z1, z1), mul(z2, z2)), add(mul(z3, z3), mul(z4, z4)))
        b = (2 * b[0], 2 * b[1])
        if (abs(a[0] - b[0])<1e-9 and abs(a[1] - b[1])<1e-9): return (z4[0]/k4, z4[1]/k4)

# Use Descartes's Theorem / Appolonian Group to find a new circle curvature & location
# given four curvatures and their corresponding centers.
def create_circle(a, b, c, d, z1, z2, z3, z4):
    d2 = 2 * (a + b + c) - d
    z5x = (2 * (a * z1[0] + b * z2[0] + c * z3[0]) - d * z4[0])/d2
    z5y = (2 * (a * z1[1] + b * z2[1] + c * z3[1]) - d * z4[1])/d2
    return (d2, (z5x, z5y))

# Check if a tuple x is "contained" within an array of tuples (i.e: sufficiently close to).
def contained(arr, x):
    for y in arr:
        if (abs(x[0] - y[0]) < 1e-9 and abs(x[1] - y[1]) < 1e-9): return True
    return False

# Seed has curvature (a, b, c, d) and must satisfy:
# = a should always be negative.
# - (a + b + c + d)**2 = 2(a**2 + b**2 + c**2 + d**2)
try:
    seed_curvatures = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
except Exception as _:
    print("Could not parse input seed.")
    sys.exit(1)

# Input parsing.
seed_curvatures = sorted(seed_curvatures)
a, b, c, d = seed_curvatures
if (a >= 0):
    print("No circle has negative curvature (there cannot be any inscribed circles).")
    sys.exit(1)
if (b < 0 or c < 0 or d < 0):
    print("There may only be one circle with negative curvature.")
    sys.exit(1)
if ((a + b + c + d)**2 != 2 * (a**2 + b**2 + c**2 + d**2)):
    print("Provided curvatures do not satisfy Descartes's Theorem.")
    sys.exit(1)

max_curvature = 100 * max(seed_curvatures)

# Place the circles at their centers.
seed_centers = [(0, 0)]
seed_centers.append((1/seed_curvatures[0] + 1/seed_curvatures[1], 0))
seed_centers.append(place_circle(seed_curvatures[0], seed_curvatures[1], seed_curvatures[2]))
seed_centers.append(find_last_circle_center(seed_curvatures[0], seed_curvatures[1],
                                            seed_curvatures[2], seed_curvatures[3],
                                            seed_centers[0], seed_centers[1], seed_centers[2]))
circles = {}
for i in range(4):
    if (seed_curvatures[i] in circles):
        circles[seed_curvatures[i]].append(seed_centers[i])
    else: circles[seed_curvatures[i]] = [seed_centers[i]]

# Use a priority queue to select groups of circles by the smallest maximum curvature in that
# group. Stop when the max curvature is reached (circles are too small to see).

queue = []
heapq.heappush(queue, (max(seed_curvatures), (seed_curvatures, seed_centers)))
while len(queue) != 0:
    val, (curvatures, centers) = heapq.heappop(queue)
    if (val > max_curvature): break
    a, b, c, d = curvatures
    z1, z2, z3, z4 = centers

    curv, cent = create_circle(a, b, c, d, z1, z2, z3, z4)
    if (curv not in circles or not contained(circles[curv], cent)):
        heapq.heappush(queue, (max((a, b, c, curv)), ((a, b, c, curv), (z1, z2, z3, cent))))
        if (curv not in circles): circles[curv] = [cent]
        else: circles[curv].append(cent)
    curv, cent = create_circle(a, b, d, c, z1, z2, z4, z3)
    if (curv not in circles or not contained(circles[curv], cent)):
        heapq.heappush(queue, (max((a, b, d, curv)), ((a, b, d, curv), (z1, z2, z4, cent))))
        if (curv not in circles): circles[curv] = [cent]
        else: circles[curv].append(cent)
    curv, cent = create_circle(a, c, d, b, z1, z3, z4, z2)
    if (curv not in circles or not contained(circles[curv], cent)):
        heapq.heappush(queue, (max((a, c, d, curv)), ((a, c, d, curv), (z1, z3, z4, cent))))
        if (curv not in circles): circles[curv] = [cent]
        else: circles[curv].append(cent)
    curv, cent = create_circle(b, c, d, a, z2, z3, z4, z1)
    curv2 = (b, c, d, curv)
    if (curv not in circles or not contained(circles[curv], cent)):
        heapq.heappush(queue, (max((b, c, d, curv)), ((b, c, d, curv), (z2, z3, z4, cent))))
        if (curv not in circles): circles[curv] = [cent]
        else: circles[curv].append(cent)

colors = [mcolors.CSS4_COLORS['lightcoral'], mcolors.CSS4_COLORS['lightblue'],
          mcolors.CSS4_COLORS['thistle'], mcolors.CSS4_COLORS['lightgreen'],
          mcolors.CSS4_COLORS['turquoise']]

# Plot all the circles using matplotlib
figure, axes = plt.subplots()
axes.set_facecolor(mcolors.CSS4_COLORS['linen'])
axes.set_aspect(1)
axes.set_xlim(1/seed_curvatures[0],-1/seed_curvatures[0])
axes.set_ylim(1/seed_curvatures[0],-1/seed_curvatures[0])
for curvature in circles:
    for center in circles[curvature]:
        c = plt.Circle(center, radius = 1/abs(curvature), fill = curvature>0)
        c.set_edgecolor(mcolors.CSS4_COLORS['black'])
        c.set_facecolor(colors[random.randint(0, len(colors) - 1)])
        axes.add_artist(c)
plt.title("Appolonian Gasket: " + str(seed_curvatures))
plt.show()