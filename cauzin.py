#!/usr/bin/env python

import sys
import array
from math import sin, cos, tan, pi as PI, tau as TAU

from PIL import Image
import numpy

SKIP_EXTRA_ROWS = 10
 
def find_peaks(values : list) -> list:
    peaks = []
    troughs = []

    # scan for first variation
    start = 0
    up = False
    for num in range(len(values)-1):
        if values[num+1] > values[num]:
            troughs.append(0)
            up = True
            start = num + 1
            break
        elif values[num+1] < values[num]:
            peaks.append(0)
            start = num + 1
            break

    # scan for additional variations
    last = start
    lastval = values[start]
    for num in range(start, len(values)-1):
        if lastval != values[num]:
            last = num
            lastval = values[num]
        if values[num+1] > values[num]:
            #trend up
            if not up:
                up = True
                troughs.append(last + ((num - last) // 2))
        elif values[num+1] < values[num]:
            #trend down
            if up:
                peaks.append(last + ((num - last) // 2))
                up = False

    # if it's just a flatline, return nothing
    if len(peaks) == 0 and len(troughs) == 0:
        return [], []

    # determine the ending trend
    if up and (len(peaks) == 0 or peaks[-1] != len(values) - 1):
        peaks.append(len(values)-1)
    elif not up and (len(troughs) == 0 or troughs[-1] != len(values) - 1):
        troughs.append(len(values)-1)

    return peaks, troughs
 
def interpolate(values):
    distance = len(values) - 1
    firstval = float(values[0])
    secondval = float(values[-1])
    diff = secondval - firstval
    slope = diff / distance
    for i in range(distance - 1):
        values[i + 1] = firstval + (slope * (i + 1))

def smooth_peaks(values, peaks):
    for num in range(len(peaks)-1):
        first = peaks[num]
        second = peaks[num+1]
        interpolate(values[first:second+1])

def image_to_ndarray(image : Image) -> numpy.ndarray:
    if image.mode != "L":
        # can't covnert paletted directly to grayscale...
        if image.mode == "P":
            image = image.convert(mode = "RGB")
        image = image.convert(mode = "L", matrix = (1/3, 1/3, 1/3, 0.0))

    #hist = image.histogram()

    #while True:
    #    peaks, _ = find_peaks(hist)
    #    if len(peaks) <= 2:
    #        break
    #    if peaks[0] != 0:
    #        peaks.insert(0, 0)
    #    if peaks[-1] != 255:
    #        peaks.append(255)

    #    smooth_peaks(hist, peaks)

    #if len(peaks) < 2:
    #    raise ValueError("Only 1 peak found, scan is too low quality?  Or the peak finding algorithm sucks.")

    #lowest = peaks[0] / 255.0
    #highest = peaks[1] / 255.0

    #print(f"Peaks: {lowest}, {highest}")

    image = numpy.ndarray(shape = (image.height, image.width), dtype = numpy.uint8, buffer = image.tobytes()) / 255.0
    # scale to peak range
    #if lowest > 0.0 or highest < 1.0:
    #    image = (image - lowest) * (1.0 / (highest - lowest))

    return image

def ndarray_to_image(image : numpy.ndarray) -> Image:
    return Image.frombytes(mode = "L", size = (image.shape[1], image.shape[0]), data = numpy.uint8(image * 255.0))

def print_all_array(values):
    for value in values:
        print(f"{value} ", end='')
    print()

def crop_ndarray(ndarray, start_row, rows, start_column, columns):
    new_array = numpy.ndarray(shape = (rows, columns), dtype = ndarray.dtype)
    for num, row in enumerate(ndarray[start_row:start_row+rows]):
        new_array[num] = row[start_column:start_column+columns]
    return new_array

def draw_line_h(image, x, y, slope, length):
    print(f"{x} {y} {slope} {length}")
    x = round(x)
    for i in range(int(length)):
        px = x + i
        py = y + (slope * i)
        py_i = int(py)
        py_f = py - py_i
        py_f1 = 1.0 - py_f

        try:
            image[py_i][px] = ((1.0 - image[py_i][px]) * py_f1) + (image[py_i][px] * py_f)
        except IndexError:
            pass
        try:
            image[py_i+1][px] = ((1.0 - image[py_i+1][px]) * py_f) + (image[py_i+1][px] * py_f1)
        except IndexError:
            pass

def sample_pixel(image, x, y):
    x_i = int(x)
    x_f = x - x_i
    x_f1 = 1.0 - x_f
    y_i = int(y)
    y_f = y - y_i

    # return "background color" for out of bounds
    p00 = 1.0
    try:
        p00 = image[y_i][x_i]
    except IndexError:
        pass
    p10 = 1.0
    try:
        p10 = image[y_i][x_i+1]
    except IndexError:
        pass
    p01 = 1.0
    try:
        p01 = image[y_i+1][x_i]
    except IndexError:
        pass
    p11 = 1.0
    try:
        p11 = image[y_i+1][x_i+1]
    except IndexError:
        pass

    return ((((p00 * x_f1) + (p10 * x_f)) * (1.0 - y_f)) +
            (((p01 * x_f1) + (p11 * x_f)) * y_f))

def sample_strip(image, x, y, angle, count=2**31):
    width = image.shape[1]
    height = image.shape[0]

    x_rate = cos(angle)
    x_limit = 2**31 # if not moving in X direction, no limit
    if x_rate < 0.0:
        x_limit = -int(x / x_rate)
    elif x_rate > 0.0:
        x_limit = int((width - x - 1) / x_rate)

    y_rate = sin(angle)
    y_limit = 2**31
    if y_rate < 0.0:
        y_limit = -int(y / y_rate)
    elif y_rate > 0.0:
        y_limit = int((height - y - 1) / y_rate)

    limit = min(x_limit, y_limit, count)
    strip = numpy.ndarray(shape = (limit,), dtype = float)

    for i in range(limit):
        strip[i] = sample_pixel(image, x + (i * x_rate), y + (i * y_rate))

    return strip

def find_transitions(strip, count=2**31):
    # detect edges by taking difference of each 2 points
    # transitions end up being biased to the left 0.5
    for num, val in enumerate(strip[:-1]):
        strip[num] = abs(strip[num+1] - val)
    # extra value
    strip[-1] = 0.0

    peaks, troughs = find_peaks(strip)
    if len(peaks) == 0:
        return [], []

    # normalize
    strip *= 1.0 / max(strip)

    # use the found peaks to find where transition points are
    # use values between peaks and troughs to "adjust" the
    # transition point to closer to where it should be
    # this might be nonsensical...
    transitions = []
    start = 0
    if peaks[0] == 0:
        # starting on a peak
        transitions.append(0.0)
        # first peak
        for off, i in enumerate(range(0, troughs[0])):
            transitions[0] += float(off * strip[i])
        start = 1
    # remaining peaks
    for num, peak in enumerate(peaks[start:]):
        transitions.append(peak)
        for off, i in enumerate(range(peak, troughs[num]+1, -1)):
            transitions[-1] -= float(off * strip[i])
        try:
            troughs[num+1]
        except IndexError:
            continue
        for off, i in enumerate(range(peak, troughs[num+1]-1)):
            transitions[-1] += float(off * strip[i])

    # correct for offset, detected edges are between samples
    for num in range(len(transitions)):
        transitions[num] += 0.5

    # record the magnitude of each transition
    magnitudes = []
    for num, transition in enumerate(transitions):
        i = int(transition)
        f = transition - i
        s0 = strip[i]
        s1 = 1.0
        try:
            s1 = strip[i+1]
        except IndexError:
            pass
        magnitudes.append(float((s0 * (1.0 - f)) + (s1 * f)))

    return transitions, magnitudes

def find_first_last_values(values, factor, whichfirst=1, whichlast=1):
    max_value = max(values) / 10.0

    first = []
    if whichfirst != 0:
        for num, value in enumerate(values):
            if value > max_value:
                first.append(num)
                whichfirst -= 1
                if whichfirst < 0:
                    continue
                elif whichfirst == 0:
                    break

    last = []
    if whichlast >= 0:
        for num, value in enumerate(reversed(values)):
            if value > max_value:
                last.append(len(values) - num - 1)
                whichlast -= 1
                if whichlast < 0:
                    continue
                elif whichlast == 0:
                    break

    return first, last

def find_shortest_angle(image, x, y, start, divisions, angle_divisions, fall_allowance=0.0):
    last_dists = [None, None]
    minangle = 0.0
    minfirst = 0.0
    mindist = 2**31
    for i in range(divisions, -1, -1):
        #strip = sample_strip(image, x, y, PI * (0.0+(i/angle_divisions)))
        strip = sample_strip(image, x, y, PI * (start+(i/angle_divisions)))
        transitions, magnitudes = find_transitions(strip)

        first, last = find_first_last_values(magnitudes, 0.1)
        first = transitions[first[0]]
        last = transitions[last[0]]

        dist = last - first
        if fall_allowance > 0.0:
            # just search for the lowest angle
            if last_dists[0] is not None and last_dists[1] is not None and \
               dist < (last_dists[0] + last_dists[1]) / 2.0 * fall_allowance:
                return minangle, minfirst, mindist
            last_dists[1] = last_dists[0]
            last_dists[0] = dist
        if dist < mindist:
            minangle = i
            minfirst = first
            mindist = dist

    return minangle, minfirst, mindist

def find_code_top_edge(image, x, y, direction):
    if direction == 1:
        # top right found, need to find the left edge 
        # TODO: UNTESTED!
        angle, _ , _ = find_shortest_angle(image, x, y, 300.0, 100, 400, 0.9)
        start_angle = angle
        angle, start, length = find_shortest_angle(image, x, y, (400.0 + angle) / 100.0, 100, 40000, 0.99)
        angle = (start_angle + (angle / 100.0)) / 400.0 * PI
    else:
        # at the top left edge, need to find the right edge
        angle, _, _ = find_shortest_angle(image, x, y, 0.0, 100, 400, 0.9)
        start_angle = angle
        angle, start, length = find_shortest_angle(image, x, y, angle / 100.0, 100, 40000, 0.999)
        angle = (start_angle + (angle / 100.0)) / 400.0 * PI
 
    return angle

# I've only seen the 4 and 5 nybble patterns so far, the rest are guessed
PATTERNS = (
    (-1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0,
      1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
     -1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,
      1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
     -1.0,  1.0, -1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,
      1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
     -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,
      1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
     -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,
      1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
     -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,
      1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
     -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0)
)

def get_top_edge_info(image, x, y, angle):
    strip = sample_strip(image, x, y, angle) * 2.0 - 1.0
    mask_signal = numpy.ndarray(shape = (strip.shape[0],), dtype = float)

    newstrip = strip.copy()
    transitions, magnitudes = find_transitions(newstrip)

    first, last = find_first_last_values(magnitudes, 0.1)
    first = round(transitions[first[0]])
    last = round(transitions[last[0]])
    length = last - first

    scores = []
    for pattern in PATTERNS:
        if length < len(pattern):
            # can't possibly be valid
            return None
        for i in range(length):
            alternation = (i / length * len(pattern))
            mask_signal[i] = pattern[int(alternation)]
        scores.append(float(sum(abs(strip[first:last] + mask_signal[:length]))) / length / 2.0)

    score = max(scores)
    bits = len(PATTERNS[scores.index(score)])

    return (bits + 1) * 2, score, length

def find_initial_direction(image, x, y):
    # determine which direction the code is found
    # cast out at 45 degree angles.  if a strip is over 45 degrees angle this will fail
    strip = sample_strip(image, x, y, PI * (3/4))
    left_transitions = find_transitions(strip)
    strip = sample_strip(image, x, y, PI * (1/4))
    right_transitions = find_transitions(strip)

    if len(right_transitions) > len(left_transitions):
        # proceeding towards the right
        return 1

    # proceeding towards the left
    return -1

def find_top_image_edge(image):
    direction = 0
    bits = 0

    angle = 0.0
    start = 0
    xes = []
    for y, row in enumerate(image):
        # determine if there's any data on this row
        strip = sample_strip(image, 0.0, y, 0.0)
        transitions, magnitudes = find_transitions(strip)
        if len(transitions) == 0:
            # if not, on to next
            continue
        first_transition_index, _ = find_first_last_values(magnitudes, 0.1, whichlast=0)
        x = transitions[first_transition_index[0]]
        if direction == 0:
            direction = find_initial_direction(image, x, y)
        # try this row
        try_angle = find_code_top_edge(image, x, y, direction)
        bits, score, length = get_top_edge_info(image, x, y, try_angle)
        if score < 0.85:
            # if this row's code score is too low, try the next
            continue
        if direction == 1:
            # find the left side (TODO: UNTESTED)
            angle = try_angle - PI
            start = y + (sin(angle) * length)
            xes.append(x + (cos(angle) * length))
        else:
            # note the rough angle for tracking the initial strip
            # finding it for every row would be super slow and unnecessary
            angle = try_angle
            start = y + 1
            xes.append(x)
        break

    print(f"Initial angle: {angle}")

    end = 0
    for y, row in enumerate(image[start:]):
        strip = sample_strip(image, 0.0, start+y, 0.0)
        transitions, magnitudes = find_transitions(strip)
        first_transition_index, _ = find_first_last_values(magnitudes, 0.1, whichlast=0)
        x = transitions[first_transition_index[0]]
        new_bits, score, length = get_top_edge_info(image, x, start+y, angle)
        if score < 0.8:
            # if this row's code score is too low, maybe at the end?
            end = y
            break
        if new_bits != bits:
            print("Acceptable code but different code density found!?")
            break
        xes.append(x)

    slopes = numpy.ndarray(shape = (len(xes)-1,), dtype = float)
    for num, x in enumerate(xes[:-1]):
        slopes[num] = xes[num+1] - x

    avg = slopes.mean()
    angle = -tan(avg)

    start += sin(angle) * xes[0]
    end += sin(angle) * xes[-1]

    # find center of left bar
    strip = sample_strip(image, 0.0, start, angle)
    transitions, magnitudes = find_transitions(strip)
    left_bar_indices, _ = find_first_last_values(magnitudes, 0.1, whichfirst=2, whichlast=0)
    bar_center = (transitions[left_bar_indices[0]] + transitions[left_bar_indices[1]]) / 2.0

    # sample from the bar towards the top of the image
    strip = sample_strip(image, cos(angle) * bar_center, start + (sin(angle) * bar_center), angle - (PI / 2))
    transitions, magnitudes = find_transitions(strip)
    # get the transition and find find the extra distance from this start to the edge
    left_bar_top_index, _ = find_first_last_values(magnitudes, 0.1, whichlast=0)
    startdiff = sin(angle - (PI / 2)) * transitions[left_bar_top_index[0]]
    start += startdiff

    return start, end, angle, bits

LEFT_TRACK_ODD   = [0, 0, 1, 0, 1]
RIGHT_TRACK_ODD  = [1, 1, 0, 0, 1]
LEFT_TRACK_EVEN  = [0, 0, 1, 1, 0]
RIGHT_TRACK_EVEN = [1, 1, 0, 0, 0]

def bindecode(stripdata, odd):
    # check for proper strip edges
    if odd:
        if stripdata[:5] != LEFT_TRACK_ODD or \
           stripdata[-5:] != RIGHT_TRACK_ODD:
            return None
    else:
        if stripdata[:5] != LEFT_TRACK_EVEN or \
           stripdata[-5:] != RIGHT_TRACK_EVEN:
            return None

    # decode dibits in to bits
    bits = stripdata[5:-5]
    for i in range(len(bits) // 2):
        if bits[i*2] == 1 and bits[(i*2)+1] == 0:
            bits[i] = 0
        elif bits[i*2] == 0 and bits[(i*2)+1] == 1:
            bits[i] = 1
        else:
            return None
    bits = bits[:len(bits) // 2]

    # calculate parity
    even_parity = 0
    odd_parity = 0
    for i in range(len(bits) // 2):
        even_parity += bits[(i*2)]
        odd_parity += bits[(i*2) + 1]
    even_parity %= 2
    odd_parity %= 2

    # check parity
    if even_parity == 0 or odd_parity == 0:
        return None

    # chop off parity bits and get only data body
    return bits[1:-1]

def bitpack(data):
    binarydata = numpy.ndarray(shape = (len(data) // 8,), dtype = numpy.uint8)
    for i in range(len(data) // 8):
        binarydata[i] = (
             data[ i * 8     ]       |
            (data[(i * 8) + 1] << 1) |
            (data[(i * 8) + 2] << 2) |
            (data[(i * 8) + 3] << 3) |
            (data[(i * 8) + 4] << 4) |
            (data[(i * 8) + 5] << 5) |
            (data[(i * 8) + 6] << 6) |
            (data[(i * 8) + 7] << 7)
        )

    return binarydata

if __name__ == "__main__":
    image = image_to_ndarray(Image.open(sys.argv[1]))
    print(f"Image rows: {image.shape[0]}  columns: {image.shape[1]}")

    # find top edge of code
    start, code_height, angle, bits = find_top_image_edge(image)
    vclock = code_height / 6.0
    print(f"Start row: {start}  Code height: {code_height}  Initial angle: {angle} ({angle/PI*360.0} deg)  Bits: {bits}")
    #draw_line_h(image, 0.0, start, angle, image.shape[1])
    #draw_line_h(image, 0.0, start+code_height, angle, image.shape[1])
    #ndarray_to_image(image).save("test.bmp")

    start_y = start+code_height

    total_rows = 0
    new_image = numpy.ndarray(shape = (image.shape[0], bits), dtype = float)
    avg_advance = -1
    expect_odd = True
    trying_alternate = False
    trying_angle = False
    last_good = 0
    last_line_decoded = 0
    bads = 0
    goods = 0
    y = 0
    bit_strips = []
    decoded_strips = []
    while True:
        good = True
        strip_bits = []

        strip = sample_strip(image, 0.0, start_y+y, angle)
        edges = strip.copy()
        transitions, magnitudes = find_transitions(edges)
        if len(transitions) == 0:
            good = False
        else:
            valid_transitions, _ = find_first_last_values(magnitudes, 0.1, whichfirst=-1, whichlast=0)
            #print(valid_transitions)
            #for i in range(len(valid_transitions)-1):
            #    print(f"{transitions[valid_transitions[i+1]] - transitions[valid_transitions[i]]} ", end='')
            #print()
            width = transitions[valid_transitions[-1]] - transitions[valid_transitions[0]]
            find_bits = bits
            if expect_odd:
                find_bits -= 1
            approx_bit_width = width / find_bits
            max_error = approx_bit_width * 0.1
            #print(f"{y} {width} {find_bits} {approx_bit_width} {max_error}")
            #print(len(valid_transitions))
            # try to sample bit ranges in to floats
            for i in range(len(valid_transitions)-1):
                start_x = transitions[valid_transitions[i]]
                end_x = transitions[valid_transitions[i+1]]
                diff = end_x - start_x
                width_bits = diff / approx_bit_width
                num_bits = round(width_bits)
                error = abs(width_bits - round(width_bits))
                #print(f"{error}-{round(width_bits)} ", end='')
                if i < len(valid_transitions) - 3:
                    if num_bits < 1 or num_bits > 2:
                        good = False
                        break
                else:
                    if num_bits < 2 or num_bits > 3:
                        good = False
                        break
                if error > max_error:
                    good = False
                    break
                num = strip[int(start_x)] * (1.0 - (start_x - int(start_x)))
                num += sum(strip[int(start_x)+1:int(end_x)])
                num += strip[int(end_x)+1] * (end_x - int(end_x))
                num /= end_x - start_x
                for j in range(num_bits):
                    strip_bits.append(float(num))
                #print(f"{num} ", end='')
            #print()

        if good:
            # try to convert floats in to binary
            bits_sorted = sorted(strip_bits)
            largest_diff = 0.0
            largest_i = 0
            for i in range(len(bits_sorted)-1):
                diff = bits_sorted[i+1] - bits_sorted[i]
                if diff > largest_diff:
                    largest_diff = diff
                    largest_i = i
            if largest_i+1 > len(bits_sorted)-1:
                #print(bits_sorted)
                break
            transition = (bits_sorted[largest_i] + bits_sorted[largest_i+1]) / 2.0
            for i in range(len(strip_bits)):
                if strip_bits[i] < transition:
                    strip_bits[i] = 0
                else:
                    strip_bits[i] = 1
            if expect_odd:
                strip_bits.append(1)
            #print(strip_bits)
            if len(strip_bits) != bits:
                # unexpected number of bits, no good
                good = False

        if not good:
            print("BAD")
            # if previous decoding steps resulted in a not good state
            # decode failed
            if last_good > 0:
                # only try to recover if sync had already been acquired
                bads += 1
                if bads == int(vclock):
                    # try alternative strip
                    expect_odd = not expect_odd
                    trying_alternate = True
                    y = last_good
                elif bads == int(vclock * 2.0):
                    # try angle resync (find_shortest_angle)
                    y = last_good
                elif bads == int(vclock * 4.0):
                    # give up, or done TODO: wrap up regardless as if completed
                    break
        else:
            print("GOOD")
            new_image[y] = strip_bits
            last_good = y
            bads = 0
            goods += 1

            if trying_alternate:
                trying_alternate = False
                # tried alternate code line and found a good code line, so done with the last one

                # check available bit strips
                # throw out ones with the wrong start/end patterns or bad parity
                # if no candidates, fail (shouldn't happen?)
                # if 1 candidate, select that
                # if more, count how many of each unique candidate and pick the one with the most
                # if ambiguous, fail
                # in case of failure, go back to last decoded strip and try to resync angle
                for bit_strip in range(len(bit_strips)):
                    # invert expect_odd since this will be decoding the previous set
                    # while the rest of the logic is working on the present
                    bit_strips[bit_strip] = bindecode(bit_strips[bit_strip], not expect_odd)

                unique = {}
                for bit_strip in range(len(bit_strips)):
                    found = False
                    for item in unique.keys():
                        if bit_strips[item] == bit_strips[bit_strip]:
                            unique[item] += 1
                            found = True
                            break
                    if not found:
                        unique[bit_strip] = 1

                if len(unique) == 0:
                    # found no viable values, TODO: try angle resync
                    break
                elif len(unique) == 1:
                    # only 1 value, just use the first/only instance
                    for key in unique.keys():
                        decoded_strips.extend(bit_strips[key])
                        break
                else:
                    # find the highest value and hope it's unique, if not, TODO: try angle resync
                    highest = 0
                    highest_key = 0
                    highest_count = 0
                    for key in unique.keys():
                        if unique[key] == highest:
                            highest_count += 1
                        elif unique[key] > highest:
                            highest = unique[key]
                            highest_key = key
                            highest_count = 1

                    if highest_count > 1:
                        # ambiguous choice TODO: try angle resync
                        break

                    decoded_strips.extend(bit_strips[highest_key])

                last_line_decoded = y
                bit_strips = []

            bit_strips.append(strip_bits)

        y += 1
        if start_y+y >= image.shape[0]:
            if len(bit_strips) > 0:
                # if remaining bit_strips, try decode those as above
                pass
            break

    data = bitpack(decoded_strips)
    with open("out.bin", 'wb') as outfile:
        outfile.write(data.tobytes())
    #ndarray_to_image(new_image[:y]).save("test.bmp")
    #ndarray_to_image(new_image).save("test.bmp")

    #length = cos(angle) * width


    # find first clock
