def beat_location(frames_after_spline, window_size, cs_new, Fs):

    k = 0
    x_index = []#zeros()
    beat_point = []#zeros()
    index = 0
    find_o = 0
    window = 0

    while 1:
        if cs_new[find_o] == 0:
            break
        elif((cs_new[find_o] > 0 and cs_new[find_o + 1] < 0) or cs_new[find_o] < 0 and cs_new[find_o + 1] > 0):
            break
        else:
            find_o = find_o + 1

    for i in range(0, frames_after_spline+1, window_size):
        if i == 0:
            maxi = 0
            lower = find_o
        else:
            maxi = 0
            if k == 0:
                lower = window_size
            elif(window - index) < Fs / 5:
                lower = x_index(k) + Fs / 2
                if (lower >= frames_after_spline):
                    lower = frames_after_spline - 1

                check = i
                while check < lower:
                    if cs_new(check) > cs_new(check - 1) and cs_new(check) > cs_new(check + 1) and cs_new(check) > beat_point(k) and cs_new(check) > cs_new(check - 15) and cs_new(check) > cs_new(check + 15):
                        beat_point[k] = cs_new(check)
                        x_index[k] = check
                    else:
                        check = check + 1
            else:
                lower = i
        flag = 0
        window = i + window_size - 1
        if window - frames_after_spline > Fs / 5:
            break
        elif window > frames_after_spline:
            window = frames_after_spline

        for j in range(lower-1, window+1):
            if (j != frames_after_spline and j < frames_after_spline) and cs_new[j] > cs_new[j+1]:
                if j != 0 and cs_new[j] > cs_new[j - 1] and cs_new[j] > 0:
                    if maxi < cs_new[j]:
                        flag = 1
                        maxi = cs_new[j]
                        index = j

        if flag != 0:
            x_index.append(index)
            beat_point.append(cs_new[index])

            #k = k + 1;
            #x_index[k] = index;
            #beat_point[k] = cs_new[index];

    return beat_point, x_index
