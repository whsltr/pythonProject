import numpy as np


# test the distribution function
def read_timeseries(path, i):
    file = 'timeseries' + str(i) + '.dat'
    file_name = path + file
    f = open(file_name)
    t = []
    Ex = []
    Ey = []
    Ez = []
    Bx = []
    By = []
    Bz = []
    B_total = []
    for line in f.readlines():
        line = line.split()
        if len(line) != 7:
            break
        t.append(float(line[0]))
        Ex.append(float(line[1]))
        Ey.append(float(line[2]))
        Ez.append(float(line[3]))
        Bx.append(float(line[4]))
        By.append(float(line[5]))
        Bz.append(float(line[6]))
        B_total.append(float(line[4]) + float(line[5]) + float(line[6]))
    f.close()
    return t, Ex, Ey, Ez, Bx, By, Bz, B_total


# def read_phase():
#     file = 'phase.dat'
#     file_name = '/home/ck/Documents/hybrid2D/practice/data_parllel/' + file
#     f = open(file_name, 'r')
#     time = []
#     vx = []
#     vy = []
#     vz = []
#     x = []
#     y = []
#     for line in f.readlines():
#         line = line.split()
#         if line == []:
#             break
#         time.append(float(line[0]))
#         vx.append(float(line[1]))
#         vy.append(float(line[2]))
#         vz.append(float(line[3]))
#         x.append(float(line[4]))
#         y.append(float(line[5]))
#     f.close()
#     return time, vx, vy, vz, x, y


def read_bx(path, t):
    file = 'bx' + chr(t) + '.dat'
    file_name = path + file
    f = open(file_name, 'r')
    bx = []
    f.readline(2)
    for line in f.readlines():
        if not line:
            break
        bx.append(float(line[0]))

    f.close()
    return bx


def read_by(path, t):
    file = 'by' + chr(t) + '.dat'
    file_name = path + file
    f = open(file_name, 'r')
    by = []
    f.readline(2)
    for line in f.readlines():
        if not line:
            break
        by.append(float(line[0]))

    f.close()
    return by


def read_bz(path, t, Nx, Ny, Npx, Npy):
    file = 'bz' + str(t) + '.dat'
    file_name = path + file
    f = open(file_name)
    bz = []
    # bz = np.array(bz)
    Bz = []
    i = 0
    f.readline()
    f.readline()
    # do not read the newline characters '\n'
    for line in f.read().splitlines():
        if not line:
            break
        bz.append(float(line))
        i = i + 1
        # the data is written in y direction then is x direction
        # data in one process
        # if i % ((512 / 8) * (128/8)) == 0:   512 is x points, 128 is y points
        if i % ((Nx / Npx) * (Ny / Npy)) == 0:
            bz = np.array(bz)
            # bz = np.reshape(bz, (int(128 / 8), int(512 / 8)), order='F')
            bz = np.reshape(bz, (int(Ny / Npy), int(Nx / Npx)), order='F')
            i = 0
            Bz.append(bz)
            bz = []

    f.close()
    return Bz

def read_by(path, t, Nx, Ny, Npx, Npy):
    file = 'by' + str(t) + '.dat'
    file_name = path + file
    f = open(file_name)
    bz = []
    # bz = np.array(bz)
    Bz = []
    i = 0
    f.readline()
    f.readline()
    # do not read the newline characters '\n'
    for line in f.read().splitlines():
        if not line:
            break
        bz.append(float(line))
        i = i + 1
        # the data is written in y direction then is x direction
        # data in one process
        # if i % ((512 / 8) * (128/8)) == 0:   512 is x points, 128 is y points
        if i % ((Nx / Npx) * (Ny / Npy)) == 0:
            bz = np.array(bz)
            # bz = np.reshape(bz, (int(128 / 8), int(512 / 8)), order='F')
            bz = np.reshape(bz, (int(Ny / Npy), int(Nx / Npx)), order='F')
            i = 0
            Bz.append(bz)
            bz = []

    f.close()
    return Bz

def read_cn(path, t):
    file = 'cn' + chr(t) + '.dat'
    file_name = path + file
    f = open(file_name, 'r')
    cn = []
    f.readline(2)
    for line in f.readlines():
        if not line:
            break
        cn.append(float(line[0]))

    f.close()
    return cn


def read_phase(path, t):
    file = 'phase' + str(t) + '.dat'
    file_name = path + file
    f = open(file_name, 'r')
    j = []
    x = []
    y = []
    vx = []
    vy = []
    vz = []
    # for i in [0, 1, 3]:
    #     file = str(i) + 'phase' + str(t) + '.dat'
    #     file_name = path + file
    #     f = open(file_name, 'r')

    for line in f.readlines():
        line = line.split()
        if not line:
            break

        # if float(line[1]) < 10:
        j.append(float(line[0]))
        x.append(float(line[1]))
        y.append(float(line[2]))
        vx.append(float(line[3]))
        vy.append(float(line[4]))
        vz.append(float(line[5]))
    f.close()
    return j, vx, vy, vz, x, y


def read_Btimeseries(path):
    file = 'timeseries.dat'
    file_name = path + file

    f = open(file_name)
    By = []
    for i in range(10000):
        by = []
        bx = f.readline()
        if not bx:
            break
        for bxo in bx.split():
            by.append(float(bxo))

        By.append(by)

    f.close()
    return By


def read_B_energy(path):
    file = 'energy.dat'
    file_name = path + file
    f = open(file_name)
    t = []
    thermal_x = []
    thermal_y = []
    thermal_z = []
    thermal_total = []
    flow_x = []
    flow_y = []
    flow_z = []
    flow_total = []
    v_total = []

    third_thermal_x = []
    third_thermal_y = []
    third_thermal_z = []
    third_thermal_total = []
    third_flow_x = []
    third_flow_y = []
    third_flow_z = []
    third_flow_total = []
    third_v_total = []

    back_thermal_x = []
    back_thermal_y = []
    back_thermal_z = []
    back_thermal_total = []
    back_flow_x = []
    back_flow_y = []
    back_flow_z = []
    back_flow_total = []
    back_v_total = []

    particle_total = []
    ex = []
    ey = []
    ez = []
    e_total = []
    bx = []
    by = []
    bz = []
    b_total = []
    total = []

    for line in f.readlines():
        line = line.split()
        if not line:
            break
        if len(line) < 29:
            break

        t.append(float(line[0]))
        thermal_x.append(float(line[1]))
        thermal_y.append(float(line[2]))
        thermal_z.append(float(line[3]))
        thermal_total.append(float(line[4]))
        flow_x.append(float(line[5]))
        flow_y.append(float(line[6]))
        flow_z.append(float(line[7]))
        flow_total.append(float(line[8]))
        v_total.append(float(line[9]))

        back_thermal_x.append(float(line[10]))
        back_thermal_y.append(float(line[11]))
        back_thermal_z.append(float(line[12]))
        back_thermal_total.append(float(line[13]))
        back_flow_x.append(float(line[14]))
        back_flow_y.append(float(line[15]))
        back_flow_z.append(float(line[16]))
        back_flow_total.append(float(line[17]))
        back_v_total.append(float(line[18]))

        third_thermal_x.append(float(line[19]))
        third_thermal_y.append(float(line[20]))
        third_thermal_z.append(float(line[21]))
        third_thermal_total.append(float(line[22]))
        third_flow_x.append(float(line[23]))
        third_flow_y.append(float(line[24]))
        third_flow_z.append(float(line[25]))
        third_flow_total.append(float(line[26]))
        third_v_total.append(float(line[27]))

        particle_total.append(float(line[28]))
        ex.append(float(line[29]))
        ey.append(float(line[30]))
        ez.append(float(line[31]))
        e_total.append(float(line[32]))
        bx.append(float(line[33]))
        by.append(float(line[34]))
        bz.append(float(line[35]))
        b_total.append(float(line[36]))
        total.append(float(line[37]))

    f.close()
    return (t, particle_total,
            ex, ey, ez, e_total, bx, by, bz, b_total, total)


def read_energy(path):
    file = 'energy.dat'
    file_name = path + file
    f = open(file_name)
    t = []
    thermal_x = []
    thermal_y = []
    thermal_z = []
    thermal_total = []
    flow_x = []
    flow_y = []
    flow_z = []
    flow_total = []
    v_total = []

    back_thermal_x = []
    back_thermal_y = []
    back_thermal_z = []
    back_thermal_total = []
    back_flow_x = []
    back_flow_y = []
    back_flow_z = []
    back_flow_total = []
    back_v_total = []

    particle_total = []
    ex = []
    ey = []
    ez = []
    e_total = []
    bx = []
    by = []
    bz = []
    b_total = []
    total = []

    for line in f.readlines():
        line = line.split()
        if not line:
            break
        if len(line) < 29:
            break

        t.append(float(line[0]))
        thermal_x.append(float(line[1]))
        thermal_y.append(float(line[2]))
        thermal_z.append(float(line[3]))
        thermal_total.append(float(line[4]))
        flow_x.append(float(line[5]))
        flow_y.append(float(line[6]))
        flow_z.append(float(line[7]))
        flow_total.append(float(line[8]))
        v_total.append(float(line[9]))

        back_thermal_x.append(float(line[10]))
        back_thermal_y.append(float(line[11]))
        back_thermal_z.append(float(line[12]))
        back_thermal_total.append(float(line[13]))
        back_flow_x.append(float(line[14]))
        back_flow_y.append(float(line[15]))
        back_flow_z.append(float(line[16]))
        back_flow_total.append(float(line[17]))
        back_v_total.append(float(line[18]))

        particle_total.append(float(line[19]))
        ex.append(float(line[20]))
        ey.append(float(line[21]))
        ez.append(float(line[22]))
        e_total.append(float(line[23]))
        bx.append(float(line[24]))
        by.append(float(line[25]))
        bz.append(float(line[26]))
        b_total.append(float(line[27]))
        total.append(float(line[28]))

    f.close()
    return (t, thermal_x, thermal_y, thermal_z, thermal_total, flow_x, flow_y, flow_z, flow_total,
            back_thermal_x, back_thermal_y, back_thermal_z, back_thermal_total,
            back_flow_x, back_flow_y, back_flow_z, back_flow_total, back_v_total, particle_total,
            ex, ey, ez, e_total, bx, by, bz, b_total, total)


def read2d_timeseries(path, b):
    file = 'timeseries_' + b + '.dat'
    file_name = path + file

    f = open(file_name)
    By = []
    for line in f.readlines():
        line = line.split()
        if not line:
            break
        # if len(line) < 512:
        #     break
        # line = float(line)
        By.append(line)

    f.close()
    return By


def read_output(path):
    omegar = []
    omegai = []
    file = path

    f = open(file)

    for line in f.readlines():
        line = line.split()
        if not line:
            break
        omegar.append(float(line[0]))
        omegai.append(float(line[1]))

        f.close()
    return omegar, omegai
