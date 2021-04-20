import numpy as np
import pyopencl as cl
import time

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
queue_speed = cl.CommandQueue(ctx)

mf = cl.mem_flags

max_time = 5
dt = 1
amount_elements = 3
N = int(max_time / dt)
weights = np.array([1.989e30, 0.3281e24, 4.811e24, 5.9761e24, 0.6331e24, 1876.6431e24, 561.801e24, 86.0541e24, 101.5921e24, 0.01191e24]) 
speeds = np.array([0.0, 58.93, 35.18, 29.66, 23.6, 13.03, 10.16, 6.52, 5.43, 5.97]) * 3600 * 1000  #convert to metrs per hours
positions =  np.array([[0.0, 0.0], [0.31, 0.1], [0.72, 0.11], [1.0, 0.12], [1.56, 0.13], [5.21, 0.14], [9.05, 0.15], [20.0, 0.16], [30.09, 0.17], [30.5, 0.18]]) * 1.496e11

x_pos = np.zeros((N, amount_elements)).astype(np.double)
y_pos = np.zeros((N, amount_elements)).astype(np.double)
speed_x = np.zeros((N, amount_elements)).astype(np.double)
speed_y = np.zeros((N, amount_elements)).astype(np.double)

init_speed = np.random.random(amount_elements)
tmp_x_init, tmp_y_init = np.random.random(amount_elements), np.random.random(amount_elements)

x_pos[0], y_pos[0] = tmp_x_init, tmp_y_init
speed_x[0], speed_y[0] =  np.zeros(amount_elements), init_speed

prg = cl.Program(ctx, """
void calc_acceleration(__global int* amount_elements, __global const double *x_pos, __global const double *y_pos, 
                        __global const double *weights, int i, double *acceleration_x_ng,  double *acceleration_y_ng)
{
    double gravi_const = 6.6742 * 1e-11 * (3600 * 3600);
    for (int j = 0; j < amount_elements[0]; j++)
    {
        if (j == i)
            continue;
        if (x_pos[j] != x_pos[i])
            acceleration_x_ng[0] +=  gravi_const * weights[j] * (x_pos[j] - x_pos[i])/pow(sqrt(pow(x_pos[j] - x_pos[i], 2) + pow(y_pos[j] - y_pos[i], 2)), 3);
        if (y_pos[j] != y_pos[i])
            acceleration_y_ng[0] += gravi_const * weights[j] * (y_pos[j] - y_pos[i])/pow(sqrt(pow(x_pos[j] - x_pos[i], 2) + pow(y_pos[j] - y_pos[i], 2)), 3);
    }
}

__kernel void calc_pos(
    __global int* dt, __global int* amount_elements, __global const double *x_pos_ng, __global const double *y_pos_ng,
    __global const double *x_speed_ng, __global const double *y_speed_ng,
    __global  double *acceleration_x_ng, __global double *acceleration_y_ng, __global double * weights_ng,
    __global double *res_g_x_pos, __global double *res_g_y_pos)
{
    int i = get_global_id(0);
    double tmp_acc_x[1];
    double tmp_acc_y[1];
    calc_acceleration(amount_elements, x_pos_ng, y_pos_ng, weights_ng, i, tmp_acc_x, tmp_acc_y);
    
    acceleration_x_ng[i] = tmp_acc_x[0];
    acceleration_y_ng[i] = tmp_acc_y[0];
    res_g_x_pos[i] = x_pos_ng[i] + x_speed_ng[i] * dt[0] +  0.5 * acceleration_x_ng[i] * (dt[0] * dt[0]);
    res_g_y_pos[i] = y_pos_ng[i] + y_speed_ng[i] * dt[0] + 0.5 * acceleration_y_ng[i] * (dt[0] * dt[0]);
}

__kernel void calc_speed(
    __global int* dt, __global int* amount_elements, __global const double *x_pos_ng, __global const double *y_pos_ng,
    __global const double *x_speed_ng, __global const double *y_speed_ng, 
    __global  double *acceleration_x_ng, __global  double *acceleration_y_ng, __global double * weights_ng,
    __global double *res_g_x_speed, __global double *res_g_y_speed)
{
    int i = get_global_id(0);
    double tmp_acc_x[1];
    double tmp_acc_y[1];
    calc_acceleration(amount_elements, x_pos_ng, y_pos_ng, weights_ng, i, tmp_acc_x, tmp_acc_y);

    res_g_x_speed[i] = x_speed_ng[i]  + 0.5 * (tmp_acc_x[0] + acceleration_x_ng[i]) * dt[0];
    res_g_y_speed[i] = y_speed_ng[i]  + 0.5 * (tmp_acc_y[0] + acceleration_y_ng[i]) * dt[0];
}
""").build()

res_g_x_pos = cl.Buffer(ctx, mf.WRITE_ONLY, x_pos[0].nbytes)
res_g_y_pos = cl.Buffer(ctx, mf.WRITE_ONLY, y_pos[0].nbytes)
res_g_x_speed = cl.Buffer(ctx, mf.WRITE_ONLY, speed_x[0].nbytes)
res_g_y_speed = cl.Buffer(ctx, mf.WRITE_ONLY, speed_y[0].nbytes)

acceleration_x_ng = cl.Buffer(ctx, mf.READ_WRITE, y_pos[0].nbytes)
acceleration_y_ng = cl.Buffer(ctx, mf.READ_WRITE, y_pos[0].nbytes)

knl_calc_pos = prg.calc_pos 
knl_calc_speed = prg.calc_speed
res_np_x_pos, res_np_y_pos = np.empty_like(x_pos), np.empty_like(y_pos)

dt_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array([dt]))
weights_ng = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)
amount_elements_ng = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array([amount_elements]))

for n in range(N - 1):
   
    x_pos_ng = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_pos[n])
    y_pos_ng = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_pos[n])
    x_speed_ng = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=speed_x[n])
    y_speed_ng = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=speed_y[n])

    knl_calc_pos(queue, x_pos[0].shape, None, dt_g, amount_elements_ng, x_pos_ng, y_pos_ng, x_speed_ng, y_speed_ng, 
                 acceleration_x_ng, acceleration_y_ng, weights_ng, res_g_x_pos, res_g_y_pos)

    cl.enqueue_copy(queue, x_pos[n + 1], res_g_x_pos)
    cl.enqueue_copy(queue, y_pos[n + 1], res_g_y_pos)
    
    
    x_pos_ng = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_pos[n + 1])
    y_pos_ng = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_pos[n + 1])

    knl_calc_speed(queue_speed, speed_x[0].shape, None, dt_g, amount_elements_ng, x_pos_ng, y_pos_ng, x_speed_ng, y_speed_ng,  
                    acceleration_x_ng, acceleration_y_ng, weights_ng, res_g_x_speed, res_g_y_speed)
    cl.enqueue_copy(queue_speed, speed_x[n + 1], res_g_x_speed)
    cl.enqueue_copy(queue_speed, speed_y[n + 1], res_g_y_speed)

tmp_x_pos, tmp_y_pos = x_pos, y_pos
tmp_spees_x, tmp_spees_y = speed_x, speed_y

x_pos = np.zeros((N, amount_elements)).astype(np.double)
y_pos = np.zeros((N, amount_elements)).astype(np.double)
speed_x = np.zeros((N, amount_elements)).astype(np.double)
speed_y = np.zeros((N, amount_elements)).astype(np.double)

x_pos[0], y_pos[0] = tmp_x_init, tmp_y_init
speed_x[0], speed_y[0] =  np.zeros(amount_elements), init_speed

def _calc_accelerations(x_pos, y_pos, i):
    v_equation_x, v_equation_y = 0, 0
    gravi_const = 6.6742 * 1e-11 * (3600 ** 2)
    for j in range(amount_elements):
        if j == i:
            continue
        if x_pos[j] != x_pos[i]:
            v_equation_x +=  gravi_const * weights[j] * (x_pos[j] - x_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
        if y_pos[j] != y_pos[i]:
            v_equation_y +=  gravi_const * weights[j] * (y_pos[j] - y_pos[i])/np.sqrt((x_pos[j] - x_pos[i]) ** 2 + (y_pos[j] - y_pos[i]) ** 2) ** 3
    
    return v_equation_x, v_equation_y
for n in range(0, N - 1):
    acceleration_x, acceleration_y = np.zeros(amount_elements), np.zeros(amount_elements)
    for i in range(amount_elements):
        acceleration_x[i], acceleration_y[i] = _calc_accelerations(x_pos[n], y_pos[n], i)
        x_pos[n + 1, i] = x_pos[n, i] + speed_x[n, i] * dt + acceleration_x[i] * (dt ** 2) / 2
        y_pos[n + 1, i] = y_pos[n, i] + speed_y[n, i] * dt + acceleration_y[i] * (dt ** 2) / 2

    for i in range(amount_elements):
        acceleration_x_next, acceleration_y_next = _calc_accelerations(x_pos[n + 1], y_pos[n + 1], i)
        speed_x[n + 1, i] = speed_x[n, i]  + 0.5 * (acceleration_x_next + acceleration_x[i]) * dt
        speed_y[n + 1, i] = speed_y[n, i]  + 0.5 * (acceleration_y_next + acceleration_y[i]) * dt

# print(tmp_x_pos)
# print()
# print(x_pos)
# print()
print(tmp_x_pos - x_pos)
print(tmp_y_pos - y_pos)
print(tmp_spees_y - speed_y)
print(tmp_spees_x - speed_x)