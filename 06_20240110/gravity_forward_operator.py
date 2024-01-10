import numpy as np

    
        

def get_gravity_fwd_operator(x_measure=None):
    # number of model cells in x/z directions
    nr_x = 9
    nr_z = 3
    
    # measurement locations
    if x_measure is None:
        x_measure = np.arange(5, 95, 10)

    # model positions
    d = 10
    h = 10
    cell_width = d
    cell_height = h

    x = np.arange(0 + (d / 2), (d + 0.5) * nr_x, d)
    print(x)
    z = np.abs(
        np.arange(0 - (h / 2), -(h + 0.5) * nr_z, -h)
    )
    print(z)

    X, Z = np.meshgrid(x, z)
    print(X)
    print(Z)

    # the forward operator has dimensions: [NR_MEASUREMENTS, NR_MODEL_CELLS]
    A = np.zeros((x_measure.size, nr_x * nr_z))
    print(A.shape)

    gamma = 6.67408 * 10e-11

    for i in range(x_measure.size):
        for j in range(nr_x * nr_z):
            # print(i, j)
            zj = Z.flatten()[j]
            xj = X.flatten()[j]
            xi = x_measure[i]

            # print(i, j)
            r1 = np.sqrt((zj - h/2) ** 2 + (xi - xj + d/2) ** 2)
            r2 = np.sqrt((zj + h/2) ** 2 + (xi - xj + d/2) ** 2)
            r3 = np.sqrt((zj - h/2) ** 2 + (xi - xj - d/2) ** 2)
            r4 = np.sqrt((zj + h/2) ** 2 + (xi - xj - d/2) ** 2)

            theta1 = np.arctan2(xi - xj + d/2,  (zj - h/2))
            theta2 = np.arctan2(xi - xj + d/2,  (zj + h/2))
            theta3 = np.arctan2(xi - xj - d/2, (zj - h/2))
            theta4 = np.arctan2(xi - xj - d/2, (zj + h/2))

            A[i, j] = 2 * gamma * (
        (xi - xj + d/2) * np.log(
            r2 * r3 / (r1 * r4)
        ) + 
        d * np.log(r4 / r3) -
        (zj + h/2) * (theta4 - theta2) +
        (zj - h/2) * (theta3 - theta1)
            )
            # break
        # break
    # convert forward operator to [mGal]
    G = A * 1e5
    
    # define a simple return dictionary
    grav = {
        'measurement_positions': x_measure,
        'model_nr_x': nr_x,
        'model_nr_z': nr_z,
        'fwd_operator_mgal': G,
    }
    
    return grav



def get_measurement_data(x_measure=None):
    # measurement locations
    if x_measure is None:
        x_measure = np.arange(5, 95, 10)

    model_true = np.zeros((3, 9))
    model_true[0, 2:3] = 1000
    model_true[1, 5] = 1000
    
    grav = get_gravity_fwd_operator(x_measure)
    A = grav['fwd_operator_mgal']
    
    d_true = A @ model_true.flatten()
    
    # add noise
    # ALWAYS initialize the random number generator!
    np.random.seed(2300)
    
    d_with_noise = d_true + np.random.normal(
        loc=0,
        scale=0.05 * d_true,
        size=d_true.size
    )
    
    return model_true, d_true, d_with_noise