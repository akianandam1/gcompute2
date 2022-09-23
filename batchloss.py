import torch
from RevisedNumericalSolver import torchstate



# Perturbs input vector using normal distribution
# takes in float standard deviation
# Requires floats
def perturb(vec, std):
    return torch.tensor([torch.normal(mean=vec[0], std=torch.tensor(std)),
                         torch.normal(mean=vec[1], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[3], std=torch.tensor(std)),
                         torch.normal(mean=vec[4], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[6], std=torch.tensor(std)),
                         torch.normal(mean=vec[7], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[9], std=torch.tensor(std)),
                         torch.normal(mean=vec[10], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[12], std=torch.tensor(std)),
                         torch.normal(mean=vec[13], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[15], std=torch.tensor(std)),
                         torch.normal(mean=vec[16], std=torch.tensor(std)),
                         0.0,], requires_grad = True)

# Compares two states and returns a numerical value rating how far apart the two states in the three
# body problem are to each other. Takes in two tensor states and returns tensor of value of distance rating
# The higher the score, the less similar they are
def nearest_position(particle, state1, state2):
    mse = torch.nn.L1Loss()
    if particle == 1:
        return mse(state1[:3], state2[:3]) + mse(state1[9:12], state2[9:12])
    elif particle == 2:
        return mse(state1[3:6], state2[3:6]) + mse(state1[12:15], state2[12:15])
    elif particle == 3:
        return mse(state1[6:9], state2[6:9]) + mse(state1[15:18], state2[15:18])
    else:
        print("bad input")


# Finds the most similar state to the initial position in a data set
def nearest_position_state(particle, state, data_set, min, max, time_step):
    i = min
    max_val = torch.tensor([100000000])
    index = -1
    while i < max:
        if nearest_position(particle, state, data_set[i]).item() < max_val.item():
            index = i
            max_val = nearest_position(particle, state, data_set[i])

        i += 1
    #print(f"Time: {index*time_step}")
    return index



def loss_values(identity, vec, m_1, m_2, m_3, lr, time_step, num_epochs, max_period, opt_func, file_name):

    optimizer = opt_func([vec], lr = lr)
    losses = []
    i = 0
    print("start")
    while i < num_epochs:
        print(i)
        input_vec = torch.cat((vec, torch.tensor([m_1, m_2, m_3])))
        if len(losses) > 10:
            if losses[-1] == losses[-3]:
                # print("Repeated")
                optimizer = torch.optim.SGD([vec], lr = .00001)

        if i > 10:
            if losses[-1] < .1:
                time_step = time_step/2
            if losses[-1] <.5:
                time_step = time_step/2
            if losses[-1] < .01:
                time_step = time_step / 2
        print(time_step)
        data_set = torchstate(input_vec, time_step, max_period, "rk4")


        optimizer.zero_grad()


        first_index = nearest_position_state(1, data_set[0], data_set, 300, len(data_set), time_step)
        first_particle_state = data_set[first_index]
        second_index = nearest_position_state(2, data_set[0], data_set, 300, len(data_set), time_step)
        second_particle_state = data_set[second_index]
        third_index = nearest_position_state(3, data_set[0], data_set, 300, len(data_set), time_step)
        third_particle_state = data_set[third_index]
        loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0],
                                                                                         second_particle_state) + nearest_position(
            3, data_set[0], third_particle_state)

        # print(" ")
        print(input_vec)
        print(vec.grad)
        print(f"{identity},{i},{loss.item()}\n")
        losses.append(loss.item())

        with open(file_name, "a") as file:
            file.write(f"{identity},{i},{loss.item()},{vec}\n")

        # print(loss)

        loss.backward()

        # Updates input vector
        optimizer.step()

        # print(f"Epoch:{i}")
        # print(" ")

        i += 1


def case1road(data, start, end):
    i = start
    while i <= end:
        print("begun")
        m_1 = float(data[i][0])
        m_2 = float(data[i][1])
        m_3 = float(data[i][2])
        x_1 = float(data[i][3])
        v_1 = float(data[i][4])
        v_2 = float(data[i][5])
        T = float(data[i][6])
        vec = torch.tensor([x_1,0,0,1,0,0,0,0,0, 0, v_1, 0, 0, v_2, 0, 0, -(m_1*v_1 + m_2*v_2)/m_3, 0], requires_grad = True)
        vec = perturb(vec, .01)
        loss_values(i, vec, m_1, m_2, m_3, .0001, .0008, 1000, int(T+2), torch.optim.NAdam, "case1road.txt")
        i += 1


def case2road(data, start, end):
    i = start
    while i <= end:
        print("begun")
        m_1 = float(data[i][0])
        m_2 = float(data[i][1])
        m_3 = float(data[i][2])
        x_1 = float(data[i][3])
        v_1 = float(data[i][4])
        v_2 = float(data[i][5])
        T = float(data[i][6])
        vec = torch.tensor([x_1,0,0,1,0,0,0,0,0, 0, v_1, 0, 0, v_2, 0, 0, -(m_1*v_1 + m_2*v_2)/m_3, 0], requires_grad = True)
        vec = perturb(vec, .01)
        loss_values(i, vec, m_1, m_2, m_3, .0001, .0008, 1000, int(T+2), torch.optim.NAdam, "case2road.txt")
        i += 1

def unequalcollisionsless(data, start, end):
    i = start
    while i <= end:
        print("begun")
        m_3 = data[i][0]
        v_1 = data[i][1]
        v_2 = data[i][2]
        T = data[i][3]
        vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, v_1, v_2, 0, v_1, v_2, 0, -2*v_1/m_3, -2*v_2/m_3, 0], requires_grad = True)
        vec = perturb(vec, .01)
        loss_values(i, vec, 1, 1, m_3, .0001, .0008, 1000, int(T+2), torch.optim.NAdam, "unequalcollisionless.txt")
        i += 1

# Must be very precise in compuattion (small learning step)
def equalfreefall(data, start, end):
    i = start
    while i <= end:
        print("begun")
        m_1 = data[i][0]
        m_2 = data[i][1]
        m_3 = data[i][2]
        x = data[i][3]
        y = data[i][4]
        T = data[i][5]
        vec = torch.tensor([-.5, 0, 0, .5, 0, 0, x, y, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], requires_grad = True)
        vec = perturb(vec, .01)
        loss_values(i, vec, m_1, m_2, m_3, .0001, .0008, 1000, int(T+2), torch.optim.NAdam, "equalfreefall.txt")
        i += 1


def equalmass(data, start, end):
    i = start
    while i <= end:
        print("begun")
        v_1 = data[i][0]
        v_2 = data[i][1]
        T = data[i][2]
        vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, v_1, v_2, 0, v_1, v_2, 0, -2*v_1, -2*v_2, 0], requires_grad = True)
        vec = perturb(vec, .01)
        loss_values(i, vec, 1, 1, 1, .0001, .0008, 1000, int(T+2), torch.optim.NAdam, "equalmass.txt")
        i += 1


def nhd(data, start, end):
    i = start
    while i <= end:
        print("begun")
        m_1 = float(data[i][0])
        m_2 = float(data[i][1])
        m_3 = float(data[i][2])
        x_1 = float(data[i][3])
        v_1 = float(data[i][4])
        v_2 = float(data[i][5])
        T = float(data[i][6])
        vec = torch.tensor([x_1, 0, 0, 1, 0, 0, 0, 0, 0, 0, v_1, 0, 0, v_2, 0, 0, -(m_1*v_1+m_2*v_2)/m_3, 0], requires_grad = True)
        vec = perturb(vec, .01)
        loss_values(i, vec, m_1, m_2, m_3, .0001, .0008, 1000, int(T+2), torch.optim.NAdam, "nhd.txt")
        i += 1


if __name__ == "__main__":
    from Datasets.nhd import values
    nhd(values, 0, 1)

