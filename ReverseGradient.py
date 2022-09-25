import random
from batchloss import nearest_position, nearest_position_state
import torch
from RevisedNumericalSolver import torchstate
print("begun")
i = 0
while True:
    with open("ReverseGradientPoints.txt") as file:
        lines = file.readlines()
    time_step = .001
    max_period = 10
    random.shuffle(lines)
    v_1 = torch.tensor([float(lines[0].split(",")[0])], requires_grad = True)
    v_2 = torch.tensor([float(lines[0].split(",")[1])], requires_grad = True)
    input_vec = torch.stack((
            torch.tensor([-1]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            v_1,
            v_2,
            torch.tensor([0]),
            v_1,
            v_2,
            torch.tensor([0]),
            -2*v_1,
            -2*v_2,
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1]),
            torch.tensor([1]),

        )).flatten()
    data_set = torchstate(input_vec, time_step, max_period, "rk4")
    first_index = nearest_position_state(1, data_set[0], data_set, 300, len(data_set), time_step)
    first_particle_state = data_set[first_index]
    second_index = nearest_position_state(2, data_set[0], data_set, 300, len(data_set), time_step)
    second_particle_state = data_set[second_index]
    third_index = nearest_position_state(3, data_set[0], data_set, 300, len(data_set), time_step)
    third_particle_state = data_set[third_index]
    loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0], second_particle_state) + nearest_position(3, data_set[0], third_particle_state)
    loss.backward()
    print(v_1, v_2)
    print(v_1.grad, v_2.grad)
    with torch.no_grad():

        v_1 += v_1.grad * .01
        v_2 += v_2.grad * .01

    v_1.grad.zero_()
    v_2.grad.zero_()
    print(f"{v_1.item()},{v_2.item()}")
    if "nan" in f"{v_1.item()},{v_2.item()}\n":
        print("passed")
        pass
    else:
        with open("ReverseGradientPoints2.txt", "a") as file:
            file.write(f"{v_1.item()},{v_2.item()}\n")
        with open("uniquepoints.txt", "a") as file:
            file.write(f"{v_1.item()},{v_2.item()}\n")
            print("written")
    
    print(f"Epoch: {i}")
    i += 1
