import numpy as np
import gym
from collections import namedtuple
import torch
import warnings
import random
import copy
import matplotlib.pyplot as plt
import timeit
import datetime
warnings.filterwarnings('ignore')

# HYPER-PARAMETERS
# MODEL---------------------------------------------------------------------------------------------
ENV = 'CartPole-v0'
in_dim, hid1_dim, hid2_dim, out_dim = 4, 64, 64, 2 # 뉴럴 넷 구조 정의

MAX_ITERATION = 500
MAX_STEPS = 200 # Max steps per 1 Iteration
GAMMA = 0.99 # 할인율
EPSILON = 0 # initial epsilon value
SUCCESS_STEPS = 195
SUCCESS_PHASE = 10

# SEARCHER---------------------------------------------------------------------------------------------
wolves_no = 16


# Network 클래스
class Net(torch.nn.Module):
    def __init__(self, n_in, n_mid1, n_mid2, n_out):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(n_in, n_mid1)
        self.fc2 = torch.nn.Linear(n_mid1, n_mid2)
        self.fc3 = torch.nn.Linear(n_mid2, n_out)

    def forward(self, x):
        h1 = torch.nn.functional.relu(self.fc1(x))
        h2 = torch.nn.functional.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output


# DQN을 수행하는 agent의 두뇌를 담당하는 클래스
class Brain:
    def __init__(self, num_states, num_actions, epsilon, wolves_no, lb, ub):
        self.num_actions = num_actions
        self.epsilon = epsilon # 초기 앱실론
        self.epsilon_decay = 0.05
        self.epsilon_min = 0
        self.wolves_no = wolves_no # GWO wolf의 수
        self.lb = lb # PSO&GWO lower bound
        self.ub = ub # PSO&GWO upper bound

        self.n_in, self.n_mid1, self.n_mid2, self.n_out = num_states, hid1_dim, hid2_dim, num_actions
        self.main_q_network = Net(self.n_in, self.n_mid1, self.n_mid2, self.n_out) # Main Q Network
        self.temp_q_network = copy.deepcopy(self.main_q_network) # Temp Q Network -> Fitness를 구할 때 사용

        self.dim = self.n_in * self.n_mid1 + self.n_mid1 * self.n_mid2 + self.n_mid2 * self.n_out
        self.bias = self.n_mid1 + self.n_mid2 + self.n_out
        self.population = np.zeros((self.wolves_no, self.dim + self.bias))
        self.alpha_pos = np.zeros(self.dim + self.bias)
        self.alpha_score = 0
        self.beta_pos = np.zeros(self.dim + self.bias)
        self.beta_score = 0
        self.delta_pos = np.zeros(self.dim + self.bias)
        self.delta_score = 0

        self.env = gym.make(ENV)

    # weight, bias 순서대로 tensor -> numpy_array로 바꿔줌
    def weight_tensor2array(self, net):
        fc1 = net.fc1.weight.detach().numpy()
        fc1 = np.ravel(fc1, order='C')

        fc2 = net.fc2.weight.detach().numpy()
        fc2 = np.ravel(fc2, order='C')

        fc3 = net.fc3.weight.detach().numpy()
        fc3 = np.ravel(fc3, order='C')

        b1 = net.fc1.bias.detach().numpy()
        b2 = net.fc2.bias.detach().numpy()
        b3 = net.fc3.bias.detach().numpy()

        result = np.concatenate((fc1, fc2, fc3, b1, b2, b3), axis=0)
        return result

    # weight를 numpy_array -> tensor1, 2, 3으로 바꿔줌
    def weight_array2tensor(self, w):
        lengths = 0
        w1 = []
        w2 = []
        w3 = []
        b1 = []
        b2 = []
        b3 = []

        for i in range(self.n_in):
            w1.append(w[lengths: self.n_mid1 + lengths])
            lengths += self.n_mid1
        w1 = np.array(w1)

        for i in range(self.n_mid1):
            w2.append(w[lengths: self.n_mid2 + lengths])
            lengths += self.n_mid2
        w2 = np.array(w2)

        for i in range(self.n_mid2):
            w3.append(w[lengths: self.n_out + lengths])
            lengths += self.n_out
        w3 = np.array(w3)

        b1 = w[self.dim : self.dim+self.n_mid1]
        b2 = w[self.dim + self.n_mid1 : self.dim + self.n_mid1 + self.n_mid2]
        b3 = w[self.dim + self.n_mid1 + self.n_mid2 :]

        return w1, w2, w3, b1, b2, b3


    def replay(self, episode):
        if episode == 0 :
            for i in range(wolves_no):
                self.population[i, :] = np.random.uniform(self.lb, self.ub, self.dim + self.bias)
                w1, w2, w3, b1, b2, b3 = self.weight_array2tensor(self.population[i, :])
                fitness = self.get_fitness(w1, w2, w3, b1, b2, b3)

                if fitness >= self.alpha_score:
                    self.delta_score = self.beta_score  # Update delta
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score  # Update beta
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness  # Update alpha
                    self.alpha_pos = self.population[i, :].copy()

                if fitness < self.alpha_score and fitness >= self.beta_score:
                    self.delta_score = self.beta_score  # Update delte
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness  # Update beta
                    self.beta_pos = self.population[i, :].copy()

                if fitness < self.alpha_score and fitness < self.beta_score and fitness >= self.delta_score:
                    self.delta_score = fitness  # Update delta
                    self.delta_pos = self.population[i, :].copy()

            self.search(episode)

        # GD phase
        else:
            self.search(episode)


    def decide_action(self, state, episode):
        # 신경망을 추론 모드로 전환
        self.main_q_network.eval()
        with torch.no_grad():
            # 신경망 출력의 최댓값에 대한 인덱스 = max(1)[1]
            action = self.main_q_network(state).max(1)[1].view(1, 1)

        return action

    def get_fitness(self, w1, w2, w3, b1, b2, b3):
        # 신경망을 추론모드로 전환
        self.temp_q_network.fc1.weight = torch.nn.Parameter(torch.from_numpy(w1.T).float())
        self.temp_q_network.fc1.bias = torch.nn.Parameter(torch.from_numpy(b1.T).float())
        self.temp_q_network.fc2.weight = torch.nn.Parameter(torch.from_numpy(w2.T).float())
        self.temp_q_network.fc2.bias = torch.nn.Parameter(torch.from_numpy(b2.T).float())
        self.temp_q_network.fc3.weight = torch.nn.Parameter(torch.from_numpy(w3.T).float())
        self.temp_q_network.fc3.bias = torch.nn.Parameter(torch.from_numpy(b3.T).float())
        self.temp_q_network.eval()

        state = self.env.reset()
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0)

        fitness = 1

        for step in range(MAX_STEPS):
            if self.epsilon <= np.random.uniform(0, 1):
                with torch.no_grad():
                    action = self.temp_q_network(state).max(1)[1].view(1, 1)
            else :
                action = torch.LongTensor([[random.randrange(self.num_actions)]])

            observation_next, reward, done, _ = self.env.step(action.item())

            if done:
                break
            else:
                fitness += reward
                state_next = observation_next
                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                state_next = torch.unsqueeze(state_next, 0)
            state = state_next

        return fitness


    def search(self, episode):
        current_weight = self.weight_tensor2array(self.main_q_network)
        self.population[1,:] = current_weight
        a = 2 - episode * ((2) / MAX_ITERATION)
        for i in range(wolves_no):
            r1 = np.random.rand(self.dim + self.bias)  # r1 is a random number in [0,1]
            r2 = np.random.rand(self.dim + self.bias)  # r2 is a random number in [0,1]
            A1 = 2 * a * r1 - a  # Equation (3.3)
            C1 = 2 * r2  # Equation (3.4)
            self.D_alpha = abs(C1 * self.alpha_pos - self.population[i, :])  # Equation (3.5)-part 1
            X1 = self.alpha_pos - A1 * self.D_alpha  # Equation (3.6)-part 1

            r1 = np.random.rand(self.dim + self.bias)
            r2 = np.random.rand(self.dim + self.bias)
            A2 = 2 * a * r1 - a  # Equation (3.3)
            C2 = 2 * r2  # Equation (3.4)
            self.D_beta = abs(C2 * self.beta_pos - self.population[i, :])  # Equation (3.5)-part 2
            X2 = self.beta_pos - A2 * self.D_beta  # Equation (3.6)-part 2

            r1 = np.random.rand(self.dim + self.bias)
            r2 = np.random.rand(self.dim + self.bias)
            A3 = 2 * a * r1 - a  # Equation (3.3)
            C3 = 2 * r2  # Equation (3.4)
            self.D_delta = abs(C3 * self.delta_pos - self.population[i, :])  # Equation (3.5)-part 3
            X3 = self.delta_pos - A3 * self.D_delta  # Equation (3.5)-part 3

            self.population[i, :] = ((X1 + X2 + X3) / 3)  # Equation (3.7)
            w1, w2, w3, b1, b2, b3 = self.weight_array2tensor(self.population[i, :])
            fitness = self.get_fitness(w1, w2, w3, b1, b2, b3)

            if fitness > self.alpha_score :
                self.delta_score = self.beta_score  # Update delta
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = self.alpha_score  # Update beta
                self.beta_pos = self.alpha_pos.copy()
                self.alpha_score = fitness  # Update alpha
                self.alpha_pos = self.population[i, :].copy()
                print("a", end='')

            elif fitness > self.beta_score:
                self.delta_score = self.beta_score  # Update delte
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = fitness  # Update beta
                self.beta_pos = self.population[i, :].copy()
                print("b", end='')

            elif fitness > self.delta_score:
                self.delta_score = fitness  # Update delta
                self.delta_pos = self.population[i, :].copy()
                print("c", end='')

        nw1, nw2, nw3, nb1, nb2, nb3 = self.weight_array2tensor(self.alpha_pos)
        self.main_q_network.fc1.weight = torch.nn.Parameter(torch.from_numpy(nw1.T).float())
        self.main_q_network.fc1.bias = torch.nn.Parameter(torch.from_numpy(nb1.T).float())
        self.main_q_network.fc2.weight = torch.nn.Parameter(torch.from_numpy(nw2.T).float())
        self.main_q_network.fc2.bias = torch.nn.Parameter(torch.from_numpy(nb2.T).float())
        self.main_q_network.fc3.weight = torch.nn.Parameter(torch.from_numpy(nw3.T).float())
        self.main_q_network.fc3.bias = torch.nn.Parameter(torch.from_numpy(nb3.T).float())



# Agent class
class Agent:
    def __init__(self, num_states, num_actions, EPSILON, wolves_no, lb, ub):
        self.brain = Brain(num_states, num_actions, EPSILON, wolves_no, lb, ub)

    def update_q_function(self, episode):
        self.brain.replay(episode)

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action



class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions, EPSILON, wolves_no, -1, 1)
        self.success_list = []
        self.loss_list = []
        self.toggle = False
        self.timer = 0
        self.result = 0

    def run(self):
        start = timeit.default_timer()
        episode_suc_list = np.zeros(SUCCESS_PHASE)
        complete_episodes = 0
        episode_final = False

        for episode in range(MAX_ITERATION):
            observation = self.env.reset()
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):
                action = self.agent.get_action(state, episode)

                observation_next, _, done, _ = self.env.step(action.item())  # reward 와 info를 사용하지 않기 때문에 _로 치환
                if done:
                    state_next = None

                    # 최근 10 epi에서 버틴 단계 수를 리스트에 저장
                    episode_suc_list = np.hstack((episode_suc_list[1:], step + 1))
                    if step < SUCCESS_STEPS:
                        complete_episodes = 0
                    else:
                        complete_episodes += 1

                else:
                    state_next = observation_next  # 관측 결과를 그대로 상태로 이용
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy -> tensor
                    state_next = torch.unsqueeze(state_next, 0)

                # Experince replay로 Q함수를 수정
                if self.agent.brain.delta_score == 200 and self.toggle == False :
                    self.toggle = True
                    print("I'm GOD")
                elif self.agent.brain.delta_score == 200 and self.toggle == True:
                    pass
                else :
                    self.agent.update_q_function(episode)

                # 관측 결과 업데이트
                state = state_next

                # 에피소드 종료 처리
                if done:
                    print('%d Episode:\n Success steps = %d \n최근 10 에피소드의 평균 단계 수 = %.1lf' % (episode, step + 1, episode_suc_list.mean()))
                    print("epsilon value = " + str(self.agent.brain.epsilon))
                    print("score : " + str(self.agent.brain.alpha_score) + "/" + str(self.agent.brain.beta_score) + "/" + str(self.agent.brain.delta_score) + "\n")
                    self.success_list.append(step+1)
                    break

            if episode_final is True:
                self.env.close()
                break

            # 10 에피소드 연속으로 195단계를 버티면 성공
            if complete_episodes >= SUCCESS_PHASE:
                stop = timeit.default_timer()
                print('Computational time:'+ str(stop - start))
                print(str(SUCCESS_PHASE) + " 에피소드 연속 성공, 마지막 에피소드:" + str(episode))
                self.timer = float(stop - start)
                self.result = episode
                episode_final = True

        if self.timer == 0 and self.result == 0 :
            stop = timeit.default_timer()
            self.timer = float(stop - start)
            self.result = 500
            self.env.close()


if __name__ == "__main__" :
    time_list = []
    result_list = []
    for i in range(20):
        print("Experiment", i)
        cartpole_env = Environment()
        cartpole_env.run()
        result_list.append(int(cartpole_env.result))
        time_list.append(float(cartpole_env.timer))

    time_list = np.array(time_list)
    result_list = np.array(result_list)
    now = datetime.datetime.now()
    print(now)
    print("20회 실험 결과")
    print("Computational_time list")
    print(time_list)
    print()
    print("Computational_time.mean:", time_list.mean(), "Computational_time.std:", time_list.std())
    print()
    print()
    print("Final_episode list")
    print(result_list)
    print()
    print("Final_episode.mean:", result_list.mean(), "Final_episode.std:", result_list.std())
