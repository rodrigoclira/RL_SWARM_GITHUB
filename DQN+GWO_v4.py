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

BATCH_SIZE = 256
CAPACITY = 100000 # capacity of Experience Replay

MAX_ITERATION = 500
MAX_STEPS = 200 # Max steps per 1 Iteration
GAMMA = 0.99 # 할인율
EPSILON = 0 # initial epsilon value
SUCCESS_STEPS = 195
SUCCESS_PHASE = 10

# SEARCHER---------------------------------------------------------------------------------------------
wolves_no = 32


# namedtuple을 이용한 트랜지션 정의
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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



# 트랜지션을 저장하기 위한 메모리 클래스
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # Memory의 최대 저장량
        self.memory = []
        self.index = 0  # Memory index

    # Queue 형식의 Experience replay memory 차용
    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # 트랜지션의 갯수가 최대 저장량을 초과하면 오래된 것부터 지우고 해당 인덱스에 새 트랜지션 부여
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # length 함수 오버로딩
    def __len__(self):
        return len(self.memory)



# DQN을 수행하는 agent의 두뇌를 담당하는 클래스
class Brain:
    def __init__(self, num_states, num_actions, epsilon, wolves_no, lb, ub):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        self.epsilon = epsilon # 초기 앱실론
        self.wolves_no = wolves_no # GWO wolf의 수
        self.lb = lb # PSO&GWO lower bound
        self.ub = ub # PSO&GWO upper bound
        self.toggle = False
        self.learning_toggle = False

        self.n_in, self.n_mid1, self.n_mid2, self.n_out = num_states, hid1_dim, hid2_dim, num_actions
        self.main_q_network = Net(self.n_in, self.n_mid1, self.n_mid2, self.n_out) # Main Q Network
        self.temp_q_network = copy.deepcopy(self.main_q_network) # Temp Q Network -> Fitness를 구할 때 사용
        self.target_q_network = copy.deepcopy(self.main_q_network) # Target Q Network

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
        self.optimizer = torch.optim.Adam(self.main_q_network.parameters(), lr=0.0001)

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
        # 저장된 트랜지션 수 확인, 만약 메모리가 배치사이즈보다 작으면 아무것도 x

        if episode == 0 and self.toggle is False:
            self.toggle = True
            for i in range(wolves_no):
                self.population[i, :] = np.random.uniform(self.lb, self.ub, self.dim + self.bias)
                w1, w2, w3, b1, b2, b3 = self.weight_array2tensor(self.population[i, :])
                fitness = self.get_fitness(w1, w2, w3, b1, b2, b3)

                if fitness > self.alpha_score:
                    self.delta_score = self.beta_score  # Update delta
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score  # Update beta
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness  # Update alpha
                    self.alpha_pos = self.population[i, :].copy()

                elif fitness > self.beta_score:
                    self.delta_score = self.beta_score  # Update delte
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness  # Update beta
                    self.beta_pos = self.population[i, :].copy()

                elif fitness > self.delta_score:
                    self.delta_score = fitness  # Update delta
                    self.delta_pos = self.population[i, :].copy()

            self.search(episode)
            return

        # GD phase
        if self.delta_score == 200 and len(self.memory) >= BATCH_SIZE:
            if self.learning_toggle is False:
                self.learning_toggle = True
                print("-------------------------Start learning-------------------------")
            self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
            self.learn(episode)

        # Metaheuristic phase
        else:
            self.search(episode)


    def decide_action(self, state, episode):
        # 최적 행동 결정
        self.main_q_network.eval()
        with torch.no_grad():
            # 신경망 출력의 최댓값에 대한 인덱스 = max(1)[1]
            action = self.main_q_network(state).max(1)[1].view(1, 1)

        return action

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

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
            action = self.temp_q_network(state).max(1)[1].view(1, 1)
            observation_next, reward, done, _ = self.env.step(action.item())

            if done:
                break
            else:
                fitness += reward
                state_next = observation_next
                state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                state_next = torch.unsqueeze(state_next, 0)
            state = state_next
        reward = torch.FloatTensor([float(reward)])
        self.memorize(state, action, state_next, reward)

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

            if fitness > self.alpha_score:
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

    def memorize(self, state, action, state_next, reward):
        self.memory.push(state, action, state_next, reward)

    def learn(self, episode):
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 메인 신경망으로 Q(s_t, a_t) 계산
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)

        # 다음 상태가 존재하는지 확인하는 마스크
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))

        # max{q(s_t+1, a)}를 계산하기 위해 먼저 전체를 0으로 초기화
        next_state_values = torch.zeros(BATCH_SIZE)
        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 메인 신경망으로 계산
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]

        # 다음 상태가 있는 것만 걸러내고, size32 를 32*1로 변환
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 타겟 신경망으로 계산
        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1,
                                                                                                     a_m_non_final_next_states).detach().squeeze()

        # 정답신호로 사용할 Q(s_t, a_t)값을 Q러닝 식으로 계산
        self.expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        self.main_q_network.train()
        loss = torch.nn.functional.smooth_l1_loss(self.state_action_values,
                                                  self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


# Agent class
class Agent:
    def __init__(self, num_states, num_actions, EPSILON, wolves_no, lb, ub):
        self.brain = Brain(num_states, num_actions, EPSILON, wolves_no, lb, ub)

    def update_q_function(self, episode):
        self.brain.replay(episode)

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()



class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions, EPSILON, wolves_no, -1, 1)
        self.success_list = []
        self.loss_list = []
        self.timer = 0
        self.result = 0

    def run(self):
        start = timeit.default_timer()
        episode_suc_list = np.zeros(SUCCESS_PHASE)
        complete_episodes = 0
        episode_final = False
        frames = []

        for episode in range(MAX_ITERATION):
            observation = self.env.reset()
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):
                # if episode_final is True:
                # frames.append(self.env.render(mode='rgb_array'))
                action = self.agent.get_action(state, episode)

                observation_next, reward, done, _ = self.env.step(action.item())  # reward 와 info를 사용하지 않기 때문에 _로 치환
                if done:
                    state_next = None

                    # 최근 10 epi에서 버틴 단계 수를 리스트에 저장
                    episode_suc_list = np.hstack((episode_suc_list[1:], step + 1))
                    if step < SUCCESS_STEPS:
                        reward = torch.FloatTensor([-1.0])
                        # 연속 성공 에피소드 기록 초기화
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        # 연속 성공 에피소드 기록 갱신
                        complete_episodes += 1

                else:
                    reward = torch.FloatTensor([float(reward)])  # 그외의 경우는 보상 0 부여
                    state_next = observation_next  # 관측 결과를 그대로 상태로 이용
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)  # numpy -> tensor
                    state_next = torch.unsqueeze(state_next, 0)

                # 메모리에 저장
                self.agent.memorize(state, action, state_next, reward)

                # Experince replay로 Q함수를 수정
                self.agent.update_q_function(episode)

                # 관측 결과 업데이트
                state = state_next

                # 에피소드 종료 처리
                if done:
                    print('%d Episode:\n Success steps = %d \n최근 10 에피소드의 평균 단계 수 = %.1lf' % (episode, step + 1, episode_suc_list.mean()))
                    print("epsilon value = " + str(self.agent.brain.epsilon))
                    print("score : " + str(self.agent.brain.alpha_score) + "/" + str(self.agent.brain.beta_score) + "/" + str(self.agent.brain.delta_score) + "\n")

                    if (episode % 10 == 0):
                        self.agent.update_target_q_function()

                    self.success_list.append(step+1)
                    break

            if episode_final is True:
                # 애니메이션 생성 및 저장
                # self.env.close()
                # display_frames_as_gif(frames)
                # print("저장 완료")
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

        '''
        self.success_list = np.array(self.success_list)

        fig, (ax1) = plt.subplots(1, figsize=(10, 10), sharex=True,)

        ax1.plot(self.success_list, label="Num of success")
        ax1.set_ylabel("Success steps")
        ax1.set_xlabel("Episodes")

        plt.title("GWODQN")
        plt.show()
        '''


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