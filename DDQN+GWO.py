import numpy as np
import gym
from collections import namedtuple
import torch
import warnings
import random
import copy
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# HYPER-PARAMETERS
# MODEL---------------------------------------------------------------------------------------------
ENV = 'CartPole-v0'
in_dim, hid1_dim, hid2_dim, out_dim = 4, 64, 64, 2 # 뉴럴 넷 구조 정의

BATCH_SIZE = 64
CAPACITY = 10000 # capacity of Experience Replay

MAX_ITERATION = 200
MAX_STEPS = 200 # Max steps per 1 Iteration
GAMMA = 0.99 # 할인율
EPSILON = 1 # initial epsilon value
SUCCESS_STEPS = 195
SUCCESS_PHASE = 10

# SEARCHER---------------------------------------------------------------------------------------------
lb, ub, dim = -10, 10, in_dim * hid1_dim + hid1_dim * hid2_dim + hid2_dim * out_dim
# PHASE = 0.1
# PSO_ITERATION = int(MAX_ITERATION * PHASE)
# GWO_ITERATION = MAX_ITERATION - PSO_ITERATION
swarm_no = 100
wolves_no = 100

inertia_w = 0.3 # inertia constant
c1 = 1 # cognitive constant
c2 = 1 # social constant

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
    def __init__(self, num_states, num_actions, epsilon, swarm_no, wolves_no, w, c1, c2, lb, ub):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        self.epsilon = epsilon # 초기 앱실론
        self.swarm_no = swarm_no # PSO particle의 수
        self.wolves_no = wolves_no # GWO wolf의 수
        self.inertia_weight = w # PSO Intertia weight
        self.c1 = c1 # PSO c1
        self.c2 = c2 # PSO c2
        self.lb = lb # PSO&GWO lower bound
        self.ub = ub # PSO&GWO upper bound
        self.toggle = False # 메모리가 샘플링이 가능해질 정도로 충분해지면 토글이 해제되고 본격적인 서칭이 시작됨

        self.n_in, self.n_mid1, self.n_mid2, self.n_out = num_states, hid1_dim, hid2_dim, num_actions
        self.main_q_network = Net(self.n_in, self.n_mid1, self.n_mid2, self.n_out) # Main Q Network
        self.temp_q_network = Net(self.n_in, self.n_mid1, self.n_mid2, self.n_out) # Temp Q Network -> Fitness를 구할 때 사용
        self.target_q_network = copy.deepcopy(self.main_q_network) # Target Q Network

        self.dim = self.n_in * self.n_mid1 + self.n_mid1 * self.n_mid2 + self.n_mid2 * self.n_out
        self.population = np.zeros((self.wolves_no, self.dim))
        self.velocity = np.zeros((self.wolves_no, self.dim))
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float("inf")

        self.beta_pos = np.zeros(dim)
        self.beta_score = float("inf")

        self.delta_pos = np.zeros(dim)
        self.delta_score = float("inf")

    # weight를 tensor -> numpy_array로 바꿔줌
    def weight_tensor2array(self, net):
        fc1 = net.fc1.weight.detach().numpy()
        fc1 = np.ravel(fc1, order='C')

        fc2 = net.fc2.weight.detach().numpy()
        fc2 = np.ravel(fc2, order='C')

        fc3 = net.fc3.weight.detach().numpy()
        fc3 = np.ravel(fc3, order='C')

        result = np.concatenate((fc1, fc2, fc3), axis=0)
        return result

    # weight를 numpy_array -> tensor1, 2, 3으로 바꿔줌
    def weight_array2tensor(self, w):
        lengths = 0
        w1 = []
        w2 = []
        w3 = []

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

        return w1, w2, w3


    def replay(self, episode):
        # 저장된 트랜지션 수 확인, 만약 메모리가 배치사이즈보다 작으면 아무것도 x
        if len(self.memory) <= BATCH_SIZE:
            return

        if len(self.memory) > BATCH_SIZE and self.toggle == False:
            self.toggle = True

            self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

            for i in range(wolves_no):
                self.population[i, :] = np.random.uniform(lb, ub, dim)
                w1, w2, w3 = self.weight_array2tensor(self.population[i, :])
                fitness = self.get_fitness(w1, w2, w3)

                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score  # Update delta
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score  # Update beta
                    self.beta_pos = self.alpha_pos.copy()
                    self.alpha_score = fitness  # Update alpha
                    self.alpha_pos = self.population[i, :].copy()

                if fitness > self.alpha_score and fitness < self.beta_score:
                    self.delta_score = self.beta_score  # Update delte
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = fitness  # Update beta
                    self.beta_pos = self.population[i, :].copy()

                if fitness > self.alpha_score and fitness > self.beta_score and fitness < self.delta_score:
                    self.delta_score = fitness  # Update delta
                    self.delta_pos = self.population[i, :].copy()

            self.search(episode)

            return

        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        self.search(episode)

    def decide_action(self, state, episode):
        # e-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다.
        # self.epsilon = (episode * 0.99) / episode
        self.epsilon = 1 * (1 / (episode + 1))

        # 최적 행동 결정
        if self.epsilon <= np.random.uniform(0, 1):
            # 신경망을 추론 모드로 전환
            self.main_q_network.eval()
            with torch.no_grad():
                # 신경망 출력의 최댓값에 대한 인덱스 = max(1)[1]
                action = self.main_q_network(state).max(1)[1].view(1, 1)

        # 무작위 행동 결정
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_fitness(self, w1, w2, w3):
        # 신경망을 추론모드로 전환
        self.temp_q_network.fc1.weight = torch.nn.Parameter(torch.from_numpy(w1.T).float())
        self.temp_q_network.fc2.weight = torch.nn.Parameter(torch.from_numpy(w2.T).float())
        self.temp_q_network.fc3.weight = torch.nn.Parameter(torch.from_numpy(w3.T).float())
        self.temp_q_network.eval()
        self.target_q_network.eval()

        # print(self.temp_q_network.fc1.weight)

        # 메인 신경망으로 Q(s_t, a_t) 계산
        q_stat = self.temp_q_network(self.state_batch).gather(1, self.action_batch)

        # 다음 상태가 존재하는지 확인하는 마스크
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))

        # max{q(s_t+1, a)}를 계산하기 위해 먼저 전체를 0으로 초기화
        next_q_value = torch.zeros(BATCH_SIZE)
        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 메인 신경망으로 계산
        a_m[non_final_mask] = self.temp_q_network(self.non_final_next_states).detach().max(1)[1]

        # size32 를 32*1로 변환
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 타겟 신경망으로 계산
        next_q_value[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 정답신호로 사용할 Q(s_t, a_t)값을 Q러닝 식으로 계산
        expected_action_values = self.reward_batch + GAMMA * next_q_value

        loss = torch.nn.functional.smooth_l1_loss(q_stat, expected_action_values.unsqueeze(1))

        return loss


    def search(self, episode):
        current_weight = self.weight_tensor2array(self.main_q_network)
        self.population[1,:] = current_weight
        a = 2 - episode * ((2) / MAX_ITERATION)
        for i in range(wolves_no):
            r1 = np.random.rand(dim)  # r1 is a random number in [0,1]
            r2 = np.random.rand(dim)  # r2 is a random number in [0,1]
            A1 = 2 * a * r1 - a  # Equation (3.3)
            C1 = 2 * r2  # Equation (3.4)
            self.D_alpha = abs(C1 * self.alpha_pos - self.population[i, :])  # Equation (3.5)-part 1
            X1 = self.alpha_pos - A1 * self.D_alpha  # Equation (3.6)-part 1

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A2 = 2 * a * r1 - a  # Equation (3.3)
            C2 = 2 * r2  # Equation (3.4)
            self.D_beta = abs(C2 * self.beta_pos - self.population[i, :])  # Equation (3.5)-part 2
            X2 = self.beta_pos - A2 * self.D_beta  # Equation (3.6)-part 2

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A3 = 2 * a * r1 - a  # Equation (3.3)
            C3 = 2 * r2  # Equation (3.4)
            self.D_delta = abs(C3 * self.delta_pos - self.population[i, :])  # Equation (3.5)-part 3
            X3 = self.delta_pos - A3 * self.D_delta  # Equation (3.5)-part 3

            self.population[i, :] = ((X1 + X2 + X3) / 3)  # Equation (3.7)
            w1, w2, w3 = self.weight_array2tensor(self.population[i, :])
            fitness = self.get_fitness(w1, w2, w3)

            if fitness < self.alpha_score:
                self.delta_score = self.beta_score  # Update delta
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = self.alpha_score  # Update beta
                self.beta_pos = self.alpha_pos.copy()
                self.alpha_score = fitness  # Update alpha
                self.alpha_pos = self.population[i, :].copy()

            if fitness > self.alpha_score and fitness < self.beta_score:
                self.delta_score = self.beta_score  # Update delte
                self.delta_pos = self.beta_pos.copy()
                self.beta_score = fitness  # Update beta
                self.beta_pos = self.population[i, :].copy()

            if fitness > self.alpha_score and fitness > self.beta_score and fitness < self.delta_score:
                self.delta_score = fitness  # Update delta
                self.delta_pos = self.population[i, :].copy()

        nw1, nw2, nw3 = self.weight_array2tensor(self.alpha_pos)
        self.main_q_network.fc1.weight = torch.nn.Parameter(torch.from_numpy(nw1.T).float())
        self.main_q_network.fc2.weight = torch.nn.Parameter(torch.from_numpy(nw2.T).float())
        self.main_q_network.fc3.weight = torch.nn.Parameter(torch.from_numpy(nw3.T).float())

        # print(self.swarm_best_pos)
        # print(self.swarm_best_sco.item())


    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


# Agent class
class Agent:
    def __init__(self, num_states, num_actions, EPSILON, swarm_no, wolves_no, w, c1, c2, lb, ub):
        self.brain = Brain(num_states, num_actions, EPSILON, swarm_no, wolves_no, w, c1, c2, lb, ub)

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
        self.agent = Agent(num_states, num_actions, EPSILON, swarm_no, wolves_no, inertia_w, c1, c2, lb, ub)
        self.success_list = []
        self.loss_list = []

    def run(self):
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

                observation_next, _, done, _ = self.env.step(action.item())  # reward 와 info를 사용하지 않기 때문에 _로 치환
                if done:
                    state_next = None

                    # 최근 10 epi에서 버틴 단계 수를 리스트에 저장
                    episode_suc_list = np.hstack((episode_suc_list[1:], step + 1))
                    if step < SUCCESS_STEPS:
                        # 도중에 봉이 쓰러지면 패널티로 보상 -1
                        reward = torch.FloatTensor([-1.0])
                        # 연속 성공 에피소드 기록 초기화
                        complete_episodes = 0
                    else:
                        # 봉이 서있는채로 끝나면 보상 1 부여
                        reward = torch.FloatTensor([1.0])
                        # 연속 성공 에피소드 기록 갱신
                        complete_episodes += 1

                else:
                    reward = torch.FloatTensor([0.0])  # 그외의 경우는 보상 0 부여
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
                    try:
                        print("Last step's loss : "+str(self.agent.brain.alpha_score.item()) + "\n")
                    except:
                        print("...warmup..."+'\n')
                    if (episode % 5 == 0):
                        self.agent.update_target_q_function()
                    self.success_list.append(step+1)
                    try:
                        self.loss_list.append(self.agent.brain.alpha_score.item())
                    except:
                        self.loss_list.append(0)

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
                print(str(SUCCESS_PHASE) + " 에피소드 연속 성공, 마지막 에피소드:" + str(episode))
                episode_final = True

        self.success_list = np.array(self.success_list)
        self.loss_list = np.array(self.loss_list)

        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10), sharex=True,)

        ax1.plot(self.success_list, label="Num of success")
        ax1.set_ylabel("success stes")
        ax1.set_xlabel("Episodes")

        ax2.plot(self.loss_list, label="Last loss value in Episode")
        ax2.set_ylabel("loss")
        ax2.set_xlabel("Episodes")

        plt.title("DDQN+GWO")
        plt.show()

if __name__ == "__main__" :
    cartpole_env = Environment()
    cartpole_env.run()