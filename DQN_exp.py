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

BATCH_SIZE = 64
CAPACITY = 100000 # capacity of Experience Replay

MAX_ITERATION = 500
MAX_STEPS = 200 # Max steps per 1 Iteration
GAMMA = 0.99 # 할인율
EPSILON = 1 # initial epsilon value
SUCCESS_STEPS = 195
SUCCESS_PHASE = 10

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
    def __init__(self, num_states, num_actions, epsilon):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        self.epsilon = epsilon # 초기 앱실론
        self.epsilon_decay = 0.05
        self.epsilon_min = 0
        self.expected_state_action_values = torch.zeros(1)

        self.n_in, self.n_mid1, self.n_mid2, self.n_out = num_states, hid1_dim, hid2_dim, num_actions
        self.main_q_network = Net(self.n_in, self.n_mid1, self.n_mid2, self.n_out) # Main Q Network
        self.target_q_network = copy.deepcopy(self.main_q_network) # Target Q Network
        self.optimizer = torch.optim.Adam(self.main_q_network.parameters(), lr=0.0001)

    def replay(self, episode):
        # 저장된 트랜지션 수 확인, 만약 메모리가 배치사이즈보다 작으면 아무것도 x

        if len(self.memory) < BATCH_SIZE:
            return

        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main_q_network()

    def decide_action(self, state, episode):
        # e-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다.
        # self.epsilon = (episode * 0.99) / episode
        if episode < 20:
            self.epsilon = 1
        elif episode == 20:
            self.epsilon = 1
            print("warm-up phase is over")
        else :
            self.epsilon = 1 - self.epsilon_decay * (episode - 20)
            if self.epsilon <= self.epsilon_min:
                self.epsilon = self.epsilon_min

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

    def get_expected_state_action_values(self):
        # 신경망을 추론모드로 전환
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
        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 정답신호로 사용할 Q(s_t, a_t)값을 Q러닝 식으로 계산
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values


    def update_main_q_network(self):
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
    def __init__(self, num_states, num_actions, EPSILON):
        self.brain = Brain(num_states, num_actions, EPSILON)

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
        self.agent = Agent(num_states, num_actions, EPSILON)
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
            if episode is 20:
                print("warm-up phase is over")
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
                    print('%d Episode:\n Success steps = %d \n최근 %d 에피소드의 평균 단계 수 = %.1lf' % (episode, step + 1, SUCCESS_PHASE, episode_suc_list.mean()))
                    print("epsilon value = " + str(self.agent.brain.epsilon))

                    if (episode % 10 == 0):
                        self.agent.update_target_q_function()
                    self.success_list.append(step+1)
                    break

            if episode_final is True:
                self.env.close()
                break

            # 목표 에피소드 연속으로 195단계를 버티면 성공
            if complete_episodes >= SUCCESS_PHASE:
                stop = timeit.default_timer()
                print('Computational time:' + str(stop - start))
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