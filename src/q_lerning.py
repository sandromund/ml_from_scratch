import pylab as plt
import networkx as nx
import random


def plot_wolrd(actions):
    points =[]
    for i in range(len(actions)):
        for j in actions[i]:
            points += [(i, j)]
    G = nx.DiGraph()
    G.add_edges_from(points)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos, alpha=0.5, arrows=True)
    nx.draw_networkx_edges(G,pos, stlye='dashdot')
    nx.draw_networkx_labels(G,pos)
    plt.show()


def update_Q_matrix(Q, state, action, value):
    Q[state] =  Q[state][:action] + [value] + Q[state][action+1:]
    return Q_matrix


if __name__ == '__main__':
    actions = [[1, 2], [2, 5], [7], [4], [7], [6, 4], [3], []]


    rewards = [-1, -1, -100, 10, -1, -1, 10, -1, 100]
    goal = 7
    start = 0
    danger = 2
    n_states = 8

    Q = [[0] * n_states] * n_states

    epsilon = 1.0            # Greed 100%
    epsilon_min = 0.005      # Minimum greed 0.05%
    epsilon_decay = 0.99993  # Decay multiplied with epsilon after each episode
    n_episodes = 100         # Amount of games
    max_steps = 10           # Maximum steps per episode
    learning_rate = 0.65

    Q_matrix = [[0] * n_states] * n_states

    for episode in range(n_episodes):
        state = start
        # score = rewards[start]
        for step in range(max_steps):
            possible = actions[state]
            if epsilon < random.uniform(0, 1):
                best = -1000
                action = -1
                for p in possible:
                    if Q[state][p] > best:
                        action = p
                        best = Q[state][action]
            else:
                action = random.sample(possible, 1)[0]

            Q = update_Q_matrix(Q, state, state, rewards[action])

            future_reward = Q[action].index(max(Q[action]))

            value = (1 - learning_rate) * Q[state][action]
            value += learning_rate * (rewards[action] + future_reward)

            Q = update_Q_matrix(Q, state, action, value)

            state = action
            if state == goal or state == danger:
                break

    print(Q)

    current_state = 0
    steps = [current_state]

    while current_state != goal:
        possible = actions[current_state]
        best = -1000
        action = -1
        for p in possible:
            if Q[state][p] > best:
                action = p
                best = Q[current_state][action]
        next_step_index = action

        steps.append(next_step_index)
        current_state = next_step_index

    # Print selected sequence of steps
    print("Selected path:")
    print(steps)
    plot_wolrd(actions)

