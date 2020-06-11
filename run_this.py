from lonelycity_env import City
from RL_brain import DeepQNetwork
import numpy as np

reward_his = []

def run_city():
    step = 0
    maxreward=-5
    for episode in range(1001):
        # initial observation
        observation = env.resetcity()
        #print('Episode=',episode)
        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                reward_his.append(reward)
                #if(episode>200) and (reward>maxreward):
                    #maxreward=reward
                if(episode>200) and (reward>0.51985524359):
                    env.draw()
                break
            step += 1

        #show the trained agent decisions
        if episode%100==0:
            print('Episode=',episode)
            #env.draw()
        
    print (maxreward)

    # end of game
    print('game over')
    env.destroy()

#add by cui
def plot_reward():
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(reward_his)), reward_his)
    plt.ylabel('Reward')
    plt.xlabel('training episodes')
    plt.show()


if __name__ == "__main__":
    env = City()
    RL = DeepQNetwork(env.n_actions, env.n_features)
    run_city
    env.after(100, run_city)
    env.mainloop()
    RL.plot_cost()
    plot_reward()