import cv2
import sys
sys.path.append("game/")
from Atari import Atari 
from BrainDQN_Nature import *
import numpy as np 
import matplotlib.pyplot as plt
from ale_python_interface import ALEInterface
'''
atari = ALEInterface()

# Get & Set the desired settings
atari.setInt(b'random_seed', 123)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = False
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    atari.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    atari.setBool('sound', True)
  atari.setBool('display_screen', True)
'''
# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
	observation = observation[26:110,:]
	#ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	#observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        return np.reshape(observation,(84,84,1))

def playAtari():
	# Step 1: init BrainDQN
	# Step 2: init Flappy Bird Game
	atari = Atari('pong.bin')
	#rom_file = 'game/pong.bin'
	#atari.loadROM(rom_file)
	#atari.setBool('display_screen', False)
	actions = len(atari.legal_actions)
	brain = BrainDQN(actions)
	
	
	# Step 3: play game
	# Step 3.1: obtain init state
	action0 = np.array([1,0,0,0])  # do nothing
	observation0, reward0, terminal = atari.next(action0)
        cv2.imwrite('orig.jpg',observation0)
	
        observation0 = cv2.cvtColor(cv2.resize(observation0, (84, 110)), cv2.COLOR_BGR2GRAY)
        cv2.imwrite('gray.jpg',observation0)
	
        observation0 = observation0[26:110,:]
	cv2.imwrite('crop.jpg',observation0)
        #cv2.waitKey(1000)
        #cv.destroyAllWindows()
        #ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
        #observation0 = cv2.cvtColor(observation0, cv2.COLOR_BGR2GRAY)
	
        brain.setInitState(observation0)
        '''
        fig = plt.figure()
        plt.imshow(observation0)
        fig.savefig('trrr.png')
        import time
        time.sleep(5)
	'''
        # Step 3.2: run the game
	while 1!= 0:
		action = brain.getAction()
		nextObservation,reward,terminal = atari.next(action)
		nextObservation = preprocess(nextObservation)
		brain.setPerception(nextObservation,action,reward,terminal)

def main():
	playAtari()

if __name__ == '__main__':
	main()
