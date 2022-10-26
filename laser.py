from matplotlib import pyplot as plt
import numpy as np
from os.path import join as pathjoin

class Laser:
  def __init__(self, I, tau_s, sigma, N_i, S_i, time, step):
    self.I = I
    self.tau_s = tau_s
    self.sigma = sigma

    self.time = time
    self.step = step
    self.timestep = np.arange(0, self.time, self.step)
    self.timelen = len(self.timestep)

    self.k = np.zeros([self.timelen, 5, 2])
    self.NS = np.zeros([self.timelen, 2])
    self.NS[0] = [N_i, S_i]

    self.e = 1.602e-19
    self.d = 8e-9
    self.c = 3e8
    self.n = 3.65
    self.g_sl = 8.16e4
    self.Gamma = 0.2
    self.Area = np.pi * (5e-6 / 2)**2
    self.eta_inj = 0.7

    self.g_th = (self.c / self.n) * self.g_sl
    self.gamma_c = (self.c / self.n) * self.Gamma * self.g_sl 
    self.g_n = (self.c / self.n) * self.sigma
    self.g_p = -self.g_n

    self.folder = 'laserfig'

  def ddt(self, time_index, current_NS):
    self.g_rm = self.g_th + self.g_n * (current_NS[0] - 1.87e24) + self.g_p * (current_NS[1] - 4.42e19)

    if (time_index % 1 == 0):
      current_time = self.timestep[time_index] 
    else:
      time_index = int(time_index - 0.5)
      current_time = (self.timestep[time_index] + self.timestep[time_index + 1]) / 2
    J = self.eta_inj * (self.I(current_time) / self.Area)

    dNdt = (-1 / self.tau_s) * current_NS[0] + (-self.g_rm) * current_NS[1] + (J / (self.e * self.d))
    dSdt = (-self.gamma_c + self.Gamma * self.g_rm) * current_NS[1]
  
    return [dNdt, dSdt]

  def solve(self):
    for i in range(self.timelen - 1):
      self.k[i][0] = self.ddt(i + 0   ,  self.NS[i]                                 )
      self.k[i][1] = self.ddt(i + 0.5 , (self.NS[i] + self.k[i][1]*(self.step/2))   )
      self.k[i][2] = self.ddt(i + 0.5 , (self.NS[i] + self.k[i][2]*(self.step/2))   )
      self.k[i][3] = self.ddt(i + 1   , (self.NS[i] + self.k[i][3]* self.step)      )
      self.k[i][4] = (self.k[i][0] + 2*self.k[i][1] + 2*self.k[i][2] + self.k[i][3]) / 6
      self.NS[i + 1] = self.NS[i] + self.k[i][4] * self.step
    return self

  def graph_time(self, name):
    fig, ax1 = plt.subplots() 
  
    ax1.set_xlabel('Time') 
    ax1.set_ylabel('N', color = '#aaaaaa') 
    plot_1 = ax1.plot(self.timestep, self.NS[:, 0], color = '#aaaaaa') 
    ax1.tick_params(axis ='y', labelcolor = '#aaaaaa') 

    ax2 = ax1.twinx() 
  
    ax2.set_ylabel('S', color = '#000000') 
    plot_2 = ax2.plot(self.timestep, self.NS[:, 1], color = '#000000') 
    ax2.tick_params(axis ='y', labelcolor = '#000000') 

    plt.savefig(pathjoin(self.folder, (name + '.png')))
    return self

  def graph_NS(self, name):
    fig, ax = plt.subplots()

    ax.plot(self.NS[:, 0], self.NS[:, 1])

    ax.set_xlabel('N')
    ax.set_ylabel('S')

    plt.savefig(pathjoin(self.folder, (name + '.png')))
    return self

class fbLaser(Laser):
  def __init__(self, I, tau_s, sigma, N_i, S_i, time, step, fb_delay, fb_factor):
    Laser.__init__(self, I, tau_s, sigma, N_i, S_i, time, step)
    self.fbdelay = fb_delay
    self.fb_factor = fb_factor

  def ddt(self, time_index, current_NS):
    feedback = 0
    self.g_rm = self.g_th + self.g_n * (current_NS[0] - 1.87e24) + self.g_p * (current_NS[1] - 4.42e19)

    if (time_index % 1 == 0):
      current_time = self.timestep[time_index] 
    else:
      time_index = int(time_index - 0.5)
      current_time = (self.timestep[time_index] + self.timestep[time_index + 1]) / 2
    J = self.eta_inj * (self.I(current_time) / self.Area)
    
    if (current_time > self.fbdelay):
      feedback = self.NS[int(time_index - (self.fbdelay / self.step))][1]
      
    dNdt = (-1 / self.tau_s) * current_NS[0] + (-self.g_rm) * current_NS[1] + (J / (self.e * self.d))
    dSdt = (-self.gamma_c + self.Gamma * self.g_rm) * current_NS[1] + self.fb_factor * feedback

    return [dNdt, dSdt]