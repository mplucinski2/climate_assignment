import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

T = np.linspace(210, 260, 400)

coeff_liquid = {'b': 6.5459673,'a_m1': -0.58002206e4,'a0': 1.3914993, 'a1': -0.48640239e-1,'a2': 0.41764768e-4,'a3': -0.14452093e-7, 'a4': 0.0}

coeff_ice = {'b': 4.1635019, 'a_m1': -0.56745359e4, 'a0': 6.3925247, 'a1': -0.96778430e-2, 'a2': 0.62215701e-6, 'a3': 0.20747825e-8,'a4': -0.94840240e-12}

def sat_pressure(T, coeff):
    ln_ep = ( coeff['b']*np.log(T) + coeff['a_m1']/T + coeff['a0'] + coeff['a1']*T + coeff['a2']*T**2 + coeff['a3']*T**3 + coeff['a4']*T**4)
    return np.exp(ln_ep)

e_liquid = sat_pressure(T, coeff_liquid)
e_ice    = sat_pressure(T, coeff_ice)

def mixing_line(T_line, T0, p0, G):
    return p0 + G*(T_line - T0)

def find_saturation_temp(T0, p0, G, T_guess, coeff):
    def objective(T):
        p_mix = p0 + G*(T - T0)
        p_sat = sat_pressure(T, coeff)
        return p_mix - p_sat
    
    T_sat = fsolve(objective, T_guess)[0]
    p_sat = p0 + G*(T_sat - T0)
    return T_sat, p_sat

c_P = 1004        
epsilon = 0.622
EI_kerosene = 1.25  
LHV_kerosene = 43.2e6  
EI_hydrogen = 8.94  
LHV_hydrogen = 120e6 

p_amb_b = 1.1 * sat_pressure(225, coeff_ice) 
print(sat_pressure(225, coeff_ice))
print(p_amb_b)
p_total_b = 220e2  
eta_b = 0.3
G_b = (c_P * p_total_b / epsilon) * (EI_kerosene / ((1-eta_b)*LHV_kerosene))
T0_b = 225 

eta_c = 0.4
G_c = (c_P * p_total_b / epsilon) * (EI_kerosene / ((1-eta_c)*LHV_kerosene))

G_d = (c_P * p_total_b / epsilon) * (EI_hydrogen / ((1-eta_b)*LHV_hydrogen))

T_sat_b, p_sat_b = find_saturation_temp(T0_b, p_amb_b, G_b, 235, coeff_ice)
T_sat_c, p_sat_c = find_saturation_temp(T0_b, p_amb_b, G_c, 235, coeff_ice)
T_sat_d, p_sat_d = find_saturation_temp(T0_b, p_amb_b, G_d, 235, coeff_ice)

T_line_b = np.linspace(T0_b, 260, 200)
T_line_c = np.linspace(T0_b, 260, 200)
T_line_d = np.linspace(T0_b, 260, 200)

mix_b = mixing_line(T_line_b, T0_b, p_amb_b, G_b)
mix_c = mixing_line(T_line_c, T0_b, p_amb_b, G_c)
mix_d = mixing_line(T_line_d, T0_b, p_amb_b, G_d)

plt.figure(figsize=(8,6))
plt.plot(T, e_liquid, 'b--', label=r'$e_\ell(T)$')
plt.plot(T, e_ice, 'r--', label=r'$e_i(T)$')
plt.plot(T_line_b, mix_b, 'k-', label='old kerosene ($\eta=0.3)$')
plt.plot(T_line_c, mix_c, 'g-', label='new kerosene ($\eta=0.4$)')
plt.plot(T_line_d, mix_d, 'm-', label='hydrogen ($\eta=0.3$)')

plt.scatter(T0_b, p_amb_b, color='green', s=80, zorder=6, label=f'Ambient point ({T0_b} K)')

plt.ylabel('partial pressure of H$_2$O [Pa]')
plt.xlabel('temperature [K]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xlim(210, 260)

plt.show()

T0_f = 230
p_total_f = 250e2 
p_amb_f = 0.60 * sat_pressure(T0_f, coeff_liquid)

G_b_f = (c_P * p_total_f / epsilon) * (EI_kerosene / ((1-eta_b)*LHV_kerosene))
G_c_f = (c_P * p_total_f / epsilon) * (EI_kerosene / ((1-eta_c)*LHV_kerosene))
G_d_f = (c_P * p_total_f / epsilon) * (EI_hydrogen / ((1-eta_b)*LHV_hydrogen))

T_sat_b_f, p_sat_b_f = find_saturation_temp(T0_f, p_amb_f, G_b_f, 245, coeff_liquid)
T_sat_c_f, p_sat_c_f = find_saturation_temp(T0_f, p_amb_f, G_c_f, 245, coeff_liquid)
T_sat_d_f, p_sat_d_f = find_saturation_temp(T0_f, p_amb_f, G_d_f, 245, coeff_liquid)

T_line_b_f = np.linspace(T0_f, 280, 200)
T_line_c_f = np.linspace(T0_f, 280, 200)
T_line_d_f = np.linspace(T0_f, 280, 200)

mix_b_f = mixing_line(T_line_b_f, T0_f, p_amb_f, G_b_f)
mix_c_f = mixing_line(T_line_c_f, T0_f, p_amb_f, G_c_f)
mix_d_f = mixing_line(T_line_d_f, T0_f, p_amb_f, G_d_f)

plt.figure(figsize=(8,6))
plt.plot(T, e_liquid, 'b--', label=r'$e_\ell(T)$')
plt.plot(T, e_ice, 'r--', label=r'$e_i(T)$')
plt.plot(T_line_b_f, mix_b_f, 'k-', label='old kerosene ($\eta=0.3$)')
plt.plot(T_line_c_f, mix_c_f, 'g-', label='new kerosene ($\eta=0.4$)')
plt.plot(T_line_d_f, mix_d_f, 'm-', label='hydrogen ($\eta=0.3$)')

plt.scatter(T0_f, p_amb_f, color='green', s=80, zorder=6, label=f'Ambient point ({T0_f} K)')

plt.xlabel('temperature [K]')
plt.xlim(210, 260)
plt.ylabel('partial pressure of H$_2$O [Pa]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
