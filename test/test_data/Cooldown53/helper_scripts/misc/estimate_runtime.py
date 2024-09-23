# %%
import numpy as np

A = 60
B = 100
C = B*10

N_A = len(np.arange(-10, -20, -2))
N_B = len(np.arange(-20, -45, -4))
N_C = len(np.arange(-50, -60, -5))

T_A = N_A*A
T_B = N_B*B
T_C = N_C*C

N_res = 7

T_mins = N_res*(T_A+T_B+T_C)/60

print(f"({N_A} high powers * {A} secs/ea) + ({N_B} med powers * {B} secs/ea) + ({N_C} low powers * {C} secs/ea) ")
print(f"   = ({T_A/60:1.2f} mins + {T_B/60:1.2f} mins + {T_C/60:1.2f} mins) * {N_res} resonators")
print(f"   = {T_mins:1.2f} mins ({T_mins/60:1.2f} hrs)")


# %%# %% time analysis
N_res = 7 #len(fcs)

T_A = 60 #HPow_elapsed_time[0]/N_res
T_B = 100
T_C = 10*T_B #LPow_elapsed_time[0]/N_res  

N_A = len(high_powers);  N_B = len(med_powers);  N_C = len(low_powers)
A = T_A/N_A;             B = T_B/N_B;            C = T_C/N_C

T_mins = N_res*(T_A+T_B+T_C)/60

total_elapsed_time = (time.time() - tStart_high)/60

print(f"Measurement time analysis")
print(f" T_res =   ({N_A} high powers * {A:1.2f} secs/ea)")
print(f"            + ({N_B} med powers * {B:1.2f} secs/ea)")
print(f"            + ({N_C} low powers * {C:1.2f} secs/ea) \n")
print(f" T_res  = ({T_A/60:1.2f} mins + {T_B/60:1.2f} mins + {T_C/60:1.2f} mins) = {T_mins:1.2f} mins/res")
print(f" T_res * {N_res} resonators = {T_mins*N_res:1.2f} mins ({T_mins*N_res/60:1.2f} hrs)")

print(time.strftime("%a, %b %d, %H:%M %p"))
print(f"\ntime.time() reported elapsed time: {total_elapsed_time:1.2f} mins ({total_elapsed_time/60:1.2f} hrs)")
# %%
