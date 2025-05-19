import os

for d in os.listdir("eval-mae-checkpoint"):
    new_d = "_".join(d.split("_")[2:])
    os.rename(f"eval-mae-checkpoint/{d}", f"eval-mae-checkpoint/{new_d}")
    print(new_d)