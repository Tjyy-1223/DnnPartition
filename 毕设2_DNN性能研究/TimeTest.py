import time

start_time = time.time()
# start_time = int(round(time.time()))

time.sleep(2)

end_time = time.time()
# end_time = int(round(time.time()))

print(f"{(end_time - start_time):.10f}")
print(f'computation time: {(end_time - start_time) / 1000 :.3f} s\n')