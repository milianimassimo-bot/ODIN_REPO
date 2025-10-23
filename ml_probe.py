import inspect, sys
import odin_ml_logger as m

print("FILE:", m.__file__)
print("SIGNATURE log_outcome:", inspect.signature(m.log_outcome))
print("SOURCE (prime 20 righe):")
src = inspect.getsource(m.log_outcome)
print("\n".join(src.splitlines()[:20]))

print("\nPYTHONPATH:")
for p in sys.path:
    if "AUTOMAZIONE" in p.upper():
        print(" -", p)
