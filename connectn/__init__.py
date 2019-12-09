import os, sys

IS_DEBUGGING = sys.gettrace() is not None
if IS_DEBUGGING:
    os.environ["NUMBA_DISABLE_JIT"] = "1"
# os.environ["NUMBA_PARALLEL_DIAGNOSTICS"] = "4"
