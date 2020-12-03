# ConnN: A framework for playing Connect Four tournaments

## Requirements

All users need the following Python packages:
- NumPy
- Stockings

Anyone that want to use the `connectn.results` module needs:
- PyTables

Both the client and the server need to have Stunnel installed.

### Server
Due to compatibility constraints, the server needs to be running in an environment with Python 3.6. 
The following Python packages also need to be installed in that environment:

- NumPy
- Stockings
- PyTables
- GridMap
- Numba
