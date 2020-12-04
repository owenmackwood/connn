# ConnN: A framework for playing Connect Four tournaments

I wrote this project to run tournaments for game-playing agents written by my students. The idea came from
an undergraduate AI course I took (many years ago now). In contrast to that course, the objective of the course I 
taught was not just to teach them about AI, but also good software development practices. My students came from
various backgrounds, and most had limited software development experience. As such, I created a few lessons to try
and get them up to speed as quickly as possible. The content of those lessons can be found in the [Wiki](https://github.com/owenmackwood/connn/wiki).

## Requirements

All users need the following Python packages:
- NumPy
- Stockings

Anyone that want to use the `connectn.results` module needs:
- PyTables

Both the client and the server need to have Stunnel installed.

### Server
Due to compatibility constraints (for GridMap), the server needs to be running in an environment with Python 3.6. 
The following Python packages also need to be installed in that environment:

- NumPy
- Stockings
- PyTables
- Numba

If you want to run on a compute cluster, you'll also need:
- GridMap

## Tournament Participants

To participate in a running tournament, you'll need to use the client script to upload your agent, and eventually 
download results. You'll need to provide your group name, password, and the location of your agent. These can
be passed as command line arguments (run `python client.py -h` for details), or hardcoded at the top of `client.py`.

## Tournament Administrators

If you want to run your own tournament, just run `server.py`. That script will start a process to run 
the actual tournament, and then will begin to handle requests from clients (to upload new agents, or download results).
If the tournament is running on a remote machine, use `screen` so it doesn't shut down when you disconnect (if you
later reconnect to the remote machine, just  run `screen -r` to re-attach to the screen session).

While the server is running, if you send an interrupt signal (by pressing `Ctrl-C` in the terminal or the stop button 
if you're running in an IDE) it will ask if you want to shut down the server (it does this by 
telling the tournament process to shutdown and then exiting). If you say no, it will ask you if you want to run an 
"all-against-all" tournament. This is not normally necessary though, since by default the tournament plays 
"all-against-all" automatically every 6 hours.

For details on the various configuration options for the server, just run `python server.py -h` on the command line.
