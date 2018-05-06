from mpi4py import MPI as mpi

def scatter_list(x):
    comm = mpi.COMM_WORLD
    cake = list()
    n_el = len(x)
    n_cpu = comm.size
    if comm.rank == 0:
        if n_el % n_cpu == 0:
            pieceSize = n_el/n_cpu
        else:
            pieceSize = n_el/n_cpu + 1
        for i, xi in enumerate(x):
            if i % pieceSize == 0:
                cake.append(list())
            cake[-1].append(xi)
    while len(cake) < n_cpu:
        cake.append(list())
    return comm.scatter(cake, root = 0)

def allgather_list(x):
    comm = mpi.COMM_WORLD
    cake = comm.allgather(x)
    y = list()
    for piece in cake:
        for xi in piece:
            y.append(xi)
    return y

