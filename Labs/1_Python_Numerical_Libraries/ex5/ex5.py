import sys

def load_file(fname):
    f=open(fname)
    N=int(f.readline())
    lCoords = []
    for line in f:
        x, y = line.split()
        lCoords.append((int(x), int(y)))
    f.close()
    return N, lCoords

def matrix_list_zeros(N):
    m = []
    for i in range(N):
        m.append([0 for j in range(N)])
    return m

def matrix_dict_zeros(N):
    m = {}
    for i in range(N):
        for j in range(N):
            m[(i,j)] = 0
    return m

def print_matrix_list(m):

    for row in m:
        for col in row:
            print ('%.1f' % col, end=' ')
        print()

def print_matrix_dict(m, n):

    for i in range(n):
        for j in range(n):
            print ('%.1f' % m[(i, j)], end=' ')
        print()
        
if __name__ == '__main__':

    N, lC = load_file(sys.argv[1])
    matrix_list = matrix_list_zeros(N) # list of lists solution
    matrix_dict = matrix_dict_zeros(N) # dictionary solution
    
    for x, y in lC:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if x+dx >= 0 and x+dx < N and y+dy >= 0 and y+dy < N:
                    matrix_list[x+dx][y+dy] += 0.2
                    matrix_dict[(x+dx, y+dy)] += 0.2
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if x+dx >= 0 and x+dx < N and y+dy >= 0 and y+dy < N:
                    matrix_list[x+dx][y+dy] += 0.3
                    matrix_dict[(x+dx, y+dy)] += 0.3
        matrix_list[x][y] += 0.5
        matrix_dict[(x, y)] += 0.5

    print_matrix_list(matrix_list)
    print()
    print_matrix_dict(matrix_dict, N)
    