import numpy as np
import itertools
# read file
def read_file(lines):
    ID = []
    Seq = []
    Raw_lab = []
    for i in range(len(lines)):
        if i % 3 == 0:
            ID.append(lines[i])
        if i % 3 == 1:
            Seq.append(lines[i])
        if i % 3 == 2:
            Raw_lab.append(lines[i])
    return ID,Seq,Raw_lab

# str to number
# mapping dictionary
def seq2num(seq, dic1):
    seq1 = []
    for s in seq:
        s1 = dic1[s]
        seq1.append(s1)
    return np.array(seq1)


def hilbert_curve(n):
    # recursion base
    if n == 1:
        return np.zeros((1, 1), np.int32)
    # make (n/2, n/2) index
    t = hilbert_curve(n // 2)
    # flip it four times and add index offsets
    a = np.flipud(np.rot90(t))
    b = t + t.size
    c = t + t.size * 2
    d = np.flipud(np.rot90(t, -1)) + t.size * 3
    # and stack four tiles into resulting array
    return np.vstack(map(np.hstack, [[a, b], [d, c]]))


def plot_hb_dna(seq, H_curve, sub_length,map_dic):
    r, c = H_curve.shape
    num_A = one_hot(seq, sub_length, map_dic)
    H_dna = np.zeros((r, c, 4 ** sub_length))
    for i in range(len(num_A)):
        x, y = np.where(H_curve == i)
        H_dna[x, y, :] = num_A[i, :]
    return H_dna


def plot_row(seq,sub_length,map_dic):
    num_A = one_hot(seq, sub_length, map_dic)
    H_dna = -1.*np.ones((500, 4 ** sub_length))
    for i in range(len(num_A)):
        H_dna[i, :] = num_A[i, :]
    return H_dna

# 1-d sequence
def plot_row1(seq,sub_length,map_dic):
    num_A = one_hot(seq, sub_length, map_dic)
    H_dna = -1.*np.ones((500, 1,4 ** sub_length))
    for i in range(len(num_A)):
        H_dna[i, 0, :] = num_A[i, :]
    return H_dna

# one hot encoding
def one_hot(sequence, sub_len, mapping_dic):
    n_ = len(sequence)
    sub_list = []
    for i in range(n_ - sub_len + 1):
        sub_list.append(sequence[i:i + sub_len])
    res_ = []
    for sub in sub_list:
        res_.append(mapping_dic[sub])

    return np.array(res_)
# cut the sequence
def cut(seq,sub_length):
    n = len(seq)
    new = []
    for i in range(n-sub_length+1):
        new.append(seq[i:i+sub_length])
    return np.array(new)

# generate elements of the sequence
def element(seq_list):
    list_ = []
    for s in seq_list:
        if s not in list_:
            list_.append(s)
    return list_

# generate mapping dictionary
def combination(elements, seq_length):
    keys = map(''.join, itertools.product(elements, repeat=seq_length))
    n_word = len(keys)
    array_word = np.eye(n_word)
    mapping_dic = {}
    for i in range(n_word):
        mapping_dic[keys[i]] = array_word[i,:]
    return mapping_dic

## Generate various curve
# diagnal snake curve
def diag_snake(m,n):
    H = np.zeros((m,n))
    count = 0
    for i in range(0,m+n-1):
        if i % 2 ==0:
            for x in range(m):
                for y in range(n):
                    if (x+y) == i:
                        H[x,y] = count
                        count +=1
        elif i%2 == 1:
            for x in range(m-1,-1,-1):
                for y in range(n):
                    if (x+y) == i:
                        H[x, y] = count
                        count += 1
    return H
# reshape curve
def reshape_curve(m,n):
    return np.array((m*n)).reshape((m,n))
# snake curve
def snake_curve(m,n):
    H = np.arange(m*n).reshape((m,n))
    for i in range(m):
        if i%2==0:
            temp = H[i,:]
            H[i,:] = temp[::-1]
    return H
