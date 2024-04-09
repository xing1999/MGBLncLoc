import pandas as pd
import random
import numpy as np
k = 3
# def load_tsv_format_data(filename, skip_head=True):
#     sequences = []
#     labels = []
#     df = pd.read_csv(filename, sep='\t')
#     k = 0.1
#     for label in df['label'].unique():
#         label_data = df[df['label'] == label]
#         label_size = len(label_data)
#         sample_size = int(label_size * k)
#         if sample_size > 0:
#             sampled_data = label_data.sample(n=sample_size, random_state=42)
#             sequences.extend(sampled_data['text'].tolist())
#             labels.extend(sampled_data['label'].tolist())
#     print("sequences number:{}".format(len(sequences)))
#     return sequences, labels

def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))
    return sequences, labels
def generate_kmers(sequence, k):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        kmers.append(kmer)
    # print(kmers)
    return kmers

def generate_vocab(sequences, k):
    vocab = set()
    for sequence in sequences:
        kmers = generate_kmers(sequence, k)
        vocab.update(kmers)
    vocab = [item for item in vocab if 'N' not in item]
    return sorted(list(vocab))

def count_total_kmers(sequences, k):
    total_kmers = 0
    for sequence in sequences:
        total_kmers += len(sequence) - k + 1
    return total_kmers

def generate_frequency_matrix(vocab, sequences):
    positions = max(len(sequence) for sequence in sequences)
    # print("positions:{}".format(positions))
    matrix = [[0] * len(vocab) for _ in range((positions-k+1))]
    for i, K_mer in enumerate(sequences):
        kmers=generate_kmers(K_mer, k)
        for j, k_mer in enumerate(kmers):
            for p,Vocab in enumerate(vocab):
                if Vocab == k_mer:
                    matrix[j][p]=matrix[j][p]+1
    return matrix

def Generate_posden(vocab,sequences,k):
    NS = count_total_kmers(sequences, k)
    Z=generate_frequency_matrix(vocab,sequences)
    Z_den = []
    # print(Z)
    for row in Z:
        new_row = [element / NS for element in row]
        Z_den.append(new_row)
    return Z_den

def calculate_MCD(K1, K2,K3,K4):
    rows = len(K1)
    cols = len(K1[0])
    p = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            min_val = min(K1[i][j], K2[i][j],K3[i][j], K4[i][j])
            if min_val == 0:
                min_val = 0.1
            p[i][j] = (K1[i][j] - K2[i][j] - K3[i][j] - K4[i][j]) / min_val
    return p

def get_MCD(Vocab,lable1_seq,lable2_seq,lable3_seq,lable4_seq,k):
    Z_lable1 = Generate_posden(Vocab, lable1_seq, k)
    Z_lable2 = Generate_posden(Vocab, lable2_seq, k)
    Z_lable3 = Generate_posden(Vocab, lable3_seq, k)
    Z_lable4 = Generate_posden(Vocab, lable4_seq, k)
    p = calculate_MCD(Z_lable1,Z_lable2 ,Z_lable3,Z_lable4)
    return p

def encode(sequences,vocab,k,p):
    PoCD = []
    for seq in sequences:
        l=len(seq)
        k_mers=generate_kmers(seq, k)
        Pocd = np.zeros([l])
        for x, k_mer in enumerate(k_mers):
            for w,voc in enumerate(vocab):
                if k_mer==voc:
                    Pocd[x]=p[x][w]
        PoCD.append(Pocd)
    return PoCD
# def get_seq():
def get_seq(file_path):
    sequences,label =load_tsv_format_data(file_path)
    # sequences,label =a,b
    lable1_seq,lable2_seq,lable3_seq,lable4_seq=[],[],[],[]
    for i in range(len(label)):
        if label[i]==0:
            a=sequences[i]
            lable1_seq.append(a)
        elif label[i]==1:
            b=sequences[i]
            lable2_seq.append(b)
        elif label[i]==2:
            c=sequences[i]
            lable3_seq.append(c)
        elif label[i]==3:
            d=sequences[i]
            lable4_seq.append(d)
    return sequences,lable1_seq,lable2_seq,lable3_seq,lable4_seq,label

# def NCP_ND_encode(sequence):
#     sequence_length = len(sequence)
#     encoding_matrix = np.zeros((4, sequence_length), dtype=np.float32)
#     NCP = {'A': [1, 1, 1], 'T': [0, 1, 0], 'G': [1, 0, 0], 'C': [0, 0, 1]}
#     # NCP
#     for pos in range(sequence_length):
#         encoding_matrix[0:3, pos] = np.asarray(np.float32(NCP[sequence[pos]]))
#
#     # ND
#     nd_encoding = []
#     for i in range(len(sequence)):
#         nd_value = 0
#         for j in range(len(sequence)):
#             if sequence[j] == sequence[i]:
#                 nd_value += 1
#         # Normalize by dividing by sequence length
#         nd_value /= len(sequence)
#         nd_encoding.append(nd_value)
#
#
#     encoding_matrix[3, :] = np.asarray(np.float32(nd_encoding))
#     return encoding_matrix

# def DPCP_encode(sequence):
#     sequence_length = len(sequence)
#     DPCP = {'AA': [0.5773884923447732, 0.6531915653378907, 0.6124592000985356, 0.8402684612384332, 0.5856582729115565,
#                    0.5476708282666789],
#             'AT': [0.7512077598863804, 0.6036675879079278, 0.6737051546096536, 0.39069870063063133, 1.0,
#                    0.76847598772376],
#             'AG': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764,
#                    0.45903777008667923],
#             'AC': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944,
#                    0.7467063799220581],
#             'TA': [0.3539063797840531, 0.15795248106354978, 0.48996729107629966, 0.1795369895818257, 0.3059118434042811,
#                    0.32686549630327577],
#             'TT': [0.5773884923447732, 0.6531915653378907, 0.0, 0.8402684612384332, 0.5856582729115565,
#                    0.5476708282666789],
#             'TG': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195,
#                    0.3501900760908136],
#             'TC': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026,
#                    0.6891727614587756],
#             'GA': [0.5525570698352168, 0.6531915653378907, 0.6124592000985356, 0.5882368974116978, 0.49856742124957026,
#                    0.6891727614587756],
#             'GT': [0.8257018549087278, 0.6531915653378907, 0.7043281318652126, 0.5882368974116978, 0.7888705476333944,
#                    0.7467063799220581],
#             'GG': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261,
#                    0.6083143907016332],
#             'GC': [0.5525570698352168, 0.6036675879079278, 0.7961968911255676, 0.5064970193495165, 0.6780274730118172,
#                    0.8400043540595654],
#             'CA': [0.32907512978081865, 0.3312861433089369, 0.5205902683318586, 0.4179453841534657, 0.45898067049412195,
#                    0.3501900760908136],
#             'CT': [0.7015450873735896, 0.6284296628760702, 0.5818362228429766, 0.6836002897416182, 0.5249586459219764,
#                    0.45903777008667923],
#             'CG': [0.2794124572680277, 0.3560480457707574, 0.48996729107629966, 0.4247569687810134, 0.5170412957708868,
#                    0.32686549630327577],
#             'CC': [0.5773884923447732, 0.7522393476914946, 0.5818362228429766, 0.6631651908463315, 0.4246720956706261,
#                    0.6083143907016332]}
#     feature_matrix = np.zeros((6, sequence_length))
#     # 计算序列中每个位置的DPCP值
#     for i in range(1, sequence_length+1):
#         if i == 1:
#             feature_matrix[:, 0] = np.array(DPCP[sequence[0] + sequence[1]])
#         if i == sequence_length:
#             feature_matrix[:, i-1] = np.array(DPCP[sequence[i-2] + sequence[i-1]])
#         elif 1<i<sequence_length:
#             di_1 = sequence[i - 2] + sequence[i-1]  # 当前碱基对
#             di_2 = sequence[i-1] + sequence[i]  # 前一个碱基对
#             feature_matrix[0:, i-1] = (np.array(DPCP[di_1]) + np.array(DPCP[di_2])) / 2
#
#     return feature_matrix


# def encode_multiple_sequences(sequences):
#     # 
#     # encoded_sequences_ncp = []
#     # encoded_sequences_dpcp = []
#     #
#     for sequence in sequences:
#         # encoded_sequence_ncp = NCP_ND_encode(sequence)
#         # encoded_sequences_ncp.append(encoded_sequence_ncp)
#         #
#         # encoded_sequence_dpcp = DPCP_encode(sequence)
#         # encoded_sequences_dpcp.append(encoded_sequence_dpcp)
#
#  
#     encoded_sequences_ncp_nd = np.array(encoded_sequences_ncp)
#     NCP_ND_matrix_input = torch.from_numpy(encoded_sequences_ncp_nd)
#
#
#     encoded_sequences_DPCP = np.array(encoded_sequences_dpcp)
#     DPCP_matrix_input = torch.from_numpy(encoded_sequences_DPCP)
#
#     return NCP_ND_matrix_input,DPCP_matrix_input


# 
# gene_sequences = ["ATCGTT","ATCGTT","ATCGTT","ATCGTT","ATCGTT"]
# gene_sequences = ["ACGUC","UUACU","UCGUA","UUCGU"]
# label = ["1","2","3","4"]
# 
# def load_data(g,l):
def load_data(file_path):#,MCD):

    sequences, lable1_seq,lable2_seq,lable3_seq,lable4_seq, label = get_seq(file_path)
    # print(sequences,lable1_seq,lable2_seq,lable3_seq,lable4_seq,label)
    # sequences, pos_seq, neg_seq, label =["ACGUC","UUACU","UCGUA","UUCGU"],["ACGUC","UCGUA"],["UUACU","UUCGU"],[]
    Vocab = generate_vocab(sequences, k)
    # print(Vocab)
    MCD=get_MCD(Vocab,lable1_seq,lable2_seq,lable3_seq,lable4_seq,k)
    data_encode=encode(sequences,Vocab,k,MCD)
    # pocd = [torch.from_numpy(array) for array in data_encode]
    # encoded_pocd = np.array(data_encode)
    # pocd = torch.from_numpy(encoded_pocd)
    # pocd = pocd.unsqueeze(1)
    # label_list_input = np.asarray([i for i in label], dtype=np.int64)
    # label_list_input = torch.from_numpy(label_list_input)

    # ncp_nd, dpcp = encode_multiple_sequences(sequences)

    # all_matrix_iinput = torch.cat((pocd,ncp_nd, dpcp), dim=1)

    # return sequences,ncp_nd, dpcp, pocd,all_matrix_iinput, label_list_input
    return label,data_encode,MCD
def load_test_data(file_path,MCD):

    sequences,label= load_tsv_format_data(file_path)
    # print(sequences,lable1_seq,lable2_seq,lable3_seq,lable4_seq,label)
    # sequences, pos_seq, neg_seq, label =["ACGUC","UUACU","UCGUA","UUCGU"],["ACGUC","UCGUA"],["UUACU","UUCGU"],[]
    Vocab = generate_vocab(sequences, k)
    # print(Vocab)
    # POCD=get_POCD(Vocab,lable1_seq,lable2_seq,k)
    data_encode=encode(sequences,Vocab,k,MCD)
    # pocd = [torch.from_numpy(array) for array in data_encode]
    # encoded_pocd = np.array(data_encode)
    # pocd = torch.from_numpy(encoded_pocd)
    # pocd = pocd.unsqueeze(1)
    # label_list_input = np.asarray([i for i in label], dtype=np.int64)
    # label_list_input = torch.from_numpy(label_list_input)

    # ncp_nd, dpcp = encode_multiple_sequences(sequences)

    # all_matrix_iinput = torch.cat((pocd,ncp_nd, dpcp), dim=1)

    # return sequences,ncp_nd, dpcp, pocd,all_matrix_iinput, label_list_input
    return sequences,label,data_encode