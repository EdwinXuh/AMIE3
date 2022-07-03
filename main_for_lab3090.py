#!/usr/bin/env python
# coding=utf-8

import sys
import random

sys.path.append('/home/vision/work/xuhong/TransGAT-iteration/')
import argparse
import logging
import os
import pickle
import time
from collections import defaultdict, Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.distributed
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import trange

from utils.early_stopping import EarlyStopping
from create_batch import Corpus
from utils.dataloader import CandidateDataset
from helper import set_logger, calculate_TN_acc, \
    get_groundings_set, find_topk, evaluate_repair, load_train_triple_rule, \
    batch_gat_loss_modify, batch_gat_loss_origin, evaluate_repair_modify, load_config, evaluate_repair_modify2, \
    calculate_inner_outer_grounding_score, calculate_inner_outer_score
from models import SpKBGATModified, SpKBGATConvOnly, TextCNN2D
from preprocess import init_embeddings, build_data
from query_strategies import EntropySampling
from rules.GroundAllRules import GroundAllRules
from rules.RuleSet import RuleSet
from utils.utils import get_unique_entity


def parse_args(dataset, experiment_name, seed, use_rule):
    args = argparse.ArgumentParser()
    args.add_argument("-seed", "--seed", type=int, default=seed, help="seed")
    args.add_argument("-patience_gat", "--patience_gat", default=20, help="early stopping patience of GAT")
    args.add_argument("-patience_conv", "--patience_conv", default=20, help="early stopping patience of ConvKB")
    args.add_argument("-rule", "--rule", default=use_rule, help="use rule or not",
                      choices=['without_rule', 'with_rule'])
    args.add_argument("-dataset", "--dataset", default=dataset, help="dataset")
    args.add_argument("-data", "--data", default="../dataset/" + dataset, help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int, default=1000, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int, default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float, default=1e-4, help="L2 regularization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float, default=1e-4, help="L2 regularization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool, default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int, default=200,
                      help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-log_out_folder", "--log_out_folder",
                      default="./checkpoints/" + experiment_name + '/' + dataset + '/logs', help="save the models.")
    args.add_argument("-embed_out_folder", "--embed_out_folder",
                      default="./checkpoints/" + experiment_name + '/' + dataset + '/embed', help="save the models.")
    args.add_argument("-detect_out_folder", "--detect_out_folder",
                      default="./checkpoints/" + experiment_name + '/' + dataset + '/detect', help="save the models.")
    args.add_argument("-repair_out_folder", "--repair_out_folder",
                      default="./checkpoints/" + experiment_name + '/' + dataset + '/repair', help="save the models.")
    args.add_argument("-outfolder", "--output_folder", default="./checkpoints/" + dataset,
                      help="Folder name to save the models.")
    
    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int, default=8544, help="Batch size for GAT")  # 86835
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int, default=2,
                      help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float, default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float, default=0.2, help="LeakyRelu alphas for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+', default=[200, 200],
                      help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+', default=[2, 2], help="Multi head attention SpGAT")
    args.add_argument("-margin", "--margin", type=float, default=1, help="Margin used in hinge loss")
    args.add_argument("-margin_rule", "--margin_rule", type=float, default=0.1, help="Margin Rule used in hinge loss")
    
    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int, default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float, default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=10,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=50, help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float, default=0,
                      help="Dropout probability for convolution layer")
    
    # 新增
    args.add_argument('--do_train', default=True)
    args.add_argument('--debug', action='store_true')
    args.add_argument('-train_rule_num', '--train_rule_num', default=10000, type=int)
    args.add_argument('-pca', '--pca', default=90, type=int)
    args.add_argument('-d_score', '--detect_score', default=1, type=int, choices=[0, 1])
    args.add_argument('-replace_num', '--replace_num', default=3, type=int, choices=[2, 3])
    
    args = args.parse_args()
    return args


def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, unique_entities_train = build_data(args.data)
    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'))
        logging.info("Initialised relations and entities from TransE")
    else:
        entity_embeddings = np.random.randn(len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(len(relation2id), args.embedding_size)
        logging.info("Initialised relations and entities randomly")
    
    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)
    if args.get_2hop:
        file = args.data + "/2hop.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(corpus.node_neighbors_2hop, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if args.use_2hop:
        logging.info("Opening node_neighbors pickle object")
        file = args.data + "/2hop.pickle"
        with open(file, 'rb') as handle:
            node_neighbors_2hop = pickle.load(handle)
        corpus.node_neighbors_2hop = node_neighbors_2hop
    return corpus, torch.cuda.FloatTensor(entity_embeddings), torch.cuda.FloatTensor(
        relation_embeddings), node_neighbors_2hop


def train_gat(args, model_gat, epochs_gat, save_path, triples, labels):
    # 这里 triples = Corpus_.train_triples
    optimizer = torch.optim.Adam(model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    
    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)
    rule_loss_func = nn.MarginRankingLoss(margin=args.margin_rule)
    
    logging.info("Number of epochs {}".format(epochs_gat))
    
    early_stopping = EarlyStopping(save_path, args.patience_gat)
    
    Corpus_.valid_triples_dict = {j: i for i, j in enumerate(triples + Corpus_.valid_triples + Corpus_.test_triples)}
    
    valid_triples = Corpus_.valid_triples
    valid_indices = np.array(list(valid_triples)).astype(np.int32)
    valid_labels = np.array([[1]] * len(valid_triples)).astype(np.float32)
    file = args.data + "/valid_2hop.pickle"
    if os.path.exists(file):
        with open(file, 'rb') as handle:
            valid_node_neighbors_2hop = pickle.load(handle)
    else:
        graph = Corpus_.get_graph(Corpus_.valid_h_t_r_matrix)
        valid_node_neighbors_2hop = Corpus_.get_further_neighbors(graph)
        with open(file, 'wb') as handle:
            pickle.dump(valid_node_neighbors_2hop, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    for epoch in trange(epochs_gat):
        np.random.shuffle(triples)
        train_indices = np.array(list(triples)).astype(np.int32)
        model_gat.train()  # getting in training mode
        epoch_loss = []
        valid_loss_list = []
        
        if len(triples) % args.batch_size_gat == 0:
            num_iter_per_epoch = len(triples) // args.batch_size_gat
        else:
            num_iter_per_epoch = (len(triples) // args.batch_size_gat) + 1
        for train_iter in range(num_iter_per_epoch):
            train_batch_indices, train_batch_values = Corpus_.get_iteration_batch(train_iter, args.batch_size_gat,
                                                                                  args.valid_invalid_ratio_gat,
                                                                                  train_indices, labels,
                                                                                  args.replace_num)
            # generate 2 hop triples
            # train_indices_numpy = train_indices.cpu().detach().numpy()
            len_pos = len(train_batch_indices) // 3
            if args.replace_num == 3:
                len_pos = len(train_batch_indices) // 4
            adj_indices = torch.LongTensor(train_batch_indices[:len_pos, [0, 2]]).t()  # rows and columns
            adj_values = torch.LongTensor(train_batch_indices[:len_pos, 1])
            train_adj_matrix = (adj_indices, adj_values)
            batch_unique_entities = get_unique_entity(train_batch_indices)
            batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args, batch_unique_entities,
                                                                      Corpus_.node_neighbors_2hop)
            batch_2hop_indices = Variable(torch.LongTensor(batch_2hop_indices)).cuda()
            
            train_batch_indices = Variable(torch.LongTensor(train_batch_indices)).cuda()
            entity_embed, relation_embed = model_gat(train_adj_matrix, train_batch_indices, batch_2hop_indices)
            optimizer.zero_grad()
            if args.rule == "with_rule":
                list2Rules, list3Rules, list2NegRules, list3NegRules = load_train_triple_rule(args, tripleGroundingMap,
                                                                                              train_batch_indices,
                                                                                              len(Corpus_.relation2id))
                loss = batch_gat_loss_modify(args, gat_loss_func, rule_loss_func, train_batch_indices, entity_embed,
                                             relation_embed, list2Rules, list3Rules, list2NegRules, list3NegRules)
            else:
                loss = batch_gat_loss_origin(args, gat_loss_func, train_batch_indices, entity_embed, relation_embed)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())
        scheduler.step()
        
        # Todo 暂时关闭验证集合
        if epoch % 5 == 0:
            with torch.no_grad():
                model_gat.eval()
                
                if len(valid_triples) % args.batch_size_gat == 0:
                    num_iters_valid_epoch = len(valid_triples) // args.batch_size_gat
                else:
                    num_iters_valid_epoch = (len(valid_triples) // args.batch_size_gat) + 1
                for valid_iters in range(num_iters_valid_epoch):
                    valid_batch_indices, valid_batch_values = Corpus_.get_iteration_batch(valid_iters,
                                                                                          args.batch_size_gat,
                                                                                          args.valid_invalid_ratio_gat,
                                                                                          valid_indices, valid_labels,
                                                                                          args.replace_num)
                    valid_batch_unique_entities = get_unique_entity(valid_batch_indices)
                    valid_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args, valid_batch_unique_entities,
                                                                                    valid_node_neighbors_2hop)
                    valid_batch_2hop_indices = Variable(torch.LongTensor(valid_batch_2hop_indices)).cuda()
                    valid_batch_indices = Variable(torch.LongTensor(valid_batch_indices)).cuda()
                    
                    valid_entity_embed, valid_relation_embed = model_gat(Corpus_.valid_h_t_r_matrix,
                                                                         valid_batch_indices, valid_batch_2hop_indices)
                    if args.rule == "with_rule":
                        validList2Rules, validList3Rules, validList2NegRules, validList3NegRules = load_train_triple_rule(
                            args, tripleGroundingMap, valid_batch_indices, len(Corpus_.relation2id))
                        valid_loss = batch_gat_loss_modify(args, gat_loss_func, rule_loss_func, valid_batch_indices,
                                                           valid_entity_embed, valid_relation_embed, validList2Rules,
                                                           validList3Rules, validList2NegRules, validList3NegRules)
                    else:
                        valid_loss = batch_gat_loss_origin(args, gat_loss_func, valid_batch_indices, valid_entity_embed,
                                                           valid_relation_embed)
                    valid_loss_list.append(valid_loss.data.item())
        
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_valid_loss = sum(valid_loss_list) / len(valid_loss_list) if len(valid_loss_list) > 0 else 0
        if epoch % 10 == 0:
            print("gat epoch {:<4d}/{} , average train loss {:<5f} , average valid loss {:<5f}".format(epoch, epochs_gat, avg_loss, avg_valid_loss))
        
        # early_stop
        early_stopping(avg_valid_loss, model_gat)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            break  # 跳出迭代，结束训练


def train_conv(args, model_gat, model_conv, epochs_conv, save_path, triples, labels):
    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)
    
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings
    
    optimizer = torch.optim.Adam(model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch=-1)
    
    margin_loss = torch.nn.SoftMarginLoss()
    
    early_stopping = EarlyStopping(save_path, args.patience_conv)
    
    valid_triples = Corpus_.valid_triples
    valid_indices = np.array(list(valid_triples)).astype(np.int32)
    valid_labels = np.array([[1]] * len(valid_triples)).astype(np.float32)
    
    for epoch in trange(epochs_conv):
        np.random.shuffle(triples)
        train_indices = np.array(list(triples)).astype(np.int32)
        model_conv.train()  # getting in training mode
        epoch_loss = []
        valid_losses = []
        if len(triples) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(triples) // args.batch_size_conv
        else:
            num_iters_per_epoch = (len(triples) // args.batch_size_conv) + 1
        
        for iters in range(num_iters_per_epoch):
            train_batch_indices, train_batch_values = Corpus_.get_iteration_batch(iters, args.batch_size_conv,
                                                                                  args.valid_invalid_ratio_conv,
                                                                                  train_indices, labels,
                                                                                  args.replace_num)
            
            train_batch_indices = Variable(torch.LongTensor(train_batch_indices)).cuda()
            train_batch_values = Variable(torch.FloatTensor(train_batch_values)).cuda()
            
            score_for_triples = model_conv(train_batch_indices)
            optimizer.zero_grad()
            loss = margin_loss(score_for_triples.view(-1), train_batch_values.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.data.item())
        scheduler.step()
        
        # 验证集 进行验证
        if epoch % 5 == 0:
            with torch.no_grad():
                model_gat.eval()
                if len(valid_triples) % args.batch_size_conv == 0:
                    num_iter_valid_epoch = len(valid_triples) // args.batch_size_conv
                else:
                    num_iter_valid_epoch = (len(valid_triples) // args.batch_size_conv) + 1
                for valid_iter in range(num_iter_valid_epoch):
                    valid_batch_indices, valid_batch_values = Corpus_.get_iteration_batch(valid_iter,
                                                                                          args.batch_size_conv,
                                                                                          args.valid_invalid_ratio_conv,
                                                                                          valid_indices, valid_labels,
                                                                                          args.replace_num)
                    
                    valid_batch_indices = Variable(torch.LongTensor(valid_batch_indices)).cuda()
                    valid_batch_values = Variable(torch.FloatTensor(valid_batch_values)).cuda()
                    score_for_valid_triples = model_conv(valid_batch_indices)
                    valid_loss = margin_loss(score_for_valid_triples.view(-1), valid_batch_values.view(-1))
                    valid_losses.append(valid_loss)
        
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_valid_loss = sum(valid_losses) / len(valid_losses) if len(valid_losses) > 0 else 0
        if epoch % 10 == 0:
            print("conv epoch {:<4d}/{} , average train loss {:<5f} , average valid loss {:<5f}".format(epoch, epochs_conv, avg_loss, avg_valid_loss))
        # early_stop
        early_stopping(avg_valid_loss, model_conv)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            # print("Early stopping")
            break  # 跳出迭代，结束训练


def evaluate_conv(args, model_conv, unique_entities):
    model_conv.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(args, model_conv, unique_entities)


def detect_TextCNN(Corpus_, error_rate, model_conv, save_path, is_save=False):
    train_triples = np.array(Corpus_.train_triples)
    valid_triples = np.array(Corpus_.valid_triples)
    test_triples = np.array(Corpus_.test_triples)
    
    train_label = np.ones(len(train_triples))
    valid_label = np.ones(len(valid_triples))
    test_label = np.ones(len(test_triples))
    
    # 构造 train/valid/test 的数据（含 noise ）
    train_noise_triples_path = os.path.join(args.data,
                                            'noise_grounding_rate_{}/train_noise_{}.txt'.format(args.pca, str(50)))
    train_noise_triples = np.loadtxt(train_noise_triples_path, dtype=np.int32)
    
    valid_noise_triples_path = os.path.join(args.data, 'noise_grounding_rate_{}/valid_noise_{}.txt'.format(args.pca,
                                                                                                           str(error_rate)))
    valid_noise_triples = np.loadtxt(valid_noise_triples_path, dtype=np.int32)
    
    test_noise_triples_path = os.path.join(args.data, 'noise_grounding_rate_{}/test_noise_{}.txt'.format(args.pca,
                                                                                                         str(error_rate)))
    test_noise_triples = np.loadtxt(test_noise_triples_path, dtype=np.int32)
    
    train_indexes = train_noise_triples[:, 3]
    train_triples[train_indexes, :] = train_noise_triples[:, 0:3]
    train_label[train_indexes] = np.zeros(len(train_indexes))
    
    valid_indexes = valid_noise_triples[:, 3]
    valid_triples[valid_indexes, :] = valid_noise_triples[:, 0:3]
    valid_label[valid_indexes] = np.zeros(len(valid_indexes))
    
    test_indexes = test_noise_triples[:, 3]
    test_triples[test_indexes, :] = test_noise_triples[:, 0:3]
    test_label[test_indexes] = np.zeros(len(test_indexes))
    
    train_noise_indexes = sorted(train_noise_triples[:, 3])
    valid_noise_indexes = sorted(valid_noise_triples[:, 3])
    test_noise_indexes = sorted(test_noise_triples[:, 3])
    
    all_train_triples = train_triples
    all_valid_triples = valid_triples
    all_test_triples = test_triples
    
    all_train_label = torch.from_numpy(train_label).long().cuda()
    all_valid_label = torch.from_numpy(valid_label).long().cuda()
    all_test_label = torch.from_numpy(test_label).long().cuda()
    
    NUM_INIT_LB = len(all_train_label) // 20 + len(all_train_label) % 20
    NUM_QUERY = len(all_train_label) // 20
    NUM_ROUND = 19
    n_pool = len(all_train_label)
    n_test = len(all_test_label)
    print('number of labeled pool: {}'.format(NUM_INIT_LB))
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
    print('number of testing pool: {}'.format(n_test))
    
    # generate initial labeled pool
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True
    # 数据处理完把 model_conv 放到 gpu 上
    model_conv.eval()
    
    # 构建邻居 triple neighbor 的 2hop map = {neighbor_head, neighbor_tail}
    entity_embedding, relation_embedding = model_conv.get_embedding()
    
    model = TextCNN2D(Corpus_, entity_embedding, relation_embedding, dropout=0.5).cuda()
    print(model)
    strategy = EntropySampling(all_train_triples, all_train_label, all_valid_triples, all_valid_label, idxs_lb, model,
                               save_path, args.detect_score)
    strategy.train(model_conv)
    P = strategy.predict(all_valid_triples, all_valid_label, model_conv).cuda()
    # acc = np.zeros(NUM_ROUND + 1)
    acc = strategy.binary_acc(P, all_valid_label)
    # acc = 1.0 * (all_valid_label == P.cuda()).sum().item() / len(all_valid_label)
    TN_acc, valid_clean_triples, valid_error_triples, log = calculate_TN_acc(valid_triples, all_valid_label, P.cuda(),
                                                                             valid_noise_indexes)
    best_TN_acc = TN_acc
    best_log = log
    print("error rate = {}, round = {}, valid accuracy = {}, log = {}".format(error_rate, 0, acc, log))
    for rd in range(1, NUM_ROUND + 1):
        # query
        q_idxs = strategy.query(NUM_QUERY)
        idxs_lb[q_idxs] = True
        
        # update
        strategy.update(idxs_lb)
        strategy.train(model_conv)
        
        # round accuracy
        P = strategy.predict(all_valid_triples, all_valid_label, model_conv).cuda()
        # acc = 1.0 * (all_valid_label == P.cuda()).sum().item() / len(all_valid_label)
        acc = strategy.binary_acc(P, all_valid_label)
        TN_acc, valid_clean_triples_tmp, error_triples_tmp, log_tmp = calculate_TN_acc(valid_triples, all_valid_label,
                                                                                       P, valid_noise_indexes)
        if log_tmp['TN-acc'] >= best_log['TN-acc'] and log_tmp['FN'] <= best_log['FN']:
            best_TN_acc = TN_acc
            error_triples = error_triples_tmp
            best_log = log_tmp
            torch.save(model.state_dict(), save_path)
        print("error rate = {}, round = {}, valid accuracy = {}, log = {}".format(error_rate, rd, acc, log_tmp))
    
    P_test = strategy.predict(all_test_triples, all_test_label, model_conv).cuda()
    test_acc = strategy.binary_acc(P_test, all_test_label)
    TN_acc, test_clean_triples, test_error_triples, log = calculate_TN_acc(test_triples, all_test_label, P_test,
                                                                           test_noise_indexes)
    logging.info("error rate = {}, test accuracy = {}, test log = {}".format(error_rate, test_acc, log))
    
    if is_save:
        # detect error triples
        test_error_triples_file_path = '{}/{}/{}_test_error_triples_{}.txt'.format(args.detect_out_folder,
                                                                                   str(args.pca), args.rule,
                                                                                   str(error_rate))
        np.savetxt(test_error_triples_file_path, test_error_triples, fmt='%d', delimiter='\t')
        
        # detect clean triples
        test_clean_triples_file_path = '{}/{}/{}_test_clean_triples_{}.txt'.format(args.detect_out_folder,
                                                                                   str(args.pca), args.rule,
                                                                                   str(error_rate))
        np.savetxt(test_clean_triples_file_path, test_clean_triples, fmt='%d', delimiter='\t')
        
        # # 生成分类器分类出的正确的三元组 的 grounding 文件
        # test_grounds_output_path = datasetPath + '/rule/test_groundings_{}_error_rate_{}.txt'.format(args.pca, error_rate)
        # if os.path.exists(test_grounds_output_path):
        #     os.remove(test_grounds_output_path)
        # detect_rule_path = datasetPath + "/rule/{}_rule_{}".format(args.dataset, args.pca)
        # GroundAllRules.loadRules(detect_rule_path, test_grounds_output_path, test_clean_triples, Corpus_.relation2id)


def repair_in_paper(Corpus_, error_rate, model_conv, type='origin'):
    model_conv.eval()
    entity_embedding, relation_embedding = model_conv.get_embedding()
    
    # 数据集构造
    test_triples = [list(triple) for triple in Corpus_.test_triples]
    test_clean_triples = deepcopy(test_triples)
    test_dirty_triples = deepcopy(test_triples)
    
    test_noise_path = os.path.join(args.data,
                                   'noise_grounding_rate_{}/test_noise_{}.txt'.format(args.pca, str(error_rate)))
    test_noise_triples = np.loadtxt(test_noise_path, dtype=np.int32)
    # 真正错误的三元组
    ground_truth_error_triples = [list(x[0:3]) for x in test_noise_triples]
    ground_truth_error_tuple = [tuple(x) for x in ground_truth_error_triples]
    # 脏的测试集的构造
    for x in test_noise_triples:
        index = int(x[3])
        test_dirty_triples[index] = list(x[0:3])
    
    # Detect 部分预测出来的错误三元组
    file_path = '{}/{}/{}_test_error_triples_{}.txt'.format(args.detect_out_folder, str(args.pca), args.rule,
                                                            str(error_rate))
    detect_error_triples = np.loadtxt(file_path, dtype=np.int32).tolist()
    # index_of_detect_error_in_test = []
    # for triple in detect_error_triples:
    #     index = test_dirty_triples.index(triple)
    #     index_of_detect_error_in_test.append(index)
    # ground_truth_repair_triples = [test_clean_triples[x] for x in index_of_detect_error_in_test]
    detect_clean_file_path = '{}/{}/{}_test_clean_triples_{}.txt'.format(args.detect_out_folder, str(args.pca),
                                                                         args.rule, str(error_rate))
    detect_clean_triples = np.loadtxt(detect_clean_file_path, dtype=np.int32).tolist()
    in_triples = defaultdict(lambda: list())
    out_triples = defaultdict(lambda: list())
    # 可通过这个找邻居，不包括预测错误的三元组
    for triple in detect_clean_triples:
        h, r, t = triple
        out_triples[h].append([r, t])
        in_triples[t].append([h, r])
    
    clean_triples = []
    origin_triples = []
    repair_triples = []
    true_error_index = []
    repair_triples_score_map = {}
    for triple_index in trange(len(detect_error_triples)):
        triple = detect_error_triples[triple_index]
        candidate_dataset = CandidateDataset(triple, len(Corpus_.entity2id), len(Corpus_.relation2id), 5000)
        candidate_triples = candidate_dataset.get_triples()
        candidate_loader = DataLoader(candidate_dataset, shuffle=False, batch_size=1, num_workers=0)
        candidate_fr = np.array([])
        for _, candidate_batches in enumerate(candidate_loader):
            candidate_batches = candidate_batches.squeeze()
            candidate_fr_tmp = model_conv.batch_test(candidate_batches).view(-1).cpu().detach().numpy()
            candidate_fr = np.concatenate((candidate_fr, candidate_fr_tmp))
        
        # 应该根据所有的候选集选出排名较高的 k 个三元组
        head_begin = 1
        head_end = head_begin + len(Corpus_.entity2id) - 2
        rela_begin = head_end
        rela_end = rela_begin + len(Corpus_.relation2id) - 1
        tail_begin = rela_end
        
        candidate_head_tensor = torch.FloatTensor(candidate_fr[head_begin:head_end]).cuda()
        candidate_rela_tensor = torch.FloatTensor(candidate_fr[rela_begin:rela_end]).cuda()
        candidate_tail_tensor = torch.FloatTensor(candidate_fr[tail_begin:]).cuda()
        _, candidate_head_top_k_index = candidate_head_tensor.topk(k=10, largest=True)
        _, candidate_rela_top_k_index = candidate_rela_tensor.topk(k=10, largest=True)
        _, candidate_tail_top_k_index = candidate_tail_tensor.topk(k=10, largest=True)
        
        candidate_rela_top_k_index += rela_begin
        candidate_tail_top_k_index += tail_begin
        
        head_top_k_indices = candidate_head_top_k_index.cpu().detach().numpy()
        rela_top_k_indices = candidate_rela_top_k_index.cpu().detach().numpy()
        tail_top_k_indices = candidate_tail_top_k_index.cpu().detach().numpy()
        
        top_k_indices = np.concatenate((head_top_k_indices, rela_top_k_indices, tail_top_k_indices), axis=0)
        top_k_values = np.concatenate(
            (candidate_fr[head_top_k_indices], candidate_fr[rela_top_k_indices], candidate_fr[tail_top_k_indices]),
            axis=0)
        # 将原本的 error_triple 在候选集中的索引也放进去
        candidate_index = 0
        top_k_indices = np.append(top_k_indices, candidate_index)
        top_k_values = np.append(top_k_values, candidate_fr[candidate_index])
        
        top_k_values = torch.FloatTensor(top_k_values).cuda()
        # 将 top_k 的候选三元组划分组
        all_power = torch.zeros(len(candidate_triples))
        IP_score = torch.zeros(len(candidate_triples))
        OP_score = torch.zeros(len(candidate_triples))
        for top_k_index, c_index in enumerate(top_k_indices):
            c_index = int(c_index)
            candidate = candidate_triples[c_index]
            c_h, c_r, c_t = candidate
            
            h_embedding = entity_embedding[[c_h]]
            r_embedding = relation_embedding[[c_r]]
            t_embedding = entity_embedding[[c_t]]
            
            IP_score[c_index] = torch.sigmoid(top_k_values[top_k_index])
            # out_power = probability of its neighbors flow into or out + probability that t and its neighbor nodes co-occur.
            # 先找到 neighbors flow into
            in_score = 0.0
            out_score = 0.0
            # 计算 outer power left
            if c_h in in_triples:
                temp_triples = torch.LongTensor(in_triples[c_h]).cuda()
                
                in_h = temp_triples[:, 0]
                in_r = temp_triples[:, 1]
                
                in_h_embedding = torch.index_select(entity_embedding, 0, in_h)
                in_r_embedding = torch.index_select(relation_embedding, 0, in_r)
                in_r_embedding = in_r_embedding + r_embedding
                in_t_embedding = t_embedding.repeat(len(in_h), 1)
                
                in_score = torch.sigmoid(
                    model_conv.batch_test_emb(in_h_embedding, in_r_embedding, in_t_embedding).view(-1))
                in_score = (1 / len(in_h)) * torch.sum(in_score)
            
            # 计算 outer power right
            if c_t in out_triples:
                temp_triples = torch.LongTensor(out_triples[c_t]).cuda()
                out_r = temp_triples[:, 0]
                out_t = temp_triples[:, 1]
                
                out_h_embedding = h_embedding.repeat(len(out_t), 1)
                out_r_embedding = torch.index_select(relation_embedding, 0, out_r)
                out_r_embedding = r_embedding + out_r_embedding
                out_t_embedding = torch.index_select(entity_embedding, 0, out_t)
                
                out_score = torch.sigmoid(
                    model_conv.batch_test_emb(out_h_embedding, out_r_embedding, out_t_embedding).view(-1))
                out_score = (1 / len(out_t)) * torch.sum(out_score)
            OP_score[c_index] = in_score + out_score
            all_power[c_index] = torch.sigmoid(IP_score[c_index] + OP_score[c_index])
        
        largest_index = torch.argmax(all_power).item()
        
        clean_triple = tuple(candidate_triples[largest_index])
        clean_triples.append(clean_triple)
        if triple in ground_truth_error_triples:
            true_error_index.append(triple_index)
            repair_triples.append(tuple(clean_triple))
            repair_triples_score_map[tuple(candidate_triples[largest_index])] = all_power[largest_index].data.item()
    
    output_file_path = os.path.join(args.data,
                                    'noise_grounding_rate_{}/clean_triples_{}.txt'.format(args.pca, str(error_rate)))
    # np.savetxt(output_file_path, clean_triples, fmt='%d', delimiter='\t')
    # evaluate_repair_modify2(clean_triples, origin_triples, Corpus_.valid_triples_dict, ground_truth_error_triples, error_rate)
    evaluate_repair_modify2(repair_triples, origin_triples, Corpus_.valid_triples_dict, ground_truth_error_triples,
                            error_rate, type)


def repair_with_groundings(Corpus_, error_rate, model_conv, type='origin'):
    model_conv.eval()
    entity_embedding, relation_embedding = model_conv.get_embedding()
    
    # 数据集构造
    test_triples = [list(triple) for triple in Corpus_.test_triples]
    test_clean_triples = deepcopy(test_triples)
    test_dirty_triples = deepcopy(test_triples)
    
    test_noise_path = os.path.join(args.data, 'noise_grounding_rate_{}/test_noise_{}.txt'.format(args.pca, str(error_rate)))
    test_noise_triples = np.loadtxt(test_noise_path, dtype=np.int32)
    # 真正错误的三元组
    ground_truth_error_triples = [list(x[0:3]) for x in test_noise_triples]
    ground_truth_error_tuple = [tuple(x) for x in ground_truth_error_triples]
    # 脏的测试集的构造
    for x in test_noise_triples:
        index = int(x[3])
        test_dirty_triples[index] = list(x[0:3])
    
    # Detect 部分预测出来的错误三元组
    file_path = '{}/{}/{}_test_error_triples_{}.txt'.format(args.detect_out_folder, str(args.pca), args.rule,
                                                            str(error_rate))
    detect_error_triples = np.loadtxt(file_path, dtype=np.int32).tolist()
    index_of_detect_error_in_test = []
    for triple in detect_error_triples:
        index = test_dirty_triples.index(triple)
        index_of_detect_error_in_test.append(index)
    ground_truth_repair_triples = [test_clean_triples[x] for x in index_of_detect_error_in_test]
    
    detect_clean_file_path = '{}/{}/{}_test_clean_triples_{}.txt'.format(args.detect_out_folder, str(args.pca),
                                                                         args.rule, str(error_rate))
    detect_clean_triples = np.loadtxt(detect_clean_file_path, dtype=np.int32).tolist()
    in_triples = defaultdict(lambda: list())
    out_triples = defaultdict(lambda: list())
    # 可通过这个找邻居，不包括预测错误的三元组
    for triple in detect_clean_triples:
        h, r, t = triple
        out_triples[h].append([r, t])
        in_triples[t].append([h, r])
    
    # grounding 相关
    grounding_path = '../dataset/{}/rule/groundings_{}.txt'.format(args.dataset, args.pca)
    test_grounds_output_path = "../../dataset/{}/rule/test_groundings_{}.txt".format(args.dataset, args.pca)
    groundings_set, groundings_map, groundings_row = get_groundings_set([grounding_path, test_grounds_output_path])
    
    clean_triples = []
    origin_triples = []
    repair_triples = []
    true_error_index = []
    repair_triples_score_map = {}
    for triple_index in trange(len(detect_error_triples)):
        triple = detect_error_triples[triple_index]
        candidate_dataset = CandidateDataset(triple, len(Corpus_.entity2id), len(Corpus_.relation2id), 5000)
        candidate_triples = candidate_dataset.get_triples()
        candidate_grounding_index = candidate_dataset.get_candidate_grounding(groundings_map)
        candidate_loader = DataLoader(candidate_dataset, shuffle=False, batch_size=1, num_workers=0)
        candidate_fr = np.array([])
        for _, candidate_batches in enumerate(candidate_loader):
            candidate_batches = candidate_batches.squeeze()
            candidate_fr_tmp = model_conv.batch_test(candidate_batches).view(-1).cpu().detach().numpy()
            candidate_fr = np.concatenate((candidate_fr, candidate_fr_tmp))
        
        # 应该根据所有的候选集选出排名较高的 k 个三元组
        head_begin = 1
        head_end = head_begin + len(Corpus_.entity2id) - 2
        rela_begin = head_end
        rela_end = rela_begin + len(Corpus_.relation2id) - 1
        tail_begin = rela_end
        
        candidate_head_tensor = torch.FloatTensor(candidate_fr[head_begin:head_end]).cuda()
        candidate_rela_tensor = torch.FloatTensor(candidate_fr[rela_begin:rela_end]).cuda()
        candidate_tail_tensor = torch.FloatTensor(candidate_fr[tail_begin:]).cuda()
        _, candidate_head_top_k_index = candidate_head_tensor.topk(k=10, largest=True)
        _, candidate_rela_top_k_index = candidate_rela_tensor.topk(k=10, largest=True)
        _, candidate_tail_top_k_index = candidate_tail_tensor.topk(k=10, largest=True)
        
        candidate_rela_top_k_index += rela_begin
        candidate_tail_top_k_index += tail_begin
        
        head_top_k_indices = candidate_head_top_k_index.cpu().detach().numpy()
        rela_top_k_indices = candidate_rela_top_k_index.cpu().detach().numpy()
        tail_top_k_indices = candidate_tail_top_k_index.cpu().detach().numpy()
        
        top_k_indices = np.concatenate((head_top_k_indices, rela_top_k_indices, tail_top_k_indices), axis=0)
        top_k_values = np.concatenate(
            (candidate_fr[head_top_k_indices], candidate_fr[rela_top_k_indices], candidate_fr[tail_top_k_indices]),
            axis=0)
        # 将原本的 error_triple 在候选集中的索引也放进去
        candidate_index = 0
        top_k_indices = np.append(top_k_indices, candidate_index)
        top_k_values = np.append(top_k_values, candidate_fr[candidate_index])
        
        # 将 候选三元组中 的结论三元组也放进去(取前三个)
        if len(candidate_grounding_index) > 0:
            if len(candidate_grounding_index) > 2:
                _, candidate_ground_top_k_indexs = find_topk(candidate_fr[candidate_grounding_index], 2)
                for i in candidate_ground_top_k_indexs:
                    top_k_indices = np.append(top_k_indices, candidate_grounding_index[i])
                    top_k_values = np.append(top_k_values, candidate_fr[candidate_grounding_index[i]])
        
        top_k_values = torch.FloatTensor(top_k_values).cuda()
        # 将 top_k 的候选三元组划分组
        all_power = torch.zeros(len(candidate_triples))
        IP_score = torch.zeros(len(candidate_triples))
        OP_score = torch.zeros(len(candidate_triples))
        G_score = torch.zeros(len(candidate_triples))
        for top_k_index, c_index in enumerate(top_k_indices):
            c_index = int(c_index)
            candidate = candidate_triples[c_index]
            c_h, c_r, c_t = candidate
            
            h_embedding = entity_embedding[[c_h]]
            r_embedding = relation_embedding[[c_r]]
            t_embedding = entity_embedding[[c_t]]
            
            IP_score[c_index] = torch.sigmoid(top_k_values[top_k_index])
            # out_power = probability of its neighbors flow into or out + probability that t and its neighbor nodes co-occur.
            # 先找到 neighbors flow into
            in_score = 0.0
            out_score = 0.0
            # 计算 outer power left
            if c_h in in_triples:
                temp_triples = torch.LongTensor(in_triples[c_h]).cuda()
                
                in_h = temp_triples[:, 0]
                in_r = temp_triples[:, 1]
                
                in_h_embedding = torch.index_select(entity_embedding, 0, in_h)
                in_r_embedding = torch.index_select(relation_embedding, 0, in_r)
                in_r_embedding = in_r_embedding + r_embedding
                in_t_embedding = t_embedding.repeat(len(in_h), 1)
                
                in_score = torch.sigmoid(
                    model_conv.batch_test_emb(in_h_embedding, in_r_embedding, in_t_embedding).view(-1))
                in_score = (1 / len(in_h)) * torch.sum(in_score)
            
            # 计算 outer power right
            if c_t in out_triples:
                temp_triples = torch.LongTensor(out_triples[c_t]).cuda()
                out_r = temp_triples[:, 0]
                out_t = temp_triples[:, 1]
                
                out_h_embedding = h_embedding.repeat(len(out_t), 1)
                out_r_embedding = torch.index_select(relation_embedding, 0, out_r)
                out_r_embedding = r_embedding + out_r_embedding
                out_t_embedding = torch.index_select(entity_embedding, 0, out_t)
                
                out_score = torch.sigmoid(
                    model_conv.batch_test_emb(out_h_embedding, out_r_embedding, out_t_embedding).view(-1))
                out_score = (1 / len(out_t)) * torch.sum(out_score)
            OP_score[c_index] = in_score + out_score
            
            # 加 groundings
            # candidate 相关的所有 groundings
            groundings_candidate = list()
            a = (c_h, c_r, c_t)
            ground_score = 0.0
            count = 1
            if a in groundings_map.keys():
                g_index_set = groundings_map[a]
                for i in g_index_set:
                    groundings_candidate.append(groundings_row[i])
                if len(groundings_candidate) > 0:
                    for g_candidate in groundings_candidate:
                        if g_candidate[0] == '2':
                            triple_1 = [int(g_candidate[1]), int(g_candidate[2]), int(g_candidate[3])]
                            # triple_2 = [int(g_candidate[4]), int(g_candidate[5]), int(g_candidate[6])]
                            confidence = float(g_candidate[7]) / 100000
                            if triple_1 in test_dirty_triples:
                                count += 1
                                ground_score += torch.sigmoid(top_k_values[top_k_index]) * confidence
                        elif g_candidate[0] == '3':
                            triple_1 = [int(g_candidate[1]), int(g_candidate[2]), int(g_candidate[3])]
                            triple_2 = [int(g_candidate[4]), int(g_candidate[5]), int(g_candidate[6])]
                            # triple_3 = [int(g_candidate[7]), int(g_candidate[8]), int(g_candidate[9])]
                            confidence = float(g_candidate[10]) / 100000
                            extra_score = torch.tensor([0.0], dtype=torch.float32).cuda()
                            if int(g_candidate[1]) == int(g_candidate[6]):
                                c_h_embedding = entity_embedding[int(g_candidate[4])]
                                c_r_embedding = relation_embedding[int(g_candidate[2])] + relation_embedding[
                                    int(g_candidate[5])]
                                c_t_embedding = entity_embedding[int(g_candidate[3])]
                                extra_score = model_conv.batch_test_emb(c_h_embedding.unsqueeze(0),
                                                                        c_r_embedding.unsqueeze(0),
                                                                        c_t_embedding.unsqueeze(0)).view(-1)
                            if int(g_candidate[3]) == int(g_candidate[4]):
                                c_h_embedding = entity_embedding[int(g_candidate[1])]
                                c_r_embedding = relation_embedding[int(g_candidate[2])] + relation_embedding[
                                    int(g_candidate[5])]
                                c_t_embedding = entity_embedding[int(g_candidate[6])]
                                extra_score = model_conv.batch_test_emb(c_h_embedding.unsqueeze(0),
                                                                        c_r_embedding.unsqueeze(0),
                                                                        c_t_embedding.unsqueeze(0)).view(-1)
                            if triple_1 in test_dirty_triples and triple_2 in test_dirty_triples:
                                count += 1
                                a = torch.add(top_k_values[top_k_index], extra_score.squeeze(-1))
                                ground_score += torch.sigmoid(a) * confidence
                    G_score[c_index] = ground_score / count
            all_power[c_index] = torch.sigmoid(IP_score[c_index] + OP_score[c_index] + G_score[c_index])
        
        largest_index = torch.argmax(all_power).item()
        
        clean_triple = tuple(candidate_triples[largest_index])
        clean_triples.append(clean_triple)
        ground_truth_repair_triple = ground_truth_repair_triples[triple_index]
        if triple in ground_truth_error_triples:
            true_error_index.append(triple_index)
            origin_triples.append(ground_truth_repair_triples[triple_index])
            repair_triples.append(tuple(clean_triple))
            repair_triples_score_map[tuple(candidate_triples[largest_index])] = all_power[largest_index].data.item()
    
    output_file_path = os.path.join(args.data,
                                    'noise_grounding_rate_{}/clean_triples_{}.txt'.format(args.pca, str(error_rate)))
    # np.savetxt(output_file_path, clean_triples, fmt='%d', delimiter='\t')
    evaluate_repair_modify2(repair_triples, origin_triples, Corpus_.valid_triples_dict, ground_truth_error_triples,
                            error_rate, type)


def calculate_unlabeled_score(Corpus_, unlabeled_triples, model_conv):
    unlabeled_triples = np.array(unlabeled_triples)
    in_triples = defaultdict(lambda: list())
    out_triples = defaultdict(lambda: list())
    # 构造邻居字典
    for triple in Corpus_.train_triples:
        h, r, t = triple
        in_triples[t].append([h, r])
        out_triples[h].append([r, t])
    
    # grounding 相关
    grounding_path = '../dataset/{}/rule/groundings_{}.txt'.format(args.dataset, args.pca)
    groundings_set, groundings_map, groundings_row = get_groundings_set([grounding_path])
    
    start_time = time.time()
    unlabeled_triples_power = calculate_inner_outer_grounding_score(model_conv, unlabeled_triples, groundings_map,
                                                                    groundings_row, in_triples, out_triples)
    # unlabeled_triples_power = calculate_inner_outer_score(model_conv, unlabeled_triples, in_triples, out_triples)
    end_time = time.time()
    logging.info("calculate_inner_outer_grounding_score use time = {}".format(end_time - start_time))
    
    # 取大于 0.5 的作为有效 unlabeled_triple
    zeros = torch.zeros_like(unlabeled_triples_power, dtype=torch.int32)
    ones = torch.ones_like(unlabeled_triples_power, dtype=torch.int32)
    unlabeled_triples_power_filter = torch.where(unlabeled_triples_power > 0.5, ones, zeros).cpu().detach().numpy()
    unlabeled_triples_power_valid_index = np.where(unlabeled_triples_power_filter > 0)
    valid_index = unlabeled_triples_power_valid_index[0].tolist()
    valid_triples = unlabeled_triples[valid_index, :]
    unlabeled_triples_valid_tuple = [tuple(x) for x in valid_triples]
    unlabeled_triples_valid_soft_score = torch.index_select(unlabeled_triples_power, 0, torch.tensor(valid_index))
    return unlabeled_triples_valid_tuple, unlabeled_triples_valid_soft_score


def re_embedding_soft_function(model_conv, batch_size_gat, lr):
    args.epochs_gat = 3600
    args.epochs_conv = 250
    args.batch_size_gat = batch_size_gat
    args.lr = lr
    
    if args.rule == 'with_rule':
        unlabeled_triples_valid_tuple, unlabeled_triples_valid_soft_score = calculate_unlabeled_score(Corpus_, unlabeled_triples, model_conv)
        logging.info("\nunlabeled triples number = {}\n".format(len(unlabeled_triples_valid_tuple)))
        
        train_triples = Corpus_.train_triples + unlabeled_triples_valid_tuple
        train_labels = np.array([[1]] * len(Corpus_.train_triples)).astype(np.float32)
        unlabeled_numpy = unlabeled_triples_valid_soft_score.unsqueeze(1).cpu().detach().numpy()
        unlabeled_labels_for_detect = unlabeled_triples_valid_soft_score.cpu().detach().numpy()
        train_labels = np.concatenate([train_labels, unlabeled_numpy], axis=0)
        
        model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim,
                                    args.entity_out_dim,
                                    args.drop_GAT, args.alpha, args.nheads_GAT)
        model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim,
                                     args.entity_out_dim,
                                     args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                     args.nheads_GAT,
                                     args.out_channels)
        model_gat = model_gat.cuda()
        model_conv = model_conv.cuda()

        dir_path = '{}/{}/'.format(args.embed_out_folder, args.pca)
        name_path = 'seed_{}_{}_{}_pca_{}_after_detect.pth'.format(args.seed, 're_trained_gat', args.rule, args.pca)
        model_soft_gat_path = dir_path + name_path
        train_gat(args, model_gat, args.epochs_gat, model_soft_gat_path, train_triples, train_labels)
        
        model_gat.load_state_dict(torch.load(model_soft_gat_path))
        model_gat = model_gat.cuda()

        name_path = 'seed_{}_{}_{}_pca_{}_after_detect.pth'.format(args.seed, 're_trained_conv', args.rule, args.pca)
        model_soft_conv_path = dir_path + name_path
        train_conv(args, model_gat, model_conv, args.epochs_conv, model_soft_conv_path, train_triples, train_labels)


def re_embedding_after_detect_TP_function(batch_size_gat, lr):
    """
    重新将 Detect 里面的 TP 部分训练训练模型
    """
    args.epochs_gat = 1000
    args.epochs_conv = 250
    args.batch_size_gat = batch_size_gat
    args.lr = lr
    
    if args.rule == 'with_rule':
        # detect clean triples
        test_clean_triples_file_path = '{}/{}/{}_test_clean_triples_{}.txt'.format(args.detect_out_folder, str(args.pca), args.rule, str(e_r))
        test_clean_triples = np.loadtxt(test_clean_triples_file_path)
        detect_clean_triples = [tuple(x) for x in test_clean_triples]
        re_train_triples = Corpus_.train_triples + detect_clean_triples
        re_train_labels = np.array([[1]] * len(re_train_triples)).astype(np.float32)
        
        model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim,
                                    args.entity_out_dim,
                                    args.drop_GAT, args.alpha, args.nheads_GAT)
        model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim,
                                     args.entity_out_dim,
                                     args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                     args.nheads_GAT,
                                     args.out_channels)
        model_gat = model_gat.cuda()
        model_conv = model_conv.cuda()
        dir_path = '{}/{}/'.format(args.embed_out_folder, args.pca)
        name_path = 'seed_{}_{}_{}_pca_{}_after_detect.pth'.format(args.seed, 're_trained_gat', args.rule, args.pca)
        model_soft_gat_path = dir_path + name_path
        train_gat(args, model_gat, args.epochs_gat, model_soft_gat_path, re_train_triples, re_train_labels)
        
        model_gat.load_state_dict(torch.load(model_soft_gat_path))
        model_gat = model_gat.cuda()
        name_path = 'seed_{}_{}_{}_pca_{}_after_detect.pth'.format(args.seed, 're_trained_conv', args.rule, args.pca)
        model_soft_conv_path = dir_path + name_path
        train_conv(args, model_gat, model_conv, args.epochs_conv, model_soft_conv_path, re_train_triples, re_train_labels)

if __name__ == '__main__':
    
    # Todo
    # KGClean(no rule)
    # Indefinable_rule              :   embedding + rule(joint embedding) + detect + repair(inter score + outer score)
    # Indefinable_rule_repair       :   embedding + rule - -repair(inter score + outer score - infer soft label, re - embedding)
    # Indefinable_repair            :   embedding --repair + rules(optimization function + soft rules - infer soft label, re - embedding)
    # Indefinable_rule_repair_rule  :   embedding + rule --repair + rules(optimization function + soft rules - infer soft label, re - embedding) (our method)
    
    dataset = 'wn18'
    global tripleGroundingMap, unlabeled_triples
    # experiment_name = [KGClean, test, final]
    experiment_name = "Indefinable_rule_repair"
    
    torch.cuda.set_device(0)
    # for pca in [90, 70, 50]:
    # 获得 dataset 文件夹的绝对路径
    # global datasetPath
    datasetPath = "/home/vision/work/xuhong/TransGAT-iteration/dataset/" + dataset
    
    # use_rule = ["with_rule", "without_rule"]
    
    args = parse_args(dataset, experiment_name, seed=37, use_rule="with_rule")  # 各类文件夹
    
    args = load_config(args)
    args.pca = 0.5
    args.replace_num = 3
    set_logger(args)
    
    embed_out_folder_pca = os.path.join(args.embed_out_folder, str(args.pca))
    if embed_out_folder_pca and not os.path.exists(embed_out_folder_pca):
        os.makedirs(embed_out_folder_pca)
    detect_out_folder_pca = os.path.join(args.detect_out_folder, str(args.pca))
    if detect_out_folder_pca and not os.path.exists(detect_out_folder_pca):
        os.makedirs(detect_out_folder_pca)
    repair_out_folder_pca = os.path.join(args.repair_out_folder, str(args.pca))
    if repair_out_folder_pca and not os.path.exists(repair_out_folder_pca):
        os.makedirs(repair_out_folder_pca)
    
    Corpus_, entity_embeddings, relation_embeddings, node_neighbors_2hop = load_data(args)
    logging.info("Initial entity dimensions {} , relation dimensions {}".format(entity_embeddings.size(),
                                                                                relation_embeddings.size()))
    logging.info("rule = {}, replace_num = {}, epoch_gat = {}, epoch_conv = {}， pca = {}".format(args.rule, args.replace_num,
                                                                                                 args.epochs_gat,
                                                                                                 args.epochs_conv, args.pca))
    if args.rule == 'with_rule':
        groundingsPath = datasetPath + '/rule/groundings_{}.txt'.format(args.pca)
        print("groundingsPath = {}".format(groundingsPath))
        if not os.path.exists(groundingsPath):
            rule_path = datasetPath + "/rule/" + dataset + "_rule_" + str(args.pca)
            groundsOutputPath = datasetPath + "/rule/groundings_" + str(args.pca) + ".txt"
            GroundAllRules.loadRules(rule_path, groundsOutputPath, Corpus_.train_triples, Corpus_.relation2id)
        GroundRules = RuleSet(len(Corpus_.entity2id), len(Corpus_.relation2id))
        GroundRules.load(groundingsPath)
        tripleGroundingMap = GroundRules.tripleGroundingMap
        unlabeled_triples = GroundRules.unlabeled_triples
    
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT)
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv, args.nheads_GAT,
                                 args.out_channels)
    model_gat = model_gat.cuda()
    model_conv = model_conv.cuda()
    
    train_triples = Corpus_.train_triples
    train_labels = np.array([[1]] * len(Corpus_.train_triples)).astype(np.float32)
    model_gat_path = '{}/{}/seed_{}_{}_{}_pca_{}.pth'.format(args.embed_out_folder, args.pca, args.seed, 'trained_gat', args.rule, args.pca)
    train_gat(args, model_gat, args.epochs_gat, model_gat_path, train_triples, train_labels)
    
    model_gat.load_state_dict(torch.load(model_gat_path))
    model_gat = model_gat.cuda()
    
    model_conv_path = '{}/{}/seed_{}_{}_{}_pca_{}.pth'.format(args.embed_out_folder, args.pca, args.seed, 'trained_conv', args.rule, args.pca)
    train_conv(args, model_gat, model_conv, args.epochs_conv, model_conv_path, train_triples, train_labels)
    
    model_conv.load_state_dict(torch.load(model_conv_path))
    model_conv = model_conv.cuda()
    evaluate_conv(args, model_conv, Corpus_.unique_entities_train)
    
    # for e_r in [30]:
    #     logging.info("\n")
    #     dir_path = '{}/{}/'.format(args.detect_out_folder, str(args.pca))
    #     name_path = '{}_{}_seed_{}_pca_{}_TextCNN_{}.ckpt'.format('D1', args.rule, args.seed, args.pca, e_r)
    #     save_path = dir_path + name_path
    #     detect_TextCNN(Corpus_, e_r, model_conv, save_path, True)
    #     repair_in_paper(Corpus_, e_r, model_conv, 'origin')
        
        
    # 基础的
    # Indefinable_rule  :   embedding + rule + detect + repair(inter score + outer score)
    if args.rule == 'with_rule' and experiment_name == "Indefinable_rule":
        pass
    
    # 需要 re-embedding + 利用 detect 里面的 TP
    if args.rule == 'with_rule' and experiment_name == "Indefinable_rule_repair":
        re_embedding_after_detect_TP_function(batch_size_gat=1204, lr=1e-4)
        
    # 需要 infer soft label 来 re-embedding
    if args.rule == 'with_rule' and experiment_name == "Indefinable_rule_repair":
        # Indefinable_rule_repair       :   embedding + rule - detect -repair(inter score + outer score) - infer soft label, re - embedding,  detect, repair
        
        re_embedding_soft_function(model_conv, batch_size_gat=1204, lr=1e-4)
        
        for e_r in [30]:
            logging.info("\n")
            dir_path = '{}/{}/'.format(args.detect_out_folder, str(args.pca))
            name_path = '{}_{}_seed_{}_pca_{}_TextCNN_{}.ckpt'.format('D2', args.rule, args.seed, args.pca, e_r)
            save_path = dir_path + name_path
            detect_TextCNN(Corpus_, e_r, model_conv, save_path, True)
            repair_in_paper(Corpus_, e_r, model_conv, 'soft')
            repair_with_groundings(Corpus_, e_r, model_conv, 'soft grounding')
    
    # Indefinable_repair            :   embedding --repair + rules(optimization function + soft rules - infer soft label, re - embedding)
    # Indefinable_rule_repair_rule  :   embedding + rule --repair + rules(optimization function + soft rules - infer soft label, re - embedding) (our method)












