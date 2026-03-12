# -*- coding: utf-8 -*-
"""
Created on Thu May 29 21:13:57 2025

@author: raphg
"""

import numpy as np
print(np.__version__)
import random
import math
import hashlib

import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F


from tqdm import trange

#plateau inital

INITIAL_BOARD = np.asarray(
    [
     [-8,-12,-10,-11,-9],
     [-7, -4, -1, -6,-5],
     [-3,  2,  0, -2, 3],
     [ 5,  6,  1,  4, 7],
     [ 9,  11, 10, 12,8]
     ]
    )

#Quelle horreur mais bon tant pis on va faire avec
#positif : soit même, négatif = adversaire. Logique

#matrice qui sert à savoir qui est plus grand que quoi à tout moment
#c'est pour les coups valides et c'est plus simple de check là dedans plutot que de faire des if à l'infini
#elle fait 25*25 car on va de -12 à 12 en passant par 0

#pour chaque liste numéro i, on a toutes les pièces que peux attaquer i
#ATTENTION : i est la valeur absolue de la pièce mdr

VALID_MOVE_MATRIX = [
     #0
     [],
     #1
     [10,11,12],
     #2
     [0,1,2],
     #3
     [0,1,2,3],
     #4
     [0,1,2,3,4],
     #5
     [0,1,2,3,4,5],
     #6
     [0,1,2,3,4,5,6],
     #7
     [0,1,2,3,4,5,6,7],
     #8
     [0,1,2,3,4,5,6,7,8],
     #9
     [0,1,2,3,4,5,6,7,8,9],
     #10 (Valet)
     [2,3,4,5,6,7,8,9,10],
     #11 (Dame)
     [2,3,4,5,6,7,8,9,10,11],
     #12 (Roi)
     [2,3,4,5,6,7,8,9,10,11,12]
     ]


MAX_NUMBER_OF_MOVES = 50 #plus de 50 coups et c'est fini et Draw
MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED = 3 #si on refait le même état 3 fois dans la partie c'est finis et Draw

def flatten_and_sum_list_of_list(list_of_list):
    flat_list = []
    for move in list_of_list:
        for element in move:
            flat_list.append(element)
    return np.sum(flat_list)

class Chenapan:
    def __init__(self):
        self.row_count = 5
        self.column_count = 5
        self.action_size = 25 #il y a 25 pièces en tout qui peuvent aller dans 25 autres position au grand maximum
        self.number_of_moves = 0
        self.biggest_loop = 0 #sur tous les états explorés, on va garder en mémoire le nombre d'état identique dans la liste des états visités
        self.list_of_positions = [] #liste de hash unique pour chaque position : permet de vérifier efficacement si on a déjà rencontré une position
      
    def reset(self):
        self.number_of_moves = 0
        self.biggest_loop = 0
        self.list_of_positions = []
    
    def get_hash(self,state):
        s = np.ascontiguousarray(state)
        return hashlib.md5(s.tobytes()).hexdigest()
        
    def get_initial_state(self):
        return np.array(INITIAL_BOARD)
    #ça devrait renvoyer une deep copy selon un random sur stackoverflow
    
    def update_biggest_loop(self,h):
        #h is "hash"
        #c is "count"
        c = self.list_of_positions.count(h)
        if c > self.biggest_loop:
            self.biggest_loop = c
        
    
    def swap(self,state,start_row, start_column, end_row, end_column):
        state[start_row,start_column], state[end_row,end_column] = state[end_row,end_column], state[start_row,start_column]
    
    def get_next_state(self, state, action, update_meta_parameters = True):
        
        #action = liste de 2 nombres entre 0 et 24 : le départ et l'arrivée
        start_row = action[0] // self.row_count
        start_column = action[0] % self.column_count
        end_row = action[1] // self.row_count
        end_column = action[1] % self.column_count
        
        self.swap(state,start_row,start_column,end_row,end_column)
        
        if(update_meta_parameters):
            #memory update
            h = self.get_hash(state)
            self.list_of_positions.append(h)
            self.number_of_moves += 1
            self.update_biggest_loop(h)
        #alors c'est compliqué : On ne peut pas tout le temps augmenter ces valeurs
        #en particulier dans les phases expand et simulation du MCTS
        
        #en effet, expand va trouver tous les coups pour un joueur et utiliser cette méthode pour ça
        #ce qui est problématique puisque l'arbre s'élargit et de s'approfondit pas
        #donc il faut mettre False à ce moment là sur l'update de ces valeur
        
        #concernant simulation, et le rollout, il faut que ça puisse update
        #puisqu'on va très profond dans l'arbre et faut bien que ça s'arrête un jour où l'autre
        #mais c'est géré en gardant la valeur d'avant le rollout en tampon
        
        
        return state
    
    def get_valid_moves(self,state,player):
        #pour chaque putain de pièce sur le plateau, on va faire la liste des actions possibles t_t
        #on va faire une liste de liste
        #l'index de la liste, c'est la case de départ (entre 0 et 24) et la liste contient des nombres entre 0 et 24 (case d'arrivée)
        #car chaque pièce à des arrivées différente et un move particulier
        
        valid_moves = []
        
        for i in range(0,self.row_count):
            for j in range(0,self.column_count):
                if state[i,j] == 0 or state[i,j]*player < 0:
                    #si c'est le joker ou du signe opposé on ignore
                    valid_moves.append([])
                
                elif abs(state[i,j]) == 1 :
                    moves = self.check_ace_moves(state,i,j)
                    valid_moves.append(moves)
                
                elif abs(state[i,j]) == 10 :
                    moves = self.check_jack_moves(state,i,j)
                    valid_moves.append(moves)
                
                elif abs(state[i,j]) == 11 :
                    moves = self.check_queen_moves(state,i,j)
                    valid_moves.append(moves)
                
                elif abs(state[i,j]) == 12 :
                    moves = self.check_king_moves(state,i,j)
                    valid_moves.append(moves)
                    
                else:
                    moves = self.check_basic_moves(state,i,j)
                    valid_moves.append(moves)
    
                
        return valid_moves
    
    def get_matrix_of_valid_moves(self,valid_moves):
        res = np.zeros((25,25))
        
        for i in range(len(valid_moves)):
            for j in valid_moves[i]:
                res[i][j] = 1
        
        return res
                
    
    def check_ace_moves(self,state,row,column):
        moves = []
        #peut se déplacer sur toutes les diagonales et toutes les horizontales
        #donc au max de 4 a gauche, à droite, haut, bas, nord-ouest,NE,SE,SO
        #go boucle for parceque nique sa mère en fait
        
        for i in range(1,5):
            
            if row-i >= 0:
                if abs(state[row-i,column]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append((row-i)*5 + column)
            if row+i <= 4:
                if abs(state[row+i,column]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append((row+i)*5 + column)
            if column-i >= 0:
                if abs(state[row,column-i]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append(row*5 + (column-i))
            if column+i <= 4:
                if abs(state[row,column+i]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append(row*5 + (column+i))
            if row-i >= 0 and column-i >= 0:
                if abs(state[row-i,column-i]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append((row-i)*5 + (column-i))
            if row+i <= 4 and column+i <= 4:
                if abs(state[row+i,column+i]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append((row+i)*5 + (column+i))
            if row-i >= 0 and column+i <= 4:
                if abs(state[row-i,column+i]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append((row-i)*5 + (column+i))
            if row+i <= 4 and column-i >= 0:
                if abs(state[row+i,column-i]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append((row+i)*5 + (column-i))
          
        return moves
    
    def check_jack_moves(self,state,row,column):
        moves = []
        
        if row-2 >= 0 and column-1 >= 0:
            if abs(state[row-2,column-1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row-2)*5 + (column-1))
        if row+2 <= 4 and column+1 <= 4:
            if abs(state[row+2,column+1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row+2)*5 + (column+1))
        if row-2 >= 0 and column+1 <= 4:
            if abs(state[row-2,column+1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row-2)*5 + (column+1))
        if row+2 <= 4 and column-1 >= 0:
            if abs(state[row+2,column-1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row+2)*5 + (column-1))
        if row-1 >= 0 and column-2 >= 0:
            if abs(state[row-1,column-2]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row-1)*5 + (column-2))
        if row+1 <= 4 and column+2 <= 4:
            if abs(state[row+1,column+2]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row+1)*5 + (column+2))
        if row-1 >= 0 and column+2 <= 4:
            if abs(state[row-1,column+2]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row-1)*5 + (column+2))
        if row+1 <= 4 and column-2 >= 0:
            if abs(state[row+1,column-2]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row+1)*5 + (column-2))
        
        return moves
    
    def check_queen_moves(self,state,row,column):
        moves = []
        
         
        for i in range(1,5):
            if row-i >= 0:
                if abs(state[row-i,column]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append((row-i)*5 + column)
            if row+i <= 4:
                if abs(state[row+i,column]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append((row+i)*5 + column)
            if column-i >= 0:
                if abs(state[row,column-i]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append(row*5 + (column-i))
            if column+i <= 4:
                if abs(state[row,column+i]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                    moves.append(row*5 + (column+i))
          
        
        return moves
    
    def check_king_moves(self,state,row,column):
        moves = []
        
        
        if row-1 >= 0:
            if abs(state[row-1,column]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row-1)*5 + column)
        if row+1 <= 4:
            if abs(state[row+1,column]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row+1)*5 + column)
        if column-1 >= 0:
            if abs(state[row,column-1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append(row*5 + (column-1))
        if column+1 <= 4:
            if abs(state[row,column+1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append(row*5 + (column+1))
        if row-1 >= 0 and column-1 >= 0:
            if abs(state[row-1,column-1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row-1)*5 + (column-1))
        if row+1 <= 4 and column+1 <= 4:
            if abs(state[row+1,column+1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row+1)*5 + (column+1))
        if row-1 >= 0 and column+1 <= 4:
            if abs(state[row-1,column+1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row-1)*5 + (column+1))
        if row+1 <= 4 and column-1 >= 0:
            if abs(state[row+1,column-1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row+1)*5 + (column-1))
              
                
        return moves
    
    def check_basic_moves(self,state,row,column):
        
        moves = []
        
        if row-1 >= 0:
        #si on ne regarde pas hors du plateau
            if abs(state[row-1,column]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row-1)*5 + column)
        #Si la valeur absolu de la destination est dans la liste correspondant à la valeur de départ
        #Alors cela veut dire que la pièce de départ peut se déplacer à cette position
        if row+1 <= 4:
            if abs(state[row+1,column]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append((row+1)*5 + column)
        if column-1 >= 0:
            if abs(state[row,column-1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append(row*5 + (column-1))
        if column+1 <= 4:
            if abs(state[row,column+1]) in VALID_MOVE_MATRIX[abs(state[row,column])]:
                moves.append(row*5 + (column+1))
        
                
        return moves
        
    
    def check_win(self, state, action):
        #Il faut vérifier qu'une action est gagnante sans modifier l'état
        #une action est gagnante si elle amène le 0 en première ligne pour le joueur +1
        # et en dernière ligne pour le joueur -1
        #une action est une paire de nombre
        #le premier correspond à la position de départ : on peut retrouver la valeur de la pièce à partir de là
        #le second la valeur d'arrivée : on peut aussi trouver la valeur d'arrivée
        #donc si la valeur d'arrivée d'un move fait par le joueur +1 est 0, et qu'il part de la première ligne, c'est gagné pour lui
        #et de même si c'est la dernière ligne pour le joueur -1
        #au final on s'en fout de la valeur de départ, seulement de sa position (forcément le premier ou dernier rang)
        
        #pour savoir le joueur qui joue, il suffit de choper non pas la valeur de départ mais son signe
        
        if action is None:
            return False,1 #c'est pour le début du jeu et l'arbre de recherche, cf MTCS.search(), forcément le joueur 1 qui joue dans ce cas
        
        start_row = action[0] // self.row_count
        start_column = action[0] % self.column_count
        end_row = action[1] // self.row_count
        end_column = action[1] % self.column_count
        
        player = np.sign(state[start_row,start_column])
        
        if(state[end_row,end_column] == 0):
            if(start_row == 0):
                return 1 == player,player
            if(start_row == 4):
                return -1 == player,player
        return False,player
    
    def get_value_and_terminated(self,state,action):
        
        win,player = self.check_win(state,action)
        
        if win: #si c'est un win, quelque soit le vainqueur
            return 1, True
        
        if flatten_and_sum_list_of_list(self.get_valid_moves(state, player)) == 0 or self.biggest_loop >= MAX_NUMBER_OF_TIME_STATE_CAN_BE_VISITED or self.number_of_moves >= MAX_NUMBER_OF_MOVES :
            #ou il n'y a pas de move disponible
            #ou on a dépassé le nombre de move possible en une partie
            #ou on a visité 3 fois un état (boucle)
            return -1, True

        return 0, False #on continue le cas échéant
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value): #la valeur d'un node dans l'arbre de recherche dépend du pdv
        return -value
    
    def change_perspective(self, state):
        #il faut retourner le plateau de 180 degré et tout multiplier par -1 pour inverser (ou non) la valeur des pieces
        #si on veut vraiment la perspective du joueur, qui cherche à faire avancer le joker
        return np.rot90(state,2) * -1
    
    def get_encoded_state(self,state):
        #On sépare le plateau en 3 couches : une couche ennemis, la couche avec le joker, et la couche du joueur
        #et on transforme le tout en float pour que l'IA les digère correctement
        
        #askip, alpha zero encodait aussi des layers en plus avec simplement une seule valeur comme la couleur
        #donc on va faire pareil avec les valeurs biggest loop et number_of_moves parceque lol
        
        adverse_layer = np.zeros(np.shape(state))
        joker_layer = np.zeros(np.shape(state))
        player_layer = np.zeros(np.shape(state))
        for i in range(self.row_count):
            for j in range(self.column_count):
                if state[i][j] < 0:
                    adverse_layer[i][j] = -state[i][j]
                elif state[i][j] > 0:
                    player_layer[i][j] = state[i][j]
                else :
                    joker_layer[i][j] = 1
                    
        number_of_moves_layer = np.ones(np.shape(state))*self.number_of_moves
        biggest_loop_layer = np.ones(np.shape(state))*self.biggest_loop
        arrays = [number_of_moves_layer,biggest_loop_layer,adverse_layer,joker_layer,player_layer]
        
        return np.stack(arrays).astype(np.float32)
    
class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual          # skip connection — same channels in and out
        return F.relu(out)


class AlphaPanNet(nn.Module):
    def __init__(self, device, num_res_blocks=4, num_hidden=64):
        super().__init__()
        self.device = device

        # Initial projection: 5 input channels -> num_hidden
        self.start_block = nn.Sequential(
            nn.Conv2d(5, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_hidden) for _ in range(num_res_blocks)]
        )

        # Value head: num_hidden channels -> scalar in [-1, 1]
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # Policy head: num_hidden channels -> 25x25 logits
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 25 * 25)
        )

        self.to(device)

    def forward(self, x):
        x = self.start_block(x)
        for block in self.res_blocks:
            x = block(x)
        policy = self.policy_head(x).view(-1, 25, 25)
        value = self.value_head(x)
        return policy, value

class Node:
    def __init__(self, game, args, state, player = 1, parent=None, action_taken=None,prior=0,visit_count=0):
        #J'ai ajouté player = 1 pour game.get_valid_moves qui en a besoin mais vu qu'on est toujours du point
        #de vue du joueur c'est pas très grave, je le laisse quand même pour l'instant
        self.game = game
        self.args = args
        self.state = state
        self.player = player
        self.parent = parent
        self.action_taken = action_taken
        #alphapan
        self.prior = prior
        
        self.children = []
        #classic mcts : avec alphapan on expand dans toutes les directions
        #
        #self.expandable_moves = game.get_valid_moves(state, player) #list of 25 lists
        #it's gonna be modified aka emptied little by little
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        
        #étant donné que les moves sont dans une list de list
        #on ne peux pas simplement additionner les moves dispo pour vérifier qu'il y en a restant ou pas
        #on va donc créer une méthode générale qui permet ça sur nimporte quelle liste de liste
        
        #number_of_expandable_moves_left = flatten_and_sum_list_of_list(self.expandable_moves)
        
        #return number_of_expandable_moves_left == 0 and len(self.children) > 0
        
        #au dessus : classic mcts, puisque qu'on expand tous les moves disponibles systématiquement
        #on en a pas besoin, si il y a au moins un enfant alors il y a forcément tous les autres
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt((self.visit_count) / (1+child.visit_count) ) * child.prior
    
    def select_random_action(self,list_of_list_of_moves):
        
        #une action est une paire de nombre
        #elles sont stockées dans une list de list de 25 elements
        #chaque element étant une liste vide ou avec des trucs dedans
        #on veut choisir au hasard parmis les actions possibles et en extraire l'indice (ça sera l'action_start)
        #puis prendre une valeur au hasard dedans qui donnera l'action d'arrivé (ça sera l'action_end)
        #et action = [action_start,action_end]
        
        #Je veux la liste des indices des actions dispo dans expandable_moves
        # On crée une liste des indices des sous-listes non vides :
        # - enumerate(liste) donne à la fois l'indice i et la sous-liste
        # - if sous_liste vérifie que la sous-liste n'est pas vide (car [] est False)
        # - on garde l'indice i des sous-listes non vides

        list_of_index_of_possible_actions = [index for index,list_of_moves in  enumerate(list_of_list_of_moves) if list_of_moves]
        
        #NORMALEMENT YA TOUJOURS DES MOVES DE DISPO
        
        #je veux selectionner un indice au hasard parmis eux
        action_start = np.random.choice(list_of_index_of_possible_actions)
        
        #je veux selectionner un nombre au hasard dans ces listes
        action_end = np.random.choice(list_of_list_of_moves[action_start])
        
        return action_start,action_end
    
    def expand(self,policy):
        
        for action,prob in np.ndenumerate(policy):
            if prob > 0 :
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, False)
                child_state = self.game.change_perspective(child_state) 
                
                child = Node(self.game, self.args, child_state, 1, self, action,prob)
                self.children.append(child)
        
        
        #classic mcts
        # action_start,action_end = self.select_random_action(self.expandable_moves)
        
        # action = [action_start,action_end]
        
        # #On vide cette action dans notre list des expandable moves
        # self.expandable_moves[action_start] = [] 
        # #hm peur que ça modifie le state globale
        # child_state = self.state.copy()
        # child_state = self.game.get_next_state(child_state, action, False)
        # child_state = self.game.change_perspective(child_state) 
        
        # child = Node(self.game, self.args, child_state, 1, self, action)
        # self.children.append(child)
        
        #return child
    
    #pas utilisé, seulement dans le cas du mcts classique
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)
        
        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = 1
        rollout_biggest_loop = self.game.biggest_loop
        rollout_number_of_moves = self.game.number_of_moves
        rollout_list_of_position = self.game.list_of_positions.copy()
        
        while True: #ça risque de boucler longtemps et de ne jamais trouver de victoire, et juste de draw après 50 coups
        
            valid_moves = self.game.get_valid_moves(rollout_state, rollout_player)
            action_start,action_end = self.select_random_action(valid_moves)
            action = [action_start,action_end]
            rollout_state = self.game.get_next_state(rollout_state, action)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                #on remet les valeurs de biggest loop et de number of moves à ce qu'il était avant la boucle
                #pour ne pas les compter pour après
                #puisqu'ils sont updaté après chaque get_next_state
                self.game.biggest_loop = rollout_biggest_loop
                self.game.number_of_moves = rollout_number_of_moves
                self.game.list_of_positions = rollout_list_of_position
                return value    
            
            rollout_player = self.game.get_opponent(rollout_player)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
    
    @torch.no_grad()
    def search(self, state):
        
        #definir root
        root = Node(self.game, self.args, state,visit_count=1)
        
        policy, root_value = self.model(

            torch.tensor( self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy,axis=2).squeeze(0).squeeze(0).cpu().numpy()
        
        policy = (1-self.args["dirichlet_epsilon"])*policy + self.args["dirichlet_epsilon"]*np.random.dirichlet([self.args["dirichlet_alpha"]]*625).reshape((25,25))
        
        valid_moves = self.game.get_valid_moves(state, root.player) 
        valid_moves_mask = self.game.get_matrix_of_valid_moves(valid_moves)
        policy *= valid_moves_mask 
        
        policy /= np.sum(policy)
        
        root.expand(policy)
        
        
        for search in range(self.args['num_searches']):
            node = root
        #selection
            while node.is_fully_expanded():
                node = node.select()
        
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
         
            if not is_terminal:
                
                policy,value = self.model(
                    
                    torch.tensor( self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                
                policy = torch.softmax(policy,axis=2).squeeze(0).squeeze(0).cpu().numpy()
                #on a une matrice de proba, on veut maintenant masquer tous les moves interdit
                valid_moves = self.game.get_valid_moves(node.state, node.player) 
                
                valid_moves_mask = self.game.get_matrix_of_valid_moves(valid_moves)
                #valid_moves est une list de list : pas pratique : faut transformer ça en matrice
                
                policy *= valid_moves_mask #multiplication point par point, la matrice servant de masque
                
                policy /= np.sum(policy)
                
                value = value.item()
                
                #expansion
                node.expand(policy)
                #simulation dans le cas du MCTS normal  
                #value = node.simulate()
       
        
        #backpropagation
            node.backpropagate(value)    
            
        #return visit counts
        action_probs = np.zeros((self.game.action_size,self.game.action_size))
        for child in root.children:
            action_taken_start,action_taken_end = child.action_taken[0], child.action_taken[1]
            action_probs[action_taken_start][action_taken_end] = child.visit_count
            
        if np.sum(action_probs) > 0:
            action_probs /= np.sum(action_probs)
        return action_probs, root_value.item()
    
class AlphaPan():
    def __init__(self,model,optimizer,game,args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        
        self.mcts = MCTS(game, args, model)
        
    def selfPlay(self):
        memory = []
        player = 1
        
        self.game.reset()
        state = self.game.get_initial_state()
        
        
        while True:
           
            if player == 1:
                
                action_probs, _ = self.mcts.search(state)

                memory.append((state,action_probs,player))

                temperature_action_probs = action_probs ** (1.0 / self.args["temperature"])
                temperature_action_probs /= np.sum(temperature_action_probs)  # re-normalize to sum=1
                action_float = np.random.choice(self.game.action_size*self.game.action_size, p=np.matrix.flatten(temperature_action_probs))
                action = (action_float//self.game.action_size,action_float%self.game.action_size)

                state = self.game.get_next_state(state, action)
               
                
            if player == -1:
                neutral_state = self.game.change_perspective(state)
                action_probs, _ = self.mcts.search(neutral_state)

                memory.append((neutral_state,action_probs,player))

                temperature_action_probs = action_probs ** (1.0 / self.args["temperature"])
                temperature_action_probs /= np.sum(temperature_action_probs)  # re-normalize to sum=1
                action_float = np.random.choice(self.game.action_size*self.game.action_size, p=np.matrix.flatten(temperature_action_probs))
                action = (action_float//self.game.action_size,action_float%self.game.action_size)
                
                neutral_state = self.game.get_next_state(neutral_state, action)
                state = self.game.change_perspective(neutral_state)
                #L'action est directement liée au neutral_state
                #il faut l'appliqué sur ce neutral_state avant d'obtenir le state
                
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                returnMemory = []
                
                for hist_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((self.game.get_encoded_state(hist_state),
                                         hist_action_probs,
                                         hist_outcome)
                                        )
                return returnMemory
            
            player = self.game.get_opponent(player)
            
                
    
    def train(self,memory):
        random.shuffle(memory)
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0
        for batchIndex in range(0,len(memory),self.args["batch_size"]):
            sample = memory[batchIndex:min(len(memory), batchIndex + self.args["batch_size"])]
            state, policy_targets, value_targets = zip(*sample) #ça créé 3 listes à partir de la liste des tuples
            state, policy_targets, value_targets = np.array(state),np.array(policy_targets),np.array(value_targets).reshape(-1,1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            if out_policy.dim() == 4:
                out_policy = out_policy.squeeze(1)  # devient (B, 25, 25)

            policy_loss = F.binary_cross_entropy_with_logits(out_policy,policy_targets)
            value_loss = F.mse_loss(out_value,value_targets)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            num_batches += 1
        return policy_loss_sum, value_loss_sum, num_batches
        
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()

            win_count = 0
            draw_count = 0
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'], desc=f"Iter {iteration:03d} self-play"):
                game_memory = self.selfPlay()
                memory += game_memory
                # Inspect last tuple's outcome: +1 = win, -1 = non-win (draw or loss)
                if game_memory:
                    last_outcome = game_memory[-1][2]
                    if last_outcome == 1:
                        win_count += 1
                    elif last_outcome == -1:
                        draw_count += 1

            self.model.train()

            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches_total = 0
            for epochs in trange(self.args['num_epochs'], desc=f"Iter {iteration:03d} training"):
                pl, vl, nb = self.train(memory)
                total_policy_loss += pl
                total_value_loss += vl
                num_batches_total += nb

            avg_pl = total_policy_loss / max(num_batches_total, 1)
            avg_vl = total_value_loss / max(num_batches_total, 1)
            total_games = self.args['num_selfPlay_iterations']
            print(
                f"Iter {iteration:03d} | "
                f"PolicyLoss={avg_pl:.4f} | "
                f"ValueLoss={avg_vl:.4f} | "
                f"WinRate={win_count/total_games:.2%} | "
                f"NonWinRate={(total_games - win_count)/total_games:.2%}"
            )

            torch.save(self.model.state_dict(), "model.pt")
            torch.save(self.optimizer.state_dict(), "optim.pt")

if __name__ == "__main__":
    game = Chenapan()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaPanNet(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # num_iterations and num_selfPlay_iterations are tunable — start with 100/100, adjust based on GPU speed and convergence
    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 100,           # was 3 — scale to 100 for real learning
        'num_selfPlay_iterations': 100,  # was 1 — scale to 100 games/iteration
        'num_epochs': 4,
        'batch_size': 64,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.1,
        'dirichlet_alpha': 0.3
    }

    alphaPan = AlphaPan(model,optimizer,game,args)
    alphaPan.learn()
    
# chenapan = Chenapan()
# player = 1

# args = {
#     'C': 2,
#     'num_searches': 100
# }

# model = AlphaPanNet()

# model.eval()

# mcts = MCTS(chenapan,args,model)

# state = chenapan.get_initial_state()

# while True:
#     print(state)
    
#     action = []
    
#     if player == 1:
        
#         valid_moves = chenapan.get_valid_moves(state,player)
#         print("valid_moves :")
#         for i in range(len(valid_moves)):
#             if len(valid_moves[i]) != 0 :
#                 print("{} => {}".format(i,valid_moves[i]))
#         print("total number of moves",chenapan.number_of_moves)
#         print("biggest loop", chenapan.biggest_loop)
#         print("choose a starting value")
#         action_start = int(input(f"{player}:"))
#         print("you chose ", action_start)
#         print("choose an ending value")
#         action_end =  int(input(f"{player}:"))
#         print("you chose ", action_end)
      
#         action = [action_start,action_end]
        
#         if not action_end in valid_moves[action_start]:
#             print("action not valid, you pass your turn")
#             continue
        
#         #le plateau est normal, on applique donc l'action qu'on voit
#         state = chenapan.get_next_state(state, action)
            
#     else:
        
#         neutral_state = chenapan.change_perspective(state)
#         # print("the bot sees this :")
#         # print(neutral_state)
#         mcts_probs = mcts.search(neutral_state)
#         action = np.unravel_index(np.argmax(mcts_probs), mcts_probs.shape)
#         print("action chosen by the bot is {} to {}".format(action[0],action[1]))
#         #le plateau est retourné et l'action adaptée à ce plateau là
#         #donc on va appliquer l'action adaptée sur ce plateau, puis le retourner pour update state
#         neutral_state = chenapan.get_next_state(neutral_state, action)
#         state = chenapan.change_perspective(neutral_state)
        
#     #state est mis à jour, on va vérifié si c'est fini ou pas
    
#     value, is_terminal = chenapan.get_value_and_terminated(state, action)
    
#     if is_terminal:
#         print(state)
#         if value == 1:
#             print(player, "won")
#         else:
#             print("draw")
#         break
        
#     player = chenapan.get_opponent(player)
    
    