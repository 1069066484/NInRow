# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: ZeroNN's training pipeline is implemented here.
    Multiprocess processing is supported.
"""
import sys
from global_defs import *
from mcts import MctsPuct
import threading
import time
from games.game import *
from nets.ZeroNN import *
from utils.data import *
import copy
import random
import multiprocessing as mp


MctsPuct.CHECK_DETAILS = False

idx = (('_' + sys.argv[1]) if len(sys.argv) > 1 else '')
rcn = '115' + idx


class ZeroNNTrainer:
    """
    ZeroNNTrainer performs ZeroNN's training. 
    Tripal groups of threads are used for training:
        1. optimization.
        2. evaluator. Evaluation must be fast; the best situation is that once a new model is trained, the it can be evaluated.
        3. self-play. Self-play should play games using the best generated model.
    And we use multiple processes for evaluator and self-play and multiple threads for optimizer and .
    The details are elaborated in the paper 'Mastering the game of Go without human knowledge'.
    """
    def __init__(self, folder, board_rows=int(rcn[0]), board_cols=int(rcn[1]), n_in_row=int(rcn[2]), 
                 train_ratio=0.5, mcts_sims=1024, self_play_cnt=3000, batch_size=512, verbose=True, n_eval_processes=2, best_player_path='zero_nns115'+idx+'/NNs/model.ckpt-5448', n_play_processes=9, plays=8, start_nozero=False, do_opt=True, mp_parallel=True):
        self.mp_parallel = mp_parallel
        self.Lock = mp.Lock if mp_parallel else threading.Lock
        self.Process = mp.Process if mp_parallel else threading.Thread


        # no sharing
        self.n_play_processes = n_play_processes
        self.n_eval_processes = n_eval_processes
        self.do_opt = do_opt
        self.folder = mkdir(folder)
        self.manager = mp.Manager()

        self.lock_train_data = self.Lock()
        self.lock_model_paths = self.Lock()
        self.lock_model_best = self.Lock()

        self.unchecked_model_paths = self.manager.list([])  if mp_parallel else list([])
        
        self.loss_hists = []

        # thread sharings
        self.batch_size = batch_size
        self.logger = log.init_LoggerMP(join(self.folder, logfn('ZeroNNTrainer-' + curr_time_str())), verbose) \
            if mp_parallel else log.Logger(join(self.folder, logfn('ZeroNNTrainer-' + curr_time_str())), verbose)
        self.train_ratio = train_ratio

        # shared constants
        self.folder_selfplay = mkdir(join(self.folder, 'selfplay'))
        self.data_path = [join(self.folder_selfplay, npfn('selfplay' + str(i))) for i in range(3)]
        self.board_rows = board_rows if board_rows > n_in_row else board_rows + 10
        self.board_cols = board_cols if board_cols > n_in_row else board_cols + 10
        self.n_in_row = n_in_row
        self.mcts_sims = mcts_sims
        self.folder_NNs = mkdir(join(self.folder, 'NNs'))
        self.path_loss_hists = join(self.folder_selfplay, npfn('selfplay' + '_loss'))
        self.shared_constants = {'data_path': self.data_path,
                                 'board_rows': self.board_rows,
                                 'board_cols': self.board_cols,
                                 'n_in_row': self.n_in_row,
                                 'mcts_sims': self.mcts_sims,
                                 'folder_NNs': self.folder_NNs,
                                 'folder_selfplay': self.folder_selfplay,
                                 'path_loss_hists': self.path_loss_hists}

        # shared variables
        curr_generation = 0

        # self.nozero_mcts is the initial MCTS
        # self.nozero_mcts use no ZeroNN for seaching and we use it to generate initial training data
        # instead of using zeroNNs with randomized parameters
        # After the first generation of zeroNN trained, self.nozero_mcts would be set None since it is not needed anymore
        if exists(self.path_loss_hists):
            self.loss_hists = np.load(self.path_loss_hists).tolist()
            try:
                curr_generation = self.loss_hists[-1][0]
            except:
                curr_generation = 0
        else:
            self.loss_hists = []

        self.loss_hists = self.manager.list(self.loss_hists) if mp_parallel else  list(self.loss_hists)
        self.shared_vars = {'self_play_cnt': self_play_cnt,
                            'plays': plays,
                            'best_player_path': best_player_path,
                            'model_avail': not start_nozero,
                            'data_avail': 0,
                            'resign_val': 100,
                            'hand_val': 0,
                            'curr_generation': curr_generation,
                            'nozero_mcts_sims': int(mcts_sims * 2),
                            'nozero_mcts': start_nozero
                            }
        self.shared_vars = self.manager.dict(self.shared_vars) if mp_parallel else dict(self.shared_vars)
        self.event_opt = mp.Event()

    @staticmethod
    def manip_train_data( shared_constants, lock_train_data, data_to_append=None, num2keep=None):
        with lock_train_data:
            data_path = shared_constants['data_path']
            if exists(data_path[0]) and exists(data_path[1]) and exists(data_path[1]):
                train_data = [np.load(p) for p in data_path]
            else:
                train_data = [np.zeros([0, shared_constants['board_rows'], shared_constants['board_cols'], 4]).astype(np.bool), 
                              np.zeros([0, shared_constants['board_rows'] * shared_constants['board_cols']]), 
                              np.zeros([0,1])]
            if data_to_append is not None:
                train_data = [np.vstack([train_data[0], data_to_append[0].astype(np.bool)]), 
                    np.vstack([train_data[1], data_to_append[1]]),
                    np.vstack([train_data[2], data_to_append[2]])]
            if num2keep is not None:
                for i in range(3):
                    train_data[i] = train_data[i][-num2keep:]
            if data_to_append is not None or num2keep is not None:
                for i in range(3):
                    np.save(data_path[i], train_data[i])
            return train_data

    @staticmethod
    def init_train_data(shared_constants):
        """
        Just ignore the function. It help build a randomized model.
        """
        return [np.zeros([0, shared_constants['board_rows'], shared_constants['board_cols'], 4]).astype(np.bool), 
                              np.zeros([0, shared_constants['board_rows'] * shared_constants['board_cols']]), 
                              np.zeros([0,1])]

    def train(self):
        """
        Entrance codes
        """
        # The optimizer and console run in multiple threads
        print("Main thread-train")
        self.event_opt.set()
        threads = ([threading.Thread(target=self.optimization)] if self.do_opt else []) \
            + [threading.Thread(target=self.console)]
        print('Optimization and console threads initialized')
        for t in threads:
            t.start()
        print('Optimization and console threads started')
        # Process = threading.Thread if self.do_opt else Process
        processes = [self.Process(target=ZeroNNTrainer.evaluator, 
                                args=(self.shared_constants, self.shared_vars, self.unchecked_model_paths, 
                                      self.lock_model_paths, self.lock_model_best, self.logger, self.event_opt)) 
                     for _ in range(self.n_eval_processes)] + \
                    [self.Process(target=ZeroNNTrainer.self_play,
                                args=(self.shared_constants, self.shared_vars, self.unchecked_model_paths, 
                                      self.lock_train_data, self.lock_model_best, self.loss_hists, 
                                      self.path_loss_hists, self.logger, self.event_opt))
                     for _ in range(self.n_play_processes)]
        print('evaluator and self_play processes initialized')
        for p in processes:
            p.start()
        print('evaluator and self_play processes started')
        while self.shared_vars['self_play_cnt'] > 0:
            time.sleep(60*10)
        for p in processes:
            p.join()
        self.logger.log('trained')


    def console(self):
        """
        Use console to help training
        """
        print("console started")
        while True:
            cmd = input()
            try:
                if cmd == 'md':
                    folder = input('Input the folder name where the three data file(np file) exists. '+
                     'The files should be named [selfplay0.npy], [selfplay1.npy] and [selfplay2.npy]:\n')
                    try:
                        data_path = [join(folder, npfn('selfplay' + str(i))) for i in range(3)]
                        train_data = [np.load(p) for p in data_path]
                        train_data = ZeroNNTrainer.manip_train_data(self.shared_constants, self.lock_train_data, train_data)
                        # self.shared_vars['data_avail'] = True
                        print("load new data succeessfully, data size = ", len(train_data[0]))
                    except:
                        print("md error: folder or files do not exist or are invalid")
                    del train_data
                elif cmd == 'pl':
                    plays = input("Input the number of games for one self-play, it should be a positive even number:\n")
                    def plerr():
                        print("pl error: ",plays," is invalid to be a self-play number")
                    try:
                        plays = int(plays)
                        if plays > 0 and plays % 2 == 0:
                            self.shared_vars['plays'] = plays
                            print("self.plays -> ", self.shared_vars['plays'])
                        else:
                            plerr()
                    except:
                        plerr()
                elif cmd == 'bs':
                    batch_size = input("Input the number of batch size, it should be a positive number:\n")
                    def bserr():
                        print("bs error")
                    try:
                        batch_size = int(batch_size)
                        if batch_size > 0:
                            self.batch_size = batch_size
                            print("Batch size reset to" ,batch_size)
                        else:
                            bserr()
                    except:
                        bserr()
                elif cmd == 'bp':
                    best_player_path = input("Input num best player path, th current is " + str(self.shared_vars['best_player_path']) + ":\n")
                    if os.path.exists(best_player_path + '.index'):
                        self.shared_vars['best_player_path'] = best_player_path
                        print("best player path -> ", best_player_path)
                    else:
                        print("bp error,",best_player_path,'is invalid')
                elif cmd == 'nm':
                    def nmerr():
                        print("nm error")
                    nozero_mcts_sims = input("Input number of simulations of nozero_mcts's search. A non-positive number means deactivation\n" + " Currently, nozero_mcts is " + str(self.shared_vars['nozero_mcts']) + ". And nozero_mcts_sims is " + str(self.shared_vars['nozero_mcts_sims']) + ":\n")
                    try:
                        nozero_mcts_sims = int(nozero_mcts_sims)
                        if nozero_mcts_sims > 0:
                            self.shared_vars['nozero_mcts_sims'] = nozero_mcts_sims
                            self.shared_vars['nozero_mcts'] = True
                            print("nozero_mcts activated. And nozero_mcts_sims is ", nozero_mcts_sims)
                        else:
                            self.shared_vars['nozero_mcts'] = False
                            print("nozero_mcts deactivated. And nozero_mcts_sims is ", nozero_mcts_sims)
                    except:
                        nmerr()
                elif cmd == 'op':
                    print("Error: Op command is unavailable in MP mode")
                    # self.shared_vars['data_avail'] = True
                elif cmd == 'pc':
                    self_play_cnt = input("Input number of play count:\n")
                    try:
                        self_play_cnt = int(self_play_cnt)
                        self.shared_vars['self_play_cnt'] = max(self_play_cnt,0)
                        print("self.self_play_cnt -> ", self.shared_vars['self_play_cnt'])
                    except:
                        print("pc error")
                elif cmd == 'te':
                    # use te will clear the screen
                    game = Game(self.board_rows,self.board_cols,self.n_in_row, Game.Player.AI, Game.Player.human, collect_ai_hists=False)
                    zeroNN1 = ZeroNN(path=join(self.folder_NNs), ckpt_idx=-1)
                    game.players[0].mcts.zeroNN = zeroNN1
                    game.players[0].mcts.max_acts = self.mcts_sims
                    game.start(graphics=True)
                elif cmd == 'di':
                    def dierr():
                        print("di error")
                    train_data = ZeroNNTrainer.manip_train_data(self.shared_constants, self.lock_train_data)
                    discard = input("Input number of data to leave, the current number is " + str(len(train_data[0])) +":\n")
                    try:
                        discard = int(discard)
                        if discard < 0 or discard > len(train_data[0]) - 1:
                            dierr()
                        train_data = ZeroNNTrainer.manip_train_data(self.shared_constants, self.lock_train_data, None, discard)
                        print("Discard successfully! The current number is ", len(train_data[0]))
                    except:
                        dierr()
                    del train_data
                elif cmd == 'hv':
                    def hverr():
                        print('hv errpr')
                    hand_val = input("Input hand_val, the current number is " + str(self.shared_vars['hand_val']) +":\n")
                    try:
                        self.shared_vars['hand_val'] = float(hand_val)
                    except:
                        hverr()
                else:
                    print("command error. (cmd=",cmd ,")")
            except:
                print("Unknown console error!")

    def optimization(self):
        # print('opt0')
        with self.lock_model_paths:
            zeroNN = ZeroNN(verbose=40,path=self.folder_NNs, ckpt_idx=-1, num_samples=100000, trained_model_paths=self.unchecked_model_paths,
                           epoch=80, batch_size=self.batch_size, save_epochs=40, logger=self.logger)
        self.logger.log('optimization start!')
        time.sleep(240)
        while self.shared_vars['self_play_cnt'] > 0:
            # print('opt1')
            while self.shared_vars['data_avail'] != 0:
                # print('opt2')
                time.sleep(240)
                if self.mp_parallel:
                    self.event_opt.clear()
            train_data = ZeroNNTrainer.manip_train_data(self.shared_constants, self.lock_train_data)
            # Wait for the models to be evaluated
            # Better models need to be selected to generate better data
            # remove old models to ease burden of evaluator
            while len(self.unchecked_model_paths) > 10:
                time.sleep(20)
                with self.lock_model_paths:
                    self.unchecked_model_paths.remove(
                        self.unchecked_model_paths[round(np.random.rand() * 8)])

            # select some playing histories to train to control overfitting
            nonrep_rand_nums = non_repeated_random_nums(len(train_data[0]), round(self.train_ratio * len(train_data[0])))
            print('optimization fit',len(nonrep_rand_nums),'of', len(train_data[0]),'...')
            for i in range(3):
                train_data[i] = train_data[i][nonrep_rand_nums]
            zeroNN.fit(train_data[0],
                       train_data[1],
                       train_data[2], 0.1)
            self.event_opt.set()
            self.shared_vars['model_avail'] = True
            time.sleep(120)
        print('optimization over')

    @staticmethod
    def evaluator(consts, vars, unchecked_model_paths, lock_model_paths, lock_model_best, logger, event_opt):
        while not vars['model_avail']:
            time.sleep(60)
        with lock_model_paths:
            while len(unchecked_model_paths) != 0:
                unchecked_model_paths.pop()
        logger.log('evaluator start!')
        while vars['self_play_cnt'] > 0 or len(unchecked_model_paths) > 5:
            while len(unchecked_model_paths) == 0:
                time.sleep(20)
            # use 'with' to lock to avoid forgetting to release it 
            with lock_model_paths:
                # wait for latest trained model
                if len(unchecked_model_paths) == 0:
                    continue
                path_to_check = unchecked_model_paths.pop()
            event_opt.wait()
            with lock_model_best:
                vars['data_avail'] += 1
            logger.log('evaluator:',vars['best_player_path'], 'VS' , path_to_check, '...')
            if vars['nozero_mcts'] == False:
                best_mcts = Mcts(
                    0,0,zeroNN=ZeroNN(verbose=False,path=consts['folder_NNs'], ckpt_idx=vars['best_player_path']),
                    max_acts_=consts['mcts_sims']//2,const_temp=0,noise=0, resign_val=vars['resign_val'], hand_val=vars['hand_val'])
            else:
                # When self.nozero_mcts is not None, the first generation of zeroNN is not generated yet
                # We double number of simulations since MCTS without zeroNN can make a faster searching,
                # which also means any trained model is able to defeat MCTS without zeroNN using doule simulations
                best_mcts = Mcts(0,0,zeroNN=None,max_acts_=consts['mcts_sims']//2,const_temp=0.2,noise=0, hand_val=vars['hand_val'])
            zeroNN_to_check = ZeroNN(verbose=False,path=consts['folder_NNs'], ckpt_idx=path_to_check)
            mcts2 = Mcts(0,0,zeroNN=zeroNN_to_check,max_acts_=consts['mcts_sims']//2,const_temp=0,noise=0, resign_val=vars['resign_val'], hand_val=vars['hand_val'])
            
            # the evaluation must be fast to select the best model
            # play only several games, but require the player to check have a overwhelming advantage over the existing player
            # if the best player is defeated, then the player to check can take the first place
            winrate1, winrate2, tie_rate, _ = \
                eval_mcts(consts['board_rows'], consts['board_cols'], consts['n_in_row'], best_mcts, mcts2, False, 8, False)
            logger.log('evaluator:',vars['best_player_path'], 'VS' , path_to_check,'--', winrate1,'-',winrate2,'-',tie_rate)
            with lock_model_best:
                vars['data_avail'] -= 1
            # if the new player wins 3 out of 4 and draws in one game, replace the best player with it
            if winrate2 >= 0.6 and winrate1 <= 0.35:
                vars['curr_generation'] += 1
                logger.log('evaluator:',path_to_check, 'defeat' , vars['best_player_path'], 'by', winrate2 - winrate1)
                logger.log(path_to_check, 'becomes generation' , vars['curr_generation'])
                with lock_model_best:
                    vars['best_player_path'] = path_to_check
                vars['nozero_mcts'] = False
        print("evaluator over")

    @staticmethod
    def self_play(consts, vars, unchecked_model_paths, lock_train_data, lock_model_best, loss_hists, path_loss_hists, logger, event_opt):
        while not vars['model_avail'] and vars['nozero_mcts'] == False:
            time.sleep(5)
        logger.log('self_play start!')
        while vars['self_play_cnt'] > 0:
            zeroNN1 = ZeroNN(verbose=False,path=consts['folder_NNs'], ckpt_idx=vars['best_player_path'])
            zeroNN2 = ZeroNN(verbose=False,path=consts['folder_NNs'], ckpt_idx=vars['best_player_path'])
            best_player_path = vars['best_player_path']
            # we do not lock for self_play_cnt
            while vars['self_play_cnt'] > 0:
                event_opt.wait()
                with lock_model_best:
                    vars['data_avail'] += 1
                vars['self_play_cnt'] -= vars['plays']
                # decay resign_val
                # rookies should always play the game to the end while masters are allowed to resign at an earlier stage
                vars['resign_val'] = max(0.75, vars['resign_val'] - vars['resign_val'] * 0.00001 * vars['plays'])
                
                # Create two identical players to 'self play'
                if vars['nozero_mcts']:
                    mcts1 = Mcts(0,0,zeroNN=None,max_acts_=vars['nozero_mcts_sims'],const_temp=1,noise=0, 
                                  temp2zero_moves=3, hand_val=vars['hand_val'])
                    mcts2 = Mcts(0,0,zeroNN=None,max_acts_=vars['nozero_mcts_sims'],const_temp=1,noise=0, 
                                  temp2zero_moves=3, hand_val=vars['hand_val'])
                else:
                    mcts1 = Mcts(0,0,zeroNN=zeroNN1,max_acts_=consts['mcts_sims'], const_temp=1, 
                                 temp2zero_moves=3, noise=0.1, resign_val=vars['resign_val'], hand_val=vars['hand_val'])
                    mcts2 = Mcts(0,0,zeroNN=zeroNN2,max_acts_=consts['mcts_sims'], const_temp=1, 
                                 temp2zero_moves=3, noise=0.1, resign_val=vars['resign_val'], hand_val=vars['hand_val'])
                t = time.time()
                logger.log('self_play:','self_play_cnt=',vars['self_play_cnt'],' self.resign_val=',vars['resign_val'])
                winrate1, winrate2, tie_rate, ai_hists = \
                    eval_mcts(consts['board_rows'], consts['board_cols'], consts['n_in_row'], mcts1, mcts2, False, vars['plays']//2, True)
                ai_hists = ZeroNNTrainer.hists2enhanced_train_data(ai_hists, consts)
                # Evaluate how the zeroNN works on the latest played game.
                # This is the real test data since the data are not feeded for zeroNN's training yet so we need to save the 
                # evaluations.
                if not vars['nozero_mcts']:
                    eval = zeroNN1.run_eval(ai_hists[0], ai_hists[1], ai_hists[2])
                    logger.log(
                            'self-play in ' + str(time.time() - t) + 's\n',
                            'sp items:        [loss_policy,  loss_value,   loss_total,    acc_value,   acc_policy]:',
                            '\n   eval:         ',eval)
                    loss_hists.append([vars['curr_generation']] + eval)
                # Append the latest data to the old.
                # save the training data in case that we need to use them to continue training
                train_data = ZeroNNTrainer.manip_train_data(consts, lock_train_data, ai_hists)
                np.save(path_loss_hists, np.array(loss_hists))
                logger.log('self_play:',winrate1, winrate2 , tie_rate,'  new data size=', 
                                ai_hists[0].shape, '   total data:', train_data[0].shape)
                with lock_model_best:
                    vars['data_avail'] -= 1
                    find_new_best = (vars['best_player_path'] != best_player_path)
                if find_new_best: 
                    # Discard some old data since a new best player is trained, we need to use data of games played by
                    # it to train new models
                    if len(train_data[0]) > 20000:
                        ZeroNNTrainer.manip_train_data(consts, lock_train_data, None, -round(len(train_data[0]) * 0.6+1))
                    break
        print("self play over")

    @staticmethod
    def try_enh(X, Y_policy, Y_value):
        """
        it takes about 0.005s to dispose one data item
        """
        if len(X) == 0:
            return
        x = X[-1]
        y_policy = Y_policy[-1]
        y_value = Y_value[-1]
        def enh(dir):
            x_enh = x.copy()
            for i in range(3):
                # print("\nbefore\n",x[:,:,i])
                x_enh[:,:,i] = matMove(x[:,:,i], dir)
                # print(x_enh[:,:,i])
            y_policy_enh = matMove(y_policy, dir)
            y_policy_enh *= 1.0 / np.sum(y_policy_enh)
            # print(np.sum(y_policy_enh))
            X.append(x_enh)
            Y_policy.append(y_policy_enh)
            Y_value.append(y_value)
        s = np.sum(x[:,:,:3])
        if s == 0:
            return
        for step in [1,2]:
            for sx, sx2, dir in \
                            [[np.sum(x[:(2+step),:,:3]), np.sum(x[-2:,:,:3]), [-step,0]],
                            [np.sum(x[:,:(2+step),:3]),  np.sum(x[:,-2:,:3]), [0,-step]],
                            [np.sum(x[-(2+step):,:,:3]), np.sum(x[:2,:,:3]), [step,0]],
                            [np.sum(x[:,-(2+step):,:3]), np.sum(x[:,:2,:3]), [0,step]]]:
                if sx < 1e-6 and (s <= 7 or  sx2 < 1e-6):
                    enh(dir)

    @staticmethod
    def hists2enhanced_train_data(ai_hists, consts):
        """
        Convert game histories to training data
        Training data are saved with a very low cost of space
        """
        X = []
        Y_policy = []
        Y_value = []
        for hist in ai_hists:
            for i in range(len(hist[0])):
                X.append(hist[1][i])
                Y_policy.append(hist[0][i])
                # the begining two steps should be treated as tie
                Y_value.append([0 if hist[2] is None else 
                                (int(hist[2] != i % 2) * 2 - 1)])
                '''
                If p0 - p1 - p0 wins, => hist[2] = 0
                Y_value.append => -1, 1, -1
                '''
                # ZeroNNTrainer.try_enh(X, Y_policy, Y_value)
        return [np.array(X, dtype=np.bool), 
                np.array(Y_policy, dtype=np.float32).reshape(-1,consts['board_rows'] * consts['board_cols']), 
                np.array(Y_value, dtype=np.int8)]

    @staticmethod
    def merge_hist_files(folder):
        """
        The indicated folder should contain base files selfplay0.npy, selfplay1.npy, selfplay2.npy,
            and to-merge files selfplay0 xx.npy, selfplay1 xx.npy and selfplay2 xx.npy.
        Then to-merge files will be merged into base files
        """
        data = [np.load(npfn(join(folder, 'selfplay') + str(i))) for i in range(3)]
        print('Original:', data[0].shape, data[1].shape, data[2].shape)
        m = 0
        for fn in os.listdir(folder):
            if fn.startswith('selfplay0'):
                m += 1
                post_fix = fn[len('selfplay0'):]
                data_new = [np.load(npfn(join(folder, 'selfplay') + str(i) + post_fix)) for i in range(3)]
                if not (data_new[0].shape[0] == data_new[1].shape[0] == data_new[2].shape[0]):
                    print('Bad data found, discard')
                    continue
                for i in range(3):
                    data[i] = np.vstack([data[i], data_new[i]])
        data[2][data[2] > 0.01] = 1
        data[2][data[2] < 0.01] = -1
        print(m, 'items merged:', data[0].shape, data[1].shape, data[2].shape)
        for i in range(3):
            np.save(npfn(join(folder, 'selfplay') + str(i)), data[i])



def main():
    trainer = ZeroNNTrainer(FOLDER_ZERO_NNS+rcn)
    trainer.train()


def _test_merge_hist_files():
    ZeroNNTrainer.merge_hist_files(r'F:\Software\vspro\NInRow\NInRow\zero_nns115\godd_mod\nomcts')


def eval_test():
    zeroNN1 = ZeroNN(verbose=False,path=mkdir(join(FOLDER_ZERO_NNS, 'NNs')), ckpt_idx=-1)
    zeroNN2 = ZeroNN(verbose=False,path=mkdir(join(FOLDER_ZERO_NNS, 'NNs')), ckpt_idx=-1)
    mcts1 = Mcts(0,0,zeroNN=zeroNN1,max_acts_=100)
    mcts2 = Mcts(0,0,zeroNN=zeroNN2,max_acts_=100)
    winrate1, winrate2, tie_rate, ai_hists = \
        eval_mcts(5, 5, 4, mcts1, mcts2, True, 1, True)

def test_try_enh():
    X = np.zeros([4,6,4])
    X[:,:,0] = np.array(
        [[0,0,1,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]]
        )
    X[:,:,1] = np.array(
        [[0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]]
        )
    X[:,:,2] = np.array(
        [[0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]]
        )
    X[:,:,3] = np.array(
        [[0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]]
        )
    X = [X]
    Y_policy = [
        np.array(
            [[0,    0,      0,      0,      0.1,    0],
            [0.1,   0,      0,      0,      0,      0.1],
            [0,     0,      0.3,      0,      0,      0],
            [0,     0.2,    0.2,    0,      0,      0]]
            )
        ]
    Y_value = [-0.5]
    t = time.time()
    ZeroNNTrainer.try_enh(X, Y_policy, Y_value)
    for i in range(100):
        X.pop()
        Y_policy.pop()
        Y_value.pop()
        ZeroNNTrainer.try_enh(X, Y_policy, Y_value)
    print('time=',time.time() - t)
    print(len(X), X[0].shape) 
    for i in range(4):
        print(X[-1][:,:,i])
    print('\n')
    print(Y_policy[-1])
    print(Y_value[-1])


if __name__=='__main__':
    # test_try_enh()
    main()
    # _test_merge_hist_files()



