#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import logging
import datetime
import time
import numpy as np

np.set_printoptions(precision=6, suppress=True)

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *
from evaluation import *
import argparse

# setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
if not os.path.isdir('logs'):
    os.makedirs('logs')
log_file = 'logs/log_{}.log'.format(datetime.datetime.now().strftime("%Y_%B_%d_%I-%M-%S%p"))
open(log_file, 'a').close()

# create logger
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# add to log file
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def log(str): logger.info(str)

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

log('is GPU available? {}'.format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description="TGGAN")

datasets = ['taxi', 'authorship', 'checkin', 'sim_100', 'sim_500', 'sim_2000']
parser.add_argument("-d", "--dataset", default="checkin", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
processes = ['rand_binomial', 'rand_poisson']
parser.add_argument("-sp", "--simProcess", default="rand_binomial", type=str,
                    help="one of: {}".format(", ".join(sorted(processes))))
parser.add_argument("-nn", "--numberNode", default=10, type=int,
                    help="if run simulation data, this is the number of nodes")
parser.add_argument("-p", "--probability", default=0.5, type=int,
                    help="if run simulation data, this is the number of time slices")
parser.add_argument("-nt", "--numberTime", default=10, type=int,
                    help="this is the number of time slices for both real data and simulation data")
parser.add_argument("-sc", "--scale", default=0.1, type=float,
                    help="scale of gaussian prior for kernel density estimation in DeepTemporalWalk")
parser.add_argument("-iw", "--init_walk_method", default='uniform', type=str,
                    help="TemporalWalk sampler")

# DeepTemporalWalk
parser.add_argument("-bs", "--batch_size", default=32, type=int,
                    help="random walks batch size in DeepTemporalWalk")
parser.add_argument("-lr", "--learningrate", default=1e-3, type=float,
                    help="if this run should run all evaluations")
parser.add_argument("-rl", "--rw_len", default=4, type=int,
                    help="random walks maximum length in DeepTemporalWalk")
parser.add_argument("--num-G-layer", default=15, type=int,
                    help="The number of layers in the Generator")
parser.add_argument("-ud", "--use_decoder", default='normal', type=str,
                    help="if decoder function")
parser.add_argument("-es", "--embedding_size", default=16, type=int,
                    help="embedding size of nodes, W_down")
parser.add_argument("-td", "--time_deconv", default=8, type=int,
                    help="deconv output channels number")
parser.add_argument("-ts", "--time_sample_num", default=4, type=int,
                    help="time sampling number")
parser.add_argument("-cm", "--constraint_method", default='min_max', type=str,
                    help="time constraint computing method")
parser.add_argument("-ne", "--n_eval_loop", default=40, type=int,
                    help="number of walk loops")
parser.add_argument("--W_down_generator_size", default=128, type=int,
                    help="The size of W_down_generator_size")
parser.add_argument("--W_down_discriminator_size", default=128, type=int,
                    help="The size of W_down_discriminator_size")
parser.add_argument("--noise-dim", default=64,
                    help="The dim of the random noise that is used as input.")
parser.add_argument("--hidden-unit", default=64,
                    help="The dim of the random noise that is used as input.")
parser.add_argument("--x-mode", default='uniform', type=str,
                    help="time constraint computing method")
parser.add_argument("--noise-type", choices=["Gaussian", "Uniform"], default="Gaussian",
                    help="The noise type to feed into the generator.")
parser.add_argument("--constraint-method", choices=["min_max", "relu", "clip"], default="min_max",
                    help="The constraint method for time budget.")
parser.add_argument("--spatial-constraint-method", choices=["nearest_point"], default="nearest_point",
                    help="The constraint method for geo location.")
parser.add_argument("--n-epochs", default=5, type=int,
                    help="max epoches")
parser.add_argument("--clip-value", type=float, default=0.05,
                    help="lower and upper clip value for disc. weights")
parser.add_argument("--n-critic", type=int, default=5,
                    help="number of training steps for discriminator per iter")
parser.add_argument("-mi", "--max_iters", default=100000, type=int,
                    help="max iterations")
parser.add_argument("-ev", "--eval_every", default=1000, type=int,
                    help="evaluation interval of epochs")

parser.add_argument("-te", "--is_test", default=True, type=bool,
                    help="if this is a testing period.")
parser.add_argument("--contact_time", default=0.01, type=float,
                    help="the contact time when evaluating the temporal network")
parser.add_argument("--early_stopping", default=1e-5, type=float,
                    help="stop training if evaluation metrics are good enough")

args = parser.parse_args()


class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, edges, n_nodes, t_end, scale, args):
        self.edges = edges
        self.n_nodes = n_nodes
        self.t_end = t_end
        self.scale = scale
        
        self.rw_len = args.rw_len
        self.init_walk_method = args.init_walk_method
        # Initialize a temporal walker
        self.walker = TemporalWalker(self.n_nodes, self.edges, self.t_end, self.scale, self.rw_len, 1,
                                init_walk_method=self.init_walk_method)
        
        self.data = self.cache(self.walker)
        
    def cache(self, walker):
        real_walk_data = []
        for i in range(20000):
            temp = walker.walk().__next__()[0]
            real_walk_data.append(temp)
        
        print("Done!")
        return real_walk_data
    
    def __getitem__(self, item):
        walks = self.data[item]
        return walks
    
    def __len__(self):
        return len(self.data)


class Generator(nn.Module):
    def __init__(self, N, n_samples, t_end, args):
        super(Generator, self).__init__()
        self.N = N
        self.t_end = t_end
        
        self.x_mode = args.x_mode
        self.rw_len = args.rw_len
        self.num_G_layer = args.num_G_layer
        self.use_decoder = args.use_decoder
        self.batch_size = args.batch_size
        self.noise_dim = args.noise_dim
        self.hidden_unit = args.hidden_unit
        self.W_down_generator_size = args.W_down_generator_size
        self.constraint_method = args.constraint_method
        self.spatial_constraint_method = args.spatial_constraint_method
        
        #self.polygon, self.zone = read_geojson(args.dataset)
        self.init_lin_1 = nn.Linear(self.noise_dim, self.W_down_generator_size)
        self.init_lin_2_h = nn.Linear(self.W_down_generator_size, self.W_down_generator_size)
        self.init_lin_2_c = nn.Linear(self.W_down_generator_size, self.W_down_generator_size)
        
        self.lin_input_x = nn.Sequential(
                            nn.Linear(2, 1, bias=False),
                            nn.Linear(1, self.W_down_generator_size),
                            nn.Tanh()
                            )
        
        self.lin_t0_loc = nn.Sequential(
                            nn.Linear(self.W_down_generator_size, self.hidden_unit),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.lin_t0_scale = nn.Sequential(
                            nn.Linear(self.W_down_generator_size, self.hidden_unit),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.lin_t0_last = nn.Linear(1, self.W_down_generator_size)
        
        self.lin_tau_loc = nn.Sequential(
                            nn.Linear(self.W_down_generator_size, self.hidden_unit),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.lin_tau_scale = nn.Sequential(
                            nn.Linear(self.W_down_generator_size, self.hidden_unit),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.lin_tau_last = nn.Linear(1, self.W_down_generator_size)
        
        self.lin_coord_x_loc = nn.Sequential(
                            nn.Linear(self.W_down_generator_size, self.hidden_unit),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.lin_coord_x_scale = nn.Sequential(
                            nn.Linear(self.W_down_generator_size, self.hidden_unit),
                            nn.Linear(self.hidden_unit, 1)
                            )
        
        self.lin_coord_y_loc = nn.Sequential(
                            nn.Linear(self.W_down_generator_size, self.hidden_unit),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.lin_coord_y_scale = nn.Sequential(
                            nn.Linear(self.W_down_generator_size, self.hidden_unit),
                            nn.Linear(self.hidden_unit, 1)
                            )
        
        self.lin_coord_last = nn.Linear(2, self.W_down_generator_size)
        
        self.lin_w_down = nn.Linear(self.W_down_generator_size, self.N)
        self.lin_w_up = nn.Linear(self.N, self.W_down_generator_size)
        
        self.lin_end = nn.Sequential(
                            nn.Linear(self.W_down_generator_size, self.hidden_unit),
                            nn.Tanh(),
                            nn.Linear(self.hidden_unit, 2)
                            )
        
        self.lstm = nn.LSTM(self.W_down_generator_size, self.W_down_generator_size, self.num_G_layer)
        
    def generate_time(self, output, name):
        if name == 't0':
            loc_t0 = self.lin_t0_loc(output)
            scale_t0 = self.lin_t0_scale(output)
            #time = [self.truncated_normal_(torch.ones(1).to(device), mean=loc_t0[i, 0], std=scale_t0[i, 0]) for i in range(self.batch_size)]
            time = torch.distributions.normal.Normal(loc_t0, torch.relu(scale_t0)+1e-7).rsample()
        elif name == 'tau':
            loc_t0 = self.lin_tau_loc(output)
            scale_t0 = self.lin_tau_scale(output)
            #time = [self.truncated_normal_(torch.ones(1).to(device), mean=loc_t0[i, 0], std=scale_t0[i, 0]) for i in range(self.batch_size)]
            time = torch.distributions.normal.Normal(loc_t0, torch.relu(scale_t0)+1e-7).rsample()
        #time = torch.stack(time)
        return time

    def generate_coordinate(self, output):
        x_loc = self.lin_coord_x_loc(output)
        x_scale = self.lin_coord_x_scale(output)
        y_loc = self.lin_coord_y_loc(output)
        y_scale = self.lin_coord_y_scale(output)
        x_coord = torch.distributions.normal.Normal(x_loc, torch.relu(x_scale)+1e-7).rsample()
        y_coord = torch.distributions.normal.Normal(y_loc, torch.relu(y_scale)+1e-7).rsample()
        return torch.cat((x_coord, y_coord), dim=1)
    
    def truncated_normal_(self, tensor, mean=0, std=0.03):
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor
    
    def time_constraint(self, t, epsilon=1e-2, method='min_max'):
        if method == 'relu':
            t = torch.relu(t) - torch.relu(t - 1.)
        elif method == 'clip':
            t = torch.clamp(t, 0., 1.)
        elif method == 'min_max':
            min_ = torch.min(t)
            max_ = torch.max(t)
            
            if min_ < epsilon:
                t = t-min_
            if max_ > 1.0:
                t = t/max_
        return t
    
    def spatial_constraint(self, coordinates, polygon, zone, method='nearest_point'):
        with torch.no_grad():
            cloest_point = []
            for coordinate in coordinates:
                point = Point(utm.from_latlon(coordinate.cpu().numpy()[0], coordinate.cpu().numpy()[1]))
                if polygon.contains(point):
                    cloest_point.append(coordinate)
                else:
                    p1, p2 = nearest_points(polygon, point)
                    nearest_point = list(map(float, re.findall("\d+\.\d+", p1.wkt)))
                    nearest_point = utm.to_latlon(nearest_point[0], nearest_point[1], zone[0], zone[1])
                    cloest_point.append(torch.Tensor(nearest_point))
            return torch.stack(cloest_point)
    
    def forward(self, z, x_input=None, t0_input=None, edge_input=None, coord_input=None, tau_input=None):
        init_c, init_h = [], []
        for _ in range(self.num_G_layer):
            intermediate = torch.tanh(self.init_lin_1(z))
            init_c.append(torch.tanh(self.init_lin_2_c(intermediate)))
            init_h.append(torch.tanh(self.init_lin_2_h(intermediate)))
        # Initialize an input tensor
        hidden = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))
        
        # generate start x, and its residual time
        if self.x_mode == 'uniform':
            if x_input is not None:
                x_output = x_input
            else:
                # generate start node binary if not need
                x_output = torch.randint(low=0, high=2, size=(self.batch_size,))
                x_output = F.one_hot(x_output, 2).to(device)
        elif self.x_mode == 'generate':
            print("This method is not implemented yet")
        
        inputs = self.lin_input_x(x_output.float())
        
        # LSTM steps
        node_outputs = []
        tau_outputs = []
        coord_outputs = []
        
        # generate the first three start elements: start x, residual time, and maximum possible length
        out, hidden = self.lstm(inputs.unsqueeze(0), hidden)
        if t0_input is not None: # for evaluation generation
            t0_res_output = t0_input
        else:
            t0_res_output = self.generate_time(out.squeeze(0), 't0')
            
            if self.constraint_method != "none":
                t0_res_output = self.time_constraint(
                    t0_res_output, method=self.constraint_method) * self.t_end
            
            t0_reference = torch.ones(self.batch_size).to(device)
            condition = torch.eq(torch.argmax(x_output, axis=-1), t0_reference)
            t0_res_output = torch.where(condition, t0_reference, t0_res_output.squeeze(-1)).unsqueeze(-1)
        
        res_time = t0_res_output
        # convert to input
        inputs = torch.tanh(self.lin_t0_last(t0_res_output))
        
        # LSTM time steps
        for i in range(self.rw_len):
            # generate temporal edge part
            for j in range(2):
                out, hidden = self.lstm(inputs.unsqueeze(0), hidden)
                if edge_input is not None and i <= self.rw_len-2:  # for evaluation generation
                    output = edge_input[:, i*2 + j]
                else:
                    # Decrease to dimension N
                    logit = self.lin_w_down(out.squeeze(0))
                    output = F.gumbel_softmax(logit, dim=1, tau=3, hard=True)
                
                node_outputs.append(output)
                # Back to dimension d
                inputs = self.lin_w_up(output)
                
                # Calculate the coordinates
                if coord_input is not None and i <= self.rw_len-2:  # for evaluation generation
                    coord_output = coord_input[:, i*2 + j]
                else:
                    out, hidden = self.lstm(inputs.unsqueeze(0), hidden)
                    coord_output = self.generate_coordinate(out.squeeze(0))
                    #print(coord_output[0, :])
                    #if self.spatial_constraint_method != "none":
                        #coord_output = self.spatial_constraint(coord_output, self.polygon, self.zone).to(device)
                        
                coord_outputs.append(coord_output)
                # Back to dimension d
                inputs = self.lin_coord_last(coord_output)
                
            # LSTM for tau
            out, hidden = self.lstm(inputs.unsqueeze(0), hidden)
            if tau_input is not None and i <= self.rw_len-2:  # for evaluation generation
                tau = tau_input[:, i]
            else:
                tau = self.generate_time(out.squeeze(0), 'tau')
                
                if self.constraint_method != "none":
                    tau = self.time_constraint(tau, method=self.constraint_method) * res_time
            
            res_time = tau
            tau_outputs.append(tau)
            # convert to input
            inputs = torch.tanh(self.lin_tau_last(tau))
        
        # LSTM for end indicator
        out, hidden = self.lstm(inputs.unsqueeze(0), hidden)
        # generate end binary
        end_logit = self.lin_end(out.squeeze(0))
        end_output = F.gumbel_softmax(end_logit, dim=1, tau=3, hard=True)
        
        node_outputs = torch.stack(node_outputs, dim=1)
        tau_outputs = torch.stack(tau_outputs, dim=1)
        coord_outputs = torch.stack(coord_outputs, dim=1)
        return x_output, t0_res_output, node_outputs, tau_outputs, end_output, coord_outputs


class Discriminator(nn.Module):
    def __init__(self, n_samples, n_nodes, args):
        super(Discriminator, self).__init__()
        self.N = n_nodes
        self.W_down_discriminator_size = args.W_down_discriminator_size
        self.hidden_unit = args.hidden_unit
        self.rw_len = args.rw_len
        self.batch_size = n_samples
        
        self.W_down_x_discriminator_1 = nn.Linear(2, 1, bias=False)
        self.W_down_x_discriminator_2 = nn.Linear(1, self.W_down_discriminator_size)
        
        self.W_down_res_discriminator = nn.Linear(1, self.W_down_discriminator_size)
        
        self.W_down_discriminator = nn.Linear(self.N, self.W_down_discriminator_size, bias=False)
        
        self.W_down_tau_discriminator = nn.Linear(1, self.W_down_discriminator_size)
        
        self.W_down_end_discriminator_1 = nn.Linear(2, 1, bias=False)
        self.W_down_end_discriminator_2 = nn.Linear(1, self.W_down_discriminator_size)
        
        self.W_down_coord_discriminator = nn.Linear(2, self.W_down_discriminator_size, bias=False)
        
        self.lstm_all = nn.LSTM(self.W_down_discriminator_size, self.W_down_discriminator_size)
        self.lstm_node = nn.LSTM(self.W_down_discriminator_size, self.W_down_discriminator_size)
        self.lstm_time = nn.LSTM(self.W_down_discriminator_size, self.W_down_discriminator_size)
        self.lstm_loc = nn.LSTM(self.W_down_discriminator_size, self.W_down_discriminator_size)
        self.lstm_node_time = nn.LSTM(self.W_down_discriminator_size, self.W_down_discriminator_size)
        self.lstm_time_loc = nn.LSTM(self.W_down_discriminator_size, self.W_down_discriminator_size)
        self.lstm_node_loc = nn.LSTM(self.W_down_discriminator_size, self.W_down_discriminator_size)
        
        self.linear_out_all = nn.Sequential(
                            nn.Linear(self.W_down_discriminator_size, self.hidden_unit),
                            nn.Tanh(),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.linear_out_time = nn.Sequential(
                            nn.Linear(self.W_down_discriminator_size, self.hidden_unit),
                            nn.Tanh(),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.linear_out_node = nn.Sequential(
                            nn.Linear(self.W_down_discriminator_size, self.hidden_unit),
                            nn.Tanh(),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.linear_out_coord = nn.Sequential(
                            nn.Linear(self.W_down_discriminator_size, self.hidden_unit),
                            nn.Tanh(),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.linear_out_node_time = nn.Sequential(
                            nn.Linear(self.W_down_discriminator_size, self.hidden_unit),
                            nn.Tanh(),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.linear_out_node_coord = nn.Sequential(
                            nn.Linear(self.W_down_discriminator_size, self.hidden_unit),
                            nn.Tanh(),
                            nn.Linear(self.hidden_unit, 1)
                            )
        self.linear_out_time_coord = nn.Sequential(
                            nn.Linear(self.W_down_discriminator_size, self.hidden_unit),
                            nn.Tanh(),
                            nn.Linear(self.hidden_unit, 1)
                            )
        
    def forward(self, z):
        x, t0_res, node_inputs, tau_inputs, end, coords = z[0], z[1], z[2], z[3], z[4], z[5]
        # Reshape x
        x = torch.tanh(self.W_down_x_discriminator_2(self.W_down_x_discriminator_1(x.view(-1, 2).float())))
        # Reshape t0_res
        t0_res = torch.tanh(self.W_down_res_discriminator(t0_res.view(-1, 1).float()))
        # Reshape node_inputs
        node_inputs = self.W_down_discriminator(node_inputs.view(-1, self.N).float())
        node_inputs = node_inputs.view(-1, self.rw_len*2, self.W_down_discriminator_size)
        # Reshape tau_inputs
        tau_inputs = torch.tanh(self.W_down_tau_discriminator(tau_inputs.reshape(-1, 1).float()))
        tau_inputs = tau_inputs.view(-1, self.rw_len, self.W_down_discriminator_size)
        # Reshape end
        end = self.W_down_end_discriminator_2(self.W_down_end_discriminator_1(end.view(-1, 2).float()))
        # Reshape coordinates
        coords = self.W_down_coord_discriminator(coords.view(-1, 2).float())
        coords = coords.view(-1, self.rw_len*2, self.W_down_discriminator_size)
        
        inputs_all = [x] + [t0_res]
        inputs_time = [x] + [t0_res]
        inputs_node = []
        inputs_coord = []
        inputs_node_time = [x] + [t0_res]
        inputs_node_coord = []
        inputs_time_coord = [x] + [t0_res]
        
        for i in range(self.rw_len):
            inputs_all += [tau_inputs[:, i, :]]+[node_inputs[:, i*2, :]]+[coords[:, i*2, :]]+[node_inputs[:, i*2+1, :]]+[coords[:, i*2+1, :]]
            inputs_time += [tau_inputs[:, i, :]]
            inputs_node += [node_inputs[:, i*2, :]]+[node_inputs[:, i*2+1, :]]
            inputs_coord += [coords[:, i*2, :]]+[coords[:, i*2+1, :]]
            inputs_node_time += [tau_inputs[:, i, :]]+[node_inputs[:, i*2, :]]+[node_inputs[:, i*2+1, :]]
            inputs_node_coord += [node_inputs[:, i*2, :]]+[coords[:, i*2, :]]+[node_inputs[:, i*2+1, :]]+[coords[:, i*2+1, :]]
            inputs_time_coord += [tau_inputs[:, i, :]]+[coords[:, i*2, :]]+[coords[:, i*2+1, :]]
            
        inputs_all += [end]
        inputs_time += [end]
        inputs_node_time += [end]
        inputs_time_coord += [end]
        
        inputs_all = torch.stack(inputs_all)
        inputs_node = torch.stack(inputs_node)
        inputs_time = torch.stack(inputs_time)
        inputs_coord = torch.stack(inputs_coord)
        inputs_node_time = torch.stack(inputs_node_time)
        inputs_node_coord = torch.stack(inputs_node_coord)
        inputs_time_coord = torch.stack(inputs_time_coord)
        
        # LSTM outputs
        out_all, _ = self.lstm_all(inputs_all)
        out_node, _ = self.lstm_node(inputs_node)
        out_time, _ = self.lstm_time(inputs_time)
        out_coord, _ = self.lstm_loc(inputs_coord)
        out_node_time, _ = self.lstm_node_time(inputs_node_time)
        out_time_coord, _ = self.lstm_time_loc(inputs_time_coord)
        out_node_coord, _ = self.lstm_node_loc(inputs_node_coord)

        return (self.linear_out_all(out_all[-1]),self.linear_out_time(out_time[-1]),self.linear_out_node(out_node[-1]),
                self.linear_out_coord(out_coord[-1]),self.linear_out_node_time(out_node_time[-1]),
                self.linear_out_node_coord(out_node_coord[-1]),self.linear_out_time_coord(out_time_coord[-1]))


def generate_discrete(generator, n_samples, t_end, args):
    """
        Generate a random walk in index representation (instead of one hot). This is faster but prevents the gradients
        from flowing into the generator, so we only use it for evaluation purposes.
    """
    start_x_0 = Variable(F.one_hot(torch.zeros(n_samples,).to(torch.int64), 2).float()).to(device)
    start_x_1 = Variable(F.one_hot(torch.ones(n_samples,).to(torch.int64), 2).float()).to(device)
    start_t0 = Variable(torch.ones(n_samples, 1).float()).to(device)
    
    initial_noise = make_noise((n_samples, args.noise_dim), args.noise_type).to(device)
    
    fake_x, fake_t0, fake_e, fake_tau, fake_end, fake_coordinate = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(args.n_eval_loop):
            if i == 0:
                fake_x_output, fake_t0_res_output, fake_node_output, fake_tau_output, fake_end_output, fake_coord_output = generator(
                        initial_noise, x_input=start_x_1, t0_input=start_t0)
            else:
                if args.rw_len == 1:
                    t0_input = fake_tau_output[:, -1, :]
                    fake_x_output, fake_t0_res_output, fake_node_output, fake_tau_output, fake_end_output, fake_coord_output = generator(
                        initial_noise, x_input=start_x_0, t0_input=t0_input)
                else:
                    t0_input = fake_tau_output[:, 0, :]
                    edge_input = fake_node_output[:, 2:, :]
                    coord_input = fake_coord_output[:, 2:, :]
                    tau_input = fake_tau_output[:, 1:, :]
                    fake_x_output, fake_t0_res_output, fake_node_output, fake_tau_output, fake_end_output, fake_coordinate_output = generator(
                            initial_noise, x_input=start_x_0, t0_input=t0_input, edge_input=edge_input, tau_input=tau_input)
        
            fake_x_outputs_discrete = torch.argmax(fake_x_output, dim=-1)
            fake_node_outputs_discrete = torch.argmax(fake_node_output, dim=-1)
            fake_end_discretes = torch.argmax(fake_end_output, dim=-1)
        
            fake_x.append(fake_x_outputs_discrete.detach().cpu().numpy())
            fake_t0.append(fake_t0_res_output.detach().cpu().numpy())
            fake_e.append(fake_node_outputs_discrete.detach().cpu().numpy())
            fake_tau.append(fake_tau_output.detach().cpu().numpy())
            fake_end.append(fake_end_discretes.detach().cpu().numpy())
            fake_coordinate.append(fake_coord_output.detach().cpu().numpy())
            
    return fake_x, fake_t0, fake_e, fake_tau, fake_end, fake_coordinate


def weights_calculation(fake_scores, real_scores, eps=1.75):
    with torch.no_grad():
        fake_scores = torch.Tensor([sum(i) for i in fake_scores])
        real_scores = torch.Tensor([sum(i) for i in real_scores])
        
        abs_value = torch.abs(fake_scores-real_scores)
        alpha = eps*torch.softmax(abs_value, dim=-1)
        return alpha


def train(dataloader, generator, discriminator, optimizer_G, optimizer_D, args, n_nodes, t_end, 
          edge_contact_time, train_edges, test_edges, save_directory, output_directory, timing_directory,
          eval_every=4, max_iter=1000, eval_transitions=1e3, max_patience=5, is_test=True):
    n_eval_iters = int(eval_transitions / args.batch_size)
    # Getting a single data iterator
    iterloader = iter(dataloader)
    batches_done = 0
    
    # Read Polygons and the corresponding zones
    if args.dataset != "authorship":
        polygon, zone = read_geojson(args.dataset)
    else:
        polygons, zones = read_geojson(args.dataset)
    
    # Initialize the time
    starting_time = time.time()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    log("Start Training...")
    for epoch in range(max_iter):
        for i, real_walk in enumerate(dataloader):
            # Getting real edge inputs
            real_edge_inputs_discrete = real_walk[:, 1:, 0:2].to(torch.long)
            real_node_inputs_discrete = real_edge_inputs_discrete.view(args.batch_size, -1)
            real_node_inputs = F.one_hot(real_node_inputs_discrete, n_nodes).to(device)
            real_tau_inputs = real_walk[:, 1:, 2:3].to(device)
            real_coords_inputs = real_walk[:, 1:, 3:]
            real_coords_inputs = real_coords_inputs.reshape(args.batch_size, 2*args.rw_len, -1).to(device)
            # Getting real indicator inputs
            real_x_input_discretes = real_walk[:, 0, 0].to(torch.long)
            real_x_inputs = F.one_hot(real_x_input_discretes, 2).to(device)
            real_end_discretes = real_walk[:, 0, 1].to(torch.long)
            real_ends = F.one_hot(real_end_discretes, 2).to(device)
            real_t0_res_inputs = real_walk[:, 0:1, 2].to(device)
            
            initial_noise = make_noise((args.batch_size, args.noise_dim), args.noise_type).to(device)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            discriminator.train()
            optimizer_D.zero_grad()
            
            fake_output = generator(initial_noise)
            
            fake_x_inputs = fake_output[0].detach()
            fake_t0_res_inputs = fake_output[1].detach()
            fake_node_inputs = fake_output[2].detach()
            fake_tau_inputs = fake_output[3].detach()
            fake_ends = fake_output[4].detach()
            fake_coordinates = fake_output[5].detach()
            
            fake_scores = discriminator((fake_x_inputs, fake_t0_res_inputs, fake_node_inputs, fake_tau_inputs, fake_ends, fake_coordinates))
            real_scores = discriminator((real_x_inputs, real_t0_res_inputs, real_node_inputs, real_tau_inputs, real_ends, real_coords_inputs))
            
            alpha = weights_calculation(fake_scores, real_scores)
            fake_scores = [x*alpha[i] for i, x in enumerate(fake_scores)]
            real_scores = [x*alpha[i] for i, x in enumerate(real_scores)]

            loss_D = torch.mean(sum(fake_scores)) - torch.mean(sum(real_scores))
            
            loss_D.backward()
            optimizer_D.step()
            
            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            
            # Train the generator every n_critic iterations
            if i % args.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate a batch of random temporal walks
                syn_x_inputs, syn_t0_res_inputs, syn_node_inputs, syn_tau_inputs, syn_ends, syn_coordinates = generator(initial_noise)
                # Adversarial loss
                syn_scores = discriminator((syn_x_inputs, syn_t0_res_inputs, syn_node_inputs, syn_tau_inputs, syn_ends, syn_coordinates))
                syn_scores = [x*alpha[i] for i, x in enumerate(syn_scores)]
                loss_G = -torch.mean(sum(syn_scores))
                
                # Loss back-propagation
                loss_G.backward()
                optimizer_G.step()
            
                log("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, max_iter, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item()))
                
            batches_done += 1
            
        # Evaluate the model's progress.
        if (epoch+1) % eval_every == 0:
            log('**** Starting Evaluating: {}-th Epoch ****'.format(epoch+1))
            
            # Creating instances for fake and real graphs
            fake_graphs, fake_x_t0, real_walks, real_x_t0 = [], [], [], []
            
            for q in range(n_eval_iters):
                fake_output = generate_discrete(generator, args.batch_size, t_end, args)
                fake_x = fake_output[0]
                fake_t0 = fake_output[1]
                fake_edges = fake_output[2]
                fake_t = fake_output[3]
                fake_end = fake_output[4]
                fake_coord = fake_output[5]
                
                smpls = None
                stop = [False] * args.batch_size
                for i in range(args.n_eval_loop):
                    x, t0, e, tau, le, coo = \
                    fake_x[i], fake_t0[i], fake_edges[i], fake_t[i], fake_end[i], fake_coord[i]
                    
                    if q == 0 and i >= args.n_eval_loop-3:
                        log('eval_iters: {} eval_loop: {}'.format(q, i))
                        log('generated [x, t0, e, tau, end, coord]\n[{}, {}, {}, {}, {}]'.format(
                            x[0], t0[0, 0], e[0, :], tau[0, :, 0], le[0],
                            ))
                        log('generated [x, t0, e, tau, end, coord]\n[{}, {}, {}, {}, {}]'.format(
                            x[1], t0[1, 0], e[1, :], tau[1, :, 0], le[1],
                            ))
                    e = e.reshape(-1, args.rw_len, 2)
                    tau = tau.reshape(-1, args.rw_len, 1)
                    coo = coo.reshape(-1, args.rw_len, 4)
                    if i == 0:
                        smpls = np.concatenate([e, tau, coo],axis=-1)
                    else:
                        new_pred = np.concatenate([e[:, -1:], tau[:, -1:], coo[:, -1:]], axis=-1)
                        smpls = np.concatenate([smpls, new_pred], axis=1)
                    
                    # judge if reach max length
                    for b in range(args.batch_size):
                        b_le = le[b]

                        if i == 0 and b_le == 1:  # end
                            stop[b] = True
                        if i > 0 and stop[b]:  # end
                            smpls[b, -1, :] = -1
                        if i > 0 and not stop[b] and b_le == 1:
                            stop[b] = True

                fake_x = np.array(fake_x).reshape(-1, 1)
                fake_t0 = np.array(fake_t0).reshape(-1, 1)
                fake_len = np.array(fake_end).reshape(-1, 1)
                fake_start = np.c_[fake_x, fake_t0, fake_len]
                fake_x_t0.append(fake_start)
                fake_graphs.append(smpls)
                
                if q%5 == 0:
                    print("Done, {} of {} eval iterations".format(q, n_eval_iters))
                
            fake_graphs = np.array(fake_graphs)
            fake_coords = fake_graphs[:, :, :, 3:]
            fake_coords = fake_coords[fake_coords[:, :, :, 0] != -1].reshape(-1, 2)
            constrained_coords = fake_coords.reshape(-1, 2)
            if args.dataset != "authorship":
                constrained_coords = spatial_constraint(constrained_coords, polygon, zone)
            else:
                constrained_coords = spatial_constraint(constrained_coords, polygons=polygons, zones=zones)
            
            fake_graphs = fake_graphs[:, :, :, :3]
            
            for _ in range(n_eval_iters):
                try:
                    true_walk = next(iterloader)
                except StopIteration:
                    iterloader = iter(dataloader)
                    true_walk = next(iterloader)
                
                # Getting real edge inputs
                real_edge = true_walk[:, 1:, 0:2].to(torch.long).numpy()
                real_x = true_walk[:, 0, 0].to(torch.long).numpy()
                real_length = true_walk[:, 0, 1].to(torch.long).numpy()
                real_tau = true_walk[:, 1:, 2:3].numpy()
                real_t0 = true_walk[:, 0:1, 2].numpy()
                
                walk = np.c_[real_edge.reshape(-1, 2), real_tau.reshape(-1, 1)]
                real_walks.append(walk)
                real_start = np.stack([real_x, real_t0[:, 0], real_length], axis=1)
                real_x_t0.append(real_start)
                
            if is_test:
                try:
                    fake_walks = fake_graphs.reshape(-1, 3)
                    fake_mask = fake_walks[:, 0] > -1
                    fake_walks = fake_walks[fake_mask]
                    fake_x_t0 = np.array(fake_x_t0).reshape(-1, 3)

                    real_walks = np.array(real_walks).reshape(-1, 3)
                    real_mask = real_walks[:, 0] > -1
                    real_walks = real_walks[real_mask]
                    real_x_t0 = np.array(real_x_t0).reshape(-1, 3)
                    
                    truth_train_time = train_edges[:, 3:4]
                    truth_train_res_time = t_end - truth_train_time
                    truth_train_walks = np.concatenate([train_edges[:, 1:3], truth_train_res_time], axis=1)
                    truth_train_x_t0 = np.c_[np.zeros((len(train_edges), 1)), truth_train_res_time]
                    truth_train_x_t0 = np.r_[truth_train_x_t0, np.ones((len(train_edges), 2))]

                    truth_test_time = test_edges[:, 3:4]
                    truth_test_res_time = t_end - truth_test_time
                    truth_test_walks = np.c_[test_edges[:, 1:3], truth_test_res_time]
                    truth_test_x_t0 = np.c_[np.zeros((len(test_edges), 1)), truth_test_res_time]
                    truth_test_x_t0 = np.r_[truth_test_x_t0, np.ones((len(test_edges), 2))]

                    fake_e_list, fake_e_counts = np.unique(fake_walks[:, 0:2], return_counts=True, axis=0)
                    real_e_list, real_e_counts = np.unique(real_walks[:, 0:2], return_counts=True, axis=0)
                    
                    truth_train_e_list, truth_train_e_counts = np.unique(truth_train_walks[:, 0:2], 
                                                                         return_counts=True, axis=0)
                    truth_test_e_list, truth_test_e_counts = np.unique(truth_test_walks[:, 0:2],
                                                                       return_counts=True, axis=0)
                    truth_e_list, truth_e_counts = np.unique(
                        np.r_[truth_test_walks[:, 0:2], truth_test_walks[:, 0:2]], return_counts=True, axis=0)
                    n_e = len(truth_e_list)
                    
                    real_x_list, real_x_counts = np.unique(real_x_t0[:, 0], return_counts=True)
                    fake_x_list, fake_x_counts = np.unique(fake_x_t0[:, 0], return_counts=True)
                    truth_x_list, truth_x_counts = real_x_list, real_x_counts
                    
                    real_len_list, real_len_counts = np.unique(real_x_t0[:, 2], return_counts=True)
                    fake_len_list, fake_len_counts = np.unique(fake_x_t0[:, 2], return_counts=True)
                    truth_len_list, truth_len_counts = real_len_list, real_len_counts
                    
                    fig = plt.figure(figsize=(2 * 9, 2 * 9))
                    fig.suptitle('Truth, Real, and Fake edges comparisons')
                    dx = 0.3
                    dy = dx
                    zpos = 0
                    
                    fake_ax = fig.add_subplot(221, projection='3d')
                    fake_ax.bar3d(fake_e_list[:, 0], fake_e_list[:, 1], zpos, dx, dy, fake_e_counts)
                    fake_ax.set_xlim([0, n_nodes])
                    fake_ax.set_ylim([0, n_nodes])
                    fake_ax.set_xticks(range(n_nodes))
                    fake_ax.set_yticks(range(n_nodes))
                    fake_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
                    fake_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
                    fake_ax.set_title('fake edges number: {}'.format(len(fake_e_list)))
                    
                    real_ax = fig.add_subplot(222, projection='3d')
                    real_ax.bar3d(real_e_list[:, 0], real_e_list[:, 1], zpos, dx, dy, real_e_counts)
                    real_ax.set_xlim([0, n_nodes])
                    real_ax.set_ylim([0, n_nodes])
                    real_ax.set_xticks(range(n_nodes))
                    real_ax.set_yticks(range(n_nodes))
                    real_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
                    real_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
                    real_ax.set_title('real edges number: {}'.format(len(real_e_list)))

                    truth_ax = fig.add_subplot(223, projection='3d')
                    truth_ax.bar3d(truth_train_e_list[:, 0], truth_train_e_list[:, 1], zpos, dx, dy,
                                   truth_train_e_counts)
                    truth_ax.set_xlim([0, n_nodes])
                    truth_ax.set_ylim([0, n_nodes])
                    truth_ax.set_xticks(range(n_nodes))
                    truth_ax.set_yticks(range(n_nodes))
                    truth_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
                    truth_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
                    truth_ax.set_title('truth train edges number: {}'.format(len(truth_train_e_list)))

                    truth_ax = fig.add_subplot(222, projection='3d')
                    truth_ax.bar3d(truth_test_e_list[:, 0], truth_test_e_list[:, 1], zpos, dx, dy, truth_test_e_counts)
                    truth_ax.set_xlim([0, n_nodes])
                    truth_ax.set_ylim([0, n_nodes])
                    truth_ax.set_xticks(range(n_nodes))
                    truth_ax.set_yticks(range(n_nodes))
                    truth_ax.set_xticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
                    truth_ax.set_yticklabels([str(n) if n % 5 == 0 else '' for n in range(n_nodes)])
                    truth_ax.set_title('truth test edges number: {}'.format(len(truth_test_e_list)))

                    plt.tight_layout()
                    #plt.show()
                    plt.savefig('{}/iter_{}_edges_counts_validation.png'.format(output_directory, epoch+1), dpi=90)
                    plt.close()

                except ValueError as e:
                    print(e)
                    log('reshape fake walks got error. Fake graphs shape: {} \n{}'.format(fake_graphs[0].shape, fake_walks[:3]))
                    continue
                    
            fake_graphs = convert_graphs(fake_graphs)
            fake_graph_file = "{}/{}_assembled_graph_iter_{}.npz".format(output_directory, timestr, epoch+1)
            fake_graphs[:, 3] = t_end - fake_graphs[:, 3]
            np.savez_compressed(fake_graph_file, fake_graphs=fake_graphs, real_walks=real_walks)
            
            real_coords = test_edges[:, 4:].reshape(-1, 2)
            real_coords = np.sort(real_coords, axis=0)
            fake_coords = np.sort(constrained_coords, axis=0)
            
            Gs = Graphs(test_edges[:, :4], N=n_nodes, tmax=t_end, edge_contact_time=edge_contact_time)
            FGs = Graphs(fake_graphs, N=n_nodes, tmax=t_end, edge_contact_time=edge_contact_time)
            mmd_avg_degree = MMD_Average_Degree_Distribution(Gs, FGs)
            mmd_avg_group_size = MMD_Average_Group_Size(Gs, FGs)
            mmd_avg_group_number = MMD_Mean_Group_Number(Gs, FGs)
            mmd_avg_coordination_number = MMD_Mean_Coordination_Number(Gs, FGs)
            mmd_mean_group_number = MMD_Mean_Group_Number(Gs, FGs)
            mmd_group_duration = MMD_Mean_Group_Duration(Gs, FGs)
            mmd_avg_coordinates = MMD(real_coords[:2000], fake_coords[:2000])
            log('mmd_avg_degree: {}'.format(mmd_avg_degree))
            log('mmd_avg_group_size: {}'.format(mmd_avg_group_size))
            log('mmd_avg_group_number: {}'.format(mmd_avg_group_number))
            log('mmd_avg_coordination_number: {}'.format(mmd_avg_coordination_number))
            log('mmd_mean_group_number: {}'.format(mmd_mean_group_number))
            log('mmd_group_duration: {}'.format(mmd_group_duration))
            log('mmd_avg_coordinates: {}'.format(mmd_avg_coordinates))
            
            log('Real Mean_Average_Degree_Distribution: \n{}'.format(Gs.Mean_Average_Degree_Distribution()))
            log('Fake Mean_Average_Degree_Distribution: \n{}'.format(FGs.Mean_Average_Degree_Distribution()))
            
            if args.early_stopping is not None:
                if mmd_avg_degree < args.early_stopping:
                    log('**** end training because evaluation is reached ****')
                    break
            
            t = time.time() - starting_time
            log('**** end evaluation **** took {} seconds so far...'.format(int(t)))
        log('The weights on each discriminator {} at {}-th Epoch'.format(alpha, epoch+1))
                
                
def main(args):
    save_directory = "snapshots-{}".format(args.dataset)
    output_directory='outputs-{}'.format(args.dataset)
    timing_directory='timings-{}'.format(args.dataset)
    
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    if not os.path.isdir(timing_directory):
        os.makedirs(timing_directory)

    log('use {} data'.format(args.dataset))
    log('-'*40)
    
    if args.dataset == "taxi":
        scale = 0.1
        train_ratio = 0.85
        t_end = 1.
        n_nodes = 66
        t0 = 0.1
        edge_contact_time = t0*0.01
    elif args.dataset == "checkin":
        scale = 0.1
        train_ratio = 0.85
        t_end = 1.
        n_nodes = 70
        t0 = 0.1
        edge_contact_time = t0*0.01
    elif args.dataset == "authorship":
        scale = 0.1
        train_ratio = 0.85
        t_end = 1.
        n_nodes = 628
        t0 = 0.1
        edge_contact_time = t0*0.1
    elif args.dataset == "sim_100":
        scale = 0.1
        train_ratio = 0.85
        t_end = 1.
        t0 = 0.1
        n_nodes = 99
        edge_contact_time = t0*0.5
    elif args.dataset == "sim_500":
        scale = 0.1
        train_ratio = 0.85
        t_end = 1.
        t0 = 0.1
        n_nodes = 495
        edge_contact_time = t0*0.5
    else:
        scale = 0.1
        train_ratio = 0.85
        t_end = 1.
        t0 = 0.1
        n_nodes = 2420
        edge_contact_time = t0*0.5
        
    edges = np.loadtxt('data/{}_dataset/{}_data.txt'.format(args.dataset, args.dataset))
    train_edges, test_edges = Split_Train_Test(edges, train_ratio)
    
    real_walks = TemporalDataset(train_edges, n_nodes, t_end, scale, args)
    dataloader = DataLoader(real_walks, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    generator = Generator(n_nodes, args.batch_size, t_end, args).to(device)
    discriminator = Discriminator(args.batch_size, n_nodes, args).to(device)
    
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.learningrate)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.learningrate)
    
    train(dataloader, generator, discriminator, optimizer_G, optimizer_D, args, n_nodes, t_end, 
          edge_contact_time, train_edges, test_edges, save_directory, output_directory, timing_directory)

if __name__ == "__main__":
    print("Start Training...")
    main(args)
