# -*- coding: utf-8 -*-


import json
import numpy as np
np.set_printoptions(precision=6, suppress=True)
from scipy.stats import gaussian_kde, norm
from datetime import date, timedelta
import re
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import utm
import pyproj
from shapely.geometry import Point, Polygon
from shapely.geometry import shape, mapping
from shapely.ops import nearest_points

class TemporalWalker:
    """
    Helper class to generate temporal random walks on the input user-trips matrix.
    The matrix gets shape: [day, hour, origin, destination]
    Parameters
    -----------
    edges: edges [[d, i, j]], shape: samples x 3
    edges_times: real time of edges [time], shape: samples x 1
    """

    def __init__(self, n_nodes, edges, t_end, scale=0.1, rw_len=4, batch_size=8,
                 init_walk_method='uniform'):
        #if edges.shape[1] != 4: raise Exception('edges must have shape: samples x 4')

        self.n_nodes = n_nodes
        self.t_end = t_end
        self.edges_days = edges[:, [0]]
        self.edges = edges[:, [1, 2]]
        self.edges_times = edges[:, [3]]
        self.coords = edges[:, 4:]
        self.rw_len = rw_len
        self.batch_size = batch_size
        # self.loc = loc
        self.scale = scale
        self.init_walk_method = init_walk_method

        # # kernel density estimation around current time
        # e_t_dict = {}
        # for i in range(len(self.edges)):
        #     e = list(self.edges[i, ])

    def walk(self):
        while True:
            yield temporal_random_walk(
                self.n_nodes, self.edges_days, self.edges, self.edges_times, self.t_end, self.coords,
                self.scale, self.rw_len, self.batch_size, self.init_walk_method)

# @jit(nopython=True)
def temporal_random_walk(n_nodes, edges_days, edges, edges_times, t_end, coords,
                         scale, rw_len, batch_size, init_walk_method):
    unique_days = np.unique(edges_days.reshape(1, -1)[0])
    walks = []

    for _ in range(batch_size):
        while True:
            # select a day with uniform distribution
            walk_day = np.random.choice(unique_days)
            mask = edges_days.reshape(1, -1)[0] == walk_day
            # subset for this day
            walk_day_edges = edges[mask]
            walk_day_times = edges_times[mask]
            walk_day_coords = coords[mask]
            # select a start edge. and unbiased or biased to the starting edges
            n = walk_day_edges.shape[0]
            if n >= rw_len: break

        n = n - rw_len + 1
        if init_walk_method is 'uniform': probs = Uniform_Prob(n)
        elif init_walk_method is 'linear': probs = Linear_Prob(n)
        elif init_walk_method is 'exp': probs = Exp_Prob(n)
        else: raise Exception('wrong init_walk_method!')

        if n == 1: start_walk_inx = 0
        else: start_walk_inx = np.random.choice(n, p=probs)

        selected_walks = walk_day_edges[start_walk_inx:start_walk_inx + rw_len]
        selected_times = walk_day_times[start_walk_inx:start_walk_inx + rw_len]
        selected_coords = walk_day_coords[start_walk_inx:start_walk_inx + rw_len]
        
        # get start residual time
        if start_walk_inx == 0: t_res_0 = t_end
        else:
            # print('selected start:', selected_walks[0])
            t_res_0 = t_end - walk_day_times[start_walk_inx-1, 0]

        # convert to residual time
        selected_times = t_end - selected_times

        # # convert to edge index
        # selected_walks = [nodes_to_edge(e[0], e[1], n_nodes) for e in selected_walks]

        # add a stop sign of -1
        x = 1
        if start_walk_inx > 0: x = 0
        walks_mat = np.c_[selected_walks, selected_times, selected_coords]
        if rw_len > len(selected_walks):
            n_stops = rw_len - len(selected_walks)
            walks_mat = np.r_[walks_mat, [[-1, -1, -1]] * n_stops]

        # add start resdidual time
        if start_walk_inx == n-1:
            is_end = 1.
        else:
            is_end = 0.
        walks_mat = np.r_[[[x] + [is_end] + [t_res_0] + list(selected_coords[0])], walks_mat]

        walks.append(walks_mat)
    return np.array(walks)


def nodes_to_edge(v, u, N):
    return v * N + u


def edge_to_nodes(e, N):
    return (e // N, e % N)


def Split_Train_Test(edges, train_ratio):
    days = sorted(np.unique(edges[:, 0]))
    b = days[int(train_ratio*len(days))]
    train_mask = edges[:, 0] <= b
    train_edges = edges[train_mask]
    test_edges = edges[ ~ train_mask]
    return train_edges, test_edges


def KDE(data):
    kernel = gaussian_kde(dataset=data, bw_method='silverman')
    return kernel


def Sample_Posterior_KDE(kernel, loc, scale, n):
    points = []
    n1 = 100
    for i in range(n):
        new_data = np.random.normal(loc, scale, n1)
        prior_probs = norm.pdf((new_data - loc) / scale)
        gau_probs = kernel.pdf(new_data)
        new_data_probs = prior_probs * gau_probs
        selected = np.random.choice(n1, p=new_data_probs / new_data_probs.sum())
        points.append(new_data[selected])
    return np.array(points)


def Exp_Prob(n):
    # n is the total number of edges
    if n == 1: return [1.]
    c = 1. / np.arange(1, n + 1, dtype=np.int)
    #     c = np.cbrt(1. / np.arange(1, n+1, dtype=np.int))
    exp_c = np.exp(c)
    return exp_c / exp_c.sum()


def Linear_Prob(n):
    # n is the total number of edges
    if n == 1: return [1.]
    c = np.arange(n+1, 1, dtype=np.int)
    return c / c.sum()


def Uniform_Prob(n):
    # n is the total number of edges
    if n == 1: return [1.]
    c = [1./n]
    return c * n


def Distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def Get_Weekday(day):
    start_day = date(2016, 5, 1)
    current_day = start_day + timedelta(day)
    return current_day.weekday()


def Is_Weekend(day):
    w = Get_Weekday(day)
    return w in [0, 6]


def get_edge_times(data):
    edge_dict = {}
    for i, j, t in data[:, 1:]:
        edge = (int(i), int(j))
        if edge in edge_dict: edge_dict[edge] = edge_dict[edge] + [t]
        else: edge_dict[edge] = [t]
    return edge_dict


def plot_edge_time_hist(edge_dict, t0, tmax, bins, ymax, save_file=None, show=True):
    edges = list(edge_dict.keys())
    n_fig = len(edges)
    hight = n_fig * 2
    fig, ax = plt.subplots(n_fig, 1, figsize=(12, hight))
    for i in range(n_fig):
        e = edges[i]
        ax[i].hist(edge_dict[e], range=[t0, tmax], bins=bins)
        ax[i].set_ylim(0, ymax)
        ax[i].set_title('edge {}'.format(e))
        if i < n_fig-1: ax[i].set_xticklabels([])
    if save_file: plt.savefig(save_file)
    if show: plt.show()


def convert_graphs(fake_graphs):
    _, _, e, k = fake_graphs.shape
    fake_graphs = fake_graphs.reshape([-1, e, k])
    tmp_list = None
    for d in range(fake_graphs.shape[0]):
        d_graph = fake_graphs[d]
        d_graph = d_graph[d_graph[:, 2] > 0.]
        d_graph = np.c_[np.array([[d]] * d_graph.shape[0]), d_graph]
        if tmp_list is None:
            tmp_list = d_graph
        else:
            tmp_list = np.r_[tmp_list, d_graph]
    return tmp_list


def spatial_constraint(coordinates, polygon=None, zone=None, polygons=None, zones=None, method='nearest_point'):
    """Apply spatial constraints on the generated coordinates"""
    cloest_point = []
    if polygons is None:
        for coordinate in coordinates:
            try:
                point = Point(utm.from_latlon(coordinate[0], coordinate[1]))
                if polygon.contains(point):
                    cloest_point.append(coordinate)
                else:
                    p1, p2 = nearest_points(polygon, point)
                    nearest_point = list(map(float, re.findall("\d+\.\d+", p1.wkt)))
                    nearest_point = utm.to_latlon(nearest_point[0], nearest_point[1], zone[0], zone[1])
                    cloest_point.append(list(nearest_point))
            except utm.error.OutOfRangeError:
                cloest_point.append(coordinate)
    else:
        for coordinate in coordinates:
            try:
                point = Point(utm.from_latlon(coordinate[0], coordinate[1]))
                flag = False
                for polygon in polygons:
                    if polygon.contains(point):
                        cloest_point.append(coordinate)
                        flag = True
                if not flag:
                    distance = [polygon.exterior.distance(point) for polygon in polygons]
                    target_polygon = polygons[np.argmin(distance)]
                    target_zone = zones[np.argmin(distance)]
                    p1, p2 = nearest_points(target_polygon, point)
                    nearest_point = list(map(float, re.findall("\d+\.\d+", p1.wkt)))
                    nearest_point = utm.to_latlon(nearest_point[0], nearest_point[1], target_zone[0], target_zone[1])
                    cloest_point.append(list(nearest_point))
            except utm.error.OutOfRangeError:
                cloest_point.append(coordinate)
    return np.array(cloest_point)
        

def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.
    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".
    Returns
    -------
    noise tensor
    """

    if type == "Gaussian":
        noise = Variable(torch.randn(shape))
    elif type == 'Uniform':
        noise = Variable(torch.randn(shape).uniform_(-1, 1))
    else:
        raise Exception("ERROR: Noise type {} not supported".format(type))
    return noise


def read_geojson(dataset):
    """Convert geojson file to a polygon(s)"""
    if dataset != "authorship":
        if dataset == "taxi":
            with open('data/{}_dataset/nyc.geojson'.format(dataset)) as f:
                data = json.load(f)
            polygon_coordinates = data['features'][0]['geometry']['coordinates'][0][0]
        elif dataset == "checkin":
            with open('data/{}_dataset/tokyo.geojson'.format(dataset)) as f:
                data = json.load(f)
            polygon_coordinates = data['features'][0]['geometry']['coordinates'][0]
        else:
            with open('data/{}_dataset/la.geojson'.format(dataset)) as f:
                data = json.load(f)
            polygon_coordinates = data['features'][0]['geometry']['coordinates'][0]

        polygon_coordinates = [i[::-1] for i in polygon_coordinates]
        polygon_coordinates = [utm.from_latlon(i[0], i[1]) for i in polygon_coordinates]
        zone_signiture = polygon_coordinates[0][2:]
        polygon = [i[:2] for i in polygon_coordinates]

        # Transform the polygon coordinate to shapely object
        polygon = Polygon(polygon)
    
        return polygon, zone_signiture
    else:
        polygons, zones = [], []
        # Read polygons for authorship dataset
        with open('data/{}_dataset/authors.geojson'.format(dataset)) as f:
            data = json.load(f)
        temp = data['features']
        polygon_coordinates = []
        for i in temp:
            polygon_coordinates.append(i['geometry']['coordinates'][0])
        
        for i in range(len(polygon_coordinates)):
            lons = np.array([j[0] for j in polygon_coordinates[i]])
            lats = np.array([j[1] for j in polygon_coordinates[i]])
            temp = utm.from_latlon(lats, lons)
            zone_signiture = temp[2:]
            polygon = list(zip(temp[0], temp[1]))
            
            # Transform the polygon coordinate to shapely object
            polygon = Polygon(polygon)
            polygons.append(polygon)
            zones.append(zone_signiture)
        return polygons, zones