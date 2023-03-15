import numpy as np
import datetime


def id2position(node_id, lat_len, lon_len):
    lat = node_id // lon_len
    lon = node_id % lon_len
    cos_lat = np.cos((0.5 - (lat + 1) / (lat_len + 1)) * np.pi)
    sin_lon = np.sin(lon / lon_len * np.pi)
    cos_lon = np.cos(lon / lon_len * np.pi)
    return cos_lat, sin_lon, cos_lon


def fetch_mesh_nodes():
    nodes = []
    for i in range(128):
        cos_lat = np.cos((0.5 - (i + 1) / 129) * np.pi)
        for j in range(320):
            sin_lon = np.sin(j / 320 * np.pi)
            cos_lon = np.cos(j / 320 * np.pi)
            nodes.append([cos_lat, sin_lon, cos_lon])
    return nodes


def fetch_mesh_edges(r):
    assert 6 >= r >= 0

    step = 2 ** (6 - r)
    edges = []
    edge_attrs = []
    for i in range(0, 128, step):
        for j in range(0, 320, step):
            cur_node = id2position(i * 320 + j, 128, 320)
            if i - step >= 0:
                edges.append([(i - step) * 320 + j, i * 320 + j])
                target_node = id2position((i - step) * 320 + j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            if i + step < 128:
                edges.append([(i + step) * 320 + j, i * 320 + j])
                target_node = id2position((i + step) * 320 + j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            if j - step >= 0:
                edges.append([i * 320 + j - step, i * 320 + j])
                target_node = id2position(i * 320 + j - step, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            else:
                edges.append([i * 320 + 320 - step, i * 320 + j])
                target_node = id2position(i * 320 + 320 - step, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            if j + step < 320:
                edges.append([i * 320 + j + step, i * 320 + j])
                target_node = id2position(i * 320 + j + step, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            else:
                edges.append([i * 320, i * 320 + j])
                target_node = id2position(i * 320, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

    return edges, edge_attrs


def fetch_grid2mesh_edges():
    lat_span = 720 / 128
    lon_span = 1440 / 320
    edges = []
    edge_attrs = []
    for i in range(720):
        for j in range(1440):
            target_mesh_i = int(i / lat_span)
            target_mesh_j = int(j / lon_span)
            edges.append([i * 1440 + j, target_mesh_i * 320 + target_mesh_j])
            cur_node = id2position(i * 1440 + j, 720, 1440)
            target_node = id2position(target_mesh_i * 320 + target_mesh_j, 128, 320)
            tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
            edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_i = int(i / lat_span - 0.1)
            if i / lat_span - 0.1 > 0 and over_mesh_i != target_mesh_i:
                edges.append([i * 1440 + j, over_mesh_i * 320 + target_mesh_j])
                target_node = id2position(over_mesh_i * 320 + target_mesh_j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_i = int(i / lat_span + 0.1)
            if i / lat_span + 0.1 < 128 and over_mesh_i != target_mesh_i:
                edges.append([i * 1440 + j, over_mesh_i * 320 + target_mesh_j])
                target_node = id2position(over_mesh_i * 320 + target_mesh_j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_j = int(j / lon_span - 0.1)
            if j / lon_span - 0.1 < 0:
                edges.append([i * 1440 + 1439, target_mesh_i * 320 + target_mesh_j])
                target_node = id2position(target_mesh_i * 320 + target_mesh_j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            elif over_mesh_j != target_mesh_j:
                edges.append([i * 1440 + j, target_mesh_i * 320 + over_mesh_j])
                target_node = id2position(target_mesh_i * 320 + over_mesh_j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_j = int(j / lon_span + 0.1)
            if j / lon_span + 0.1 > 320:
                edges.append([i * 1440, target_mesh_i * 320 + target_mesh_j])
                target_node = id2position(target_mesh_i * 320 + target_mesh_j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            elif over_mesh_j != target_mesh_j:
                edges.append([i * 1440 + j, target_mesh_i * 320 + over_mesh_j])
                target_node = id2position(target_mesh_i * 320 + over_mesh_j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

    return edges, edge_attrs


def fetch_mesh2grid_edges():
    lat_span = 720 / 128
    lon_span = 1440 / 320
    edges = []
    edge_attrs = []
    for i in range(720):
        for j in range(1440):
            target_mesh_i = int(i / lat_span)
            target_mesh_j = int(j / lon_span)
            edges.append([target_mesh_i * 320 + target_mesh_j, i * 1440 + j])
            target_node = id2position(i * 1440 + j, 720, 1440)
            cur_node = id2position(target_mesh_i * 320 + target_mesh_j, 128, 320)
            tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
            edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_i = int(i / lat_span - 0.3)
            if i / lat_span - 0.3 > 0 and over_mesh_i != target_mesh_i:
                edges.append([over_mesh_i * 320 + target_mesh_j, i * 1440 + j])
                cur_node = id2position(over_mesh_i * 320 + target_mesh_j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_i = int(i / lat_span + 0.3)
            if i / lat_span + 0.3 < 128 and over_mesh_i != target_mesh_i:
                edges.append([over_mesh_i * 320 + target_mesh_j, i * 1440 + j])
                cur_node = id2position(over_mesh_i * 320 + target_mesh_j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_j = int(j / lon_span - 0.3)
            if j / lon_span - 0.3 < 0:
                edges.append([target_mesh_i * 320 + 319, i * 1440 + j])
                cur_node = id2position(target_mesh_i * 320 + 319, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            elif over_mesh_j != target_mesh_j:
                edges.append([target_mesh_i * 320 + over_mesh_j, i * 1440 + j])
                cur_node = id2position(target_mesh_i * 320 + over_mesh_j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

            over_mesh_j = int(j / lon_span + 0.3)
            if j / lon_span + 0.3 > 320:
                edges.append([target_mesh_i * 320, i * 1440 + j])
                cur_node = id2position(target_mesh_i * 320, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)
            elif over_mesh_j != target_mesh_j:
                edges.append([target_mesh_i * 320 + over_mesh_j, i * 1440 + j])
                cur_node = id2position(target_mesh_i * 320 + over_mesh_j, 128, 320)
                tmp_attr = [target_node[k] - cur_node[k] for k in range(3)]
                edge_attrs.append([np.sqrt(np.sum(np.square(tmp_attr)))] + tmp_attr)

    return edges, edge_attrs


## 特征抽取
def fetch_time_features(cursor_time):

    year_hours = (datetime.date(cursor_time.year + 1, 1, 1) - datetime.date(cursor_time.year, 1, 1)).days * 24
    next_year_hours = (datetime.date(cursor_time.year + 2, 1, 1) - datetime.date(cursor_time.year + 1, 1, 1)).days * 24

    cur_hour = (cursor_time - datetime.datetime(cursor_time.year, 1, 1)) / datetime.timedelta(hours=1)
    time_features = []
    for j in range(1440):
        # local time
        local_hour = cur_hour + j * 24 / 1440
        if local_hour > year_hours:
            tr = (local_hour - year_hours) / next_year_hours
        else:
            tr = local_hour / year_hours

        time_features.append([[np.sin(2 * np.pi * tr), np.cos(2 * np.pi * tr)]] * 720)

    return np.transpose(np.asarray(time_features), [1, 0, 2])


def fetch_constant_features():
    constant_features = []
    for i in range(720):
        tmp = []
        for j in range(1440):
            tmp.append(id2position(i * 1440 + j, 720, 1440))
        constant_features.append(tmp)
    return np.asarray(constant_features)