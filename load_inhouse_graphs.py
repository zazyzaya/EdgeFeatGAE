import torch 
import json 

from tqdm import tqdm
from torch_geometric.data import Data 

# Last timestamp before malware in system
DATE_OF_EVIL = "2019-07-19T18:07:59.460425Z"

def pico_file_loader(fname, keep=['client', 'ts', 'service']):
    # Not really in the right schema to just use json.loads on the 
    # whole thing. Each line is its own json object
    with open(fname, 'r') as f:
        lines = f.read().split('\n')
        logs = [json.loads(l) for l in lines if len(l) > 1]

    # Filter out noisey logs. Only care about TGS kerb logs (for now)
    unflogs = [l for l in logs if 'request_type' in l.keys()]
    
    # Get rid of extranious data, and make sure required data exists
    logs = []
    for l in unflogs:
        try:
            logs.append({k:l[k] for k in keep})
        except KeyError as e:
            continue 

    return logs 

def pico_logs_to_graph(logs):
    '''
    (Useful) Kerb logs have the following structure:
                                   (where they access from)
    client:   USR (or) COMPUTER$ / INSTANCE . subdomain . subdomain (etc)

             (optional)    (the computer)                     (optional)
    service:  service   /  TOP LEVEL .    sub domain . (etc)  @  realm  

    Worth noting, service names for computers are in all caps, client names are.. varied.
    To be safe, capitalize everything
    '''
    tr_set = True
    tr_set_end = 1

    cl_to_id = {}  # Map of string names to numeric 0-|N|
    cl_cnt = 0         # Cur id

    service_to_id = {}
    sv_cnt = 0

    src = []        # User machine or name
    dst = []        # Service machine
    edge_feats = [] # The service

    for l in tqdm(logs, total=len(logs)):
        if tr_set:
            if l['ts'] == DATE_OF_EVIL:
                tr_set = False 
            else: 
                tr_set_end += 1

        # First parse out client 
        client = l['client'].split('/')[0] # Don't really care about instance 
        client = client.upper()
        
        if client in cl_to_id:
            src.append(cl_to_id[client])
        else:
            src.append(cl_cnt)
            cl_to_id[client] = cl_cnt
            cl_cnt += 1
        
        # Then parse out server & service 
        srv = l['service'].split('/')
        
        # As far as I can tell, just 2 cases:
        if len(srv) == 1:
            server = srv[0].upper()
            server = server.split('@')[0] # ignore realm if it exists
            feat = 'NONE'              # didn't want any service (I think this is a default login?)

        else:
            feat = srv[0].upper() 
            server = srv[1].split('.')[0].upper() # Only care about top-level (also slices out realm)

        # Add in id of server 
        if server in cl_to_id:
            dst.append(cl_to_id[server])
        else:
            dst.append(cl_cnt)
            cl_to_id[server] = cl_cnt
            cl_cnt += 1

        # Add edge feature (service rendered)
        if feat in service_to_id:
            edge_feats.append(service_to_id[feat])
        else:
            edge_feats.append(sv_cnt)
            service_to_id[feat] = sv_cnt 
            sv_cnt += 1

    # No node feats really
    x = torch.eye(cl_cnt)
    ei = torch.tensor([src, dst])
    tr_mask = torch.zeros(ei.size(1)).bool()
    tr_mask[:tr_set_end] = True
    
    # Make one-hot edge feats
    feats = torch.zeros((ei.size(1), max(edge_feats)+1))
    idx = torch.tensor([edge_feats]).T
    feats = feats.scatter(1, idx, 1)

    # Finally, return Data object
    data = Data(
        x=x, edge_index=ei, edge_attr=feats, 
        num_nodes=cl_cnt, tr_mask=tr_mask,
        node_map=cl_to_id
    )
    return data

import os 
def load_pico():
    F_LOC = '/mnt/raid0_24TB/datasets/pico/bro/'
    days = [os.path.join(F_LOC,d) for d in os.listdir(F_LOC)]
    days.sort()

    logs = []
    for d in days:
        kerb_logs = [os.path.join(d, l) for l in os.listdir(d) if 'kerb' in l]
        kerb_logs.sort()

        list_o_logs = [pico_file_loader(l) for l in kerb_logs]
        for l in list_o_logs:
            logs += l

    return pico_logs_to_graph(logs)

load_pico()