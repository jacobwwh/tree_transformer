#enable top-down propagation with dgl message passing

from dgl import backend as F
from dgl import traversal as trv
from dgl.heterograph import DGLHeteroGraph, _create_compute_graph
from dgl._ffi.function import _init_api
from dgl.base import ALL, SLICE_FULL, NTYPE, NID, ETYPE, EID, is_all, DGLError, dgl_warning
from dgl import core
from dgl import graph_index
from dgl import heterograph_index
from dgl import utils
from dgl import backend as F
from dgl.frame import Frame
from dgl.view import HeteroNodeView, HeteroNodeDataView, HeteroEdgeView, HeteroEdgeDataView
from dgl import function as fn
from dgl.udf import NodeBatch, EdgeBatch
from dgl import ops
from dgl.core import is_builtin, invoke_gspmm, invoke_gsddmm, invoke_edge_udf, invoke_node_udf, _bucketing
import numpy as np

def prop_nodes_topdown(graph,
                    message_func,
                    reduce_func,
                    reverse=False,
                    apply_node_func=None):
    assert isinstance(graph, DGLHeteroGraph), \
        'DGLGraph is deprecated, Please use DGLHeteroGraph'
    assert len(graph.canonical_etypes) == 1, \
        'prop_nodes_topo only support homogeneous graph'
    # TODO(murphy): Graph traversal currently is only supported on
    # CPP graphs. Move graph to CPU as a workaround,
    # which should be fixed in the future.
    nodes_gen = trv.topological_nodes_generator(graph.cpu(), reverse)
    nodes_gen = [F.copy_to(frontier, graph.device) for frontier in nodes_gen]
    nodes_gen=nodes_gen[::-1]
    prop_nodes(graph, nodes_gen, message_func, reduce_func, apply_node_func)

def prop_nodes(graph,nodes_generator,
               message_func,
               reduce_func,
               apply_node_func=None,
               etype=None):

    for node_frontier in nodes_generator:
        #print('frontier',node_frontier)
        pull(graph,node_frontier, message_func, reduce_func, apply_node_func, etype=etype)

def pull(graph,v,
         message_func,
         reduce_func,
         apply_node_func=None,
         etype=None,
         inplace=False):

    if inplace:
        raise DGLError('The `inplace` option is removed in v0.5.')
    v = utils.prepare_tensor(graph, v, 'v')
    if len(v) == 0:
        # no computation
        return
    etid = graph.get_etype_id(etype)
    _, dtid = graph._graph.metagraph.find_edge(etid)
    etype = graph.canonical_etypes[etid]
    g = graph if etype is None else graph[etype]
    # call message passing on subgraph
    src, dst, eid = g.in_edges(v, form='all')
    #print('src-dst',src,dst)
    ndata = message_passing(graph,_create_compute_graph(g, src, dst, eid, v),
                                 message_func, reduce_func, apply_node_func)
    graph._set_n_repr(dtid, v, ndata)

def message_passing(graph,g, mfunc, rfunc, afunc):
    if g.number_of_edges() == 0:
        # No message passing is triggered.
        ndata = {}
    elif (is_builtin(mfunc) and is_builtin(rfunc) and
          getattr(ops, '{}_{}'.format(mfunc.name, rfunc.name), None) is not None):
        # invoke fused message passing
        ndata = invoke_gspmm(g, mfunc, rfunc)
    else:
        # invoke message passing in two separate steps
        # message phase
        if is_builtin(mfunc):
            msgdata = invoke_gsddmm(g, mfunc)
        else:
            orig_eid = g.edata.get(EID, None)
            msgdata = invoke_edge_udf(g, ALL, g.canonical_etypes[0], mfunc, orig_eid=orig_eid)
        # reduce phase
        if is_builtin(rfunc):
            msg = rfunc.msg_field
            ndata = invoke_gspmm(g, fn.copy_e(msg, msg), rfunc, edata=msgdata)
        else:
            orig_nid = g.dstdata.get(NID, None)
            ndata = invoke_udf_reduce(graph,g, rfunc, msgdata, orig_nid=orig_nid)
    # apply phase
    if afunc is not None:
        for k, v in g.dstdata.items():   # include original node features
            if k not in ndata:
                ndata[k] = v
        orig_nid = g.dstdata.get(NID, None)
        ndata = invoke_node_udf(g, ALL, g.dsttypes[0], afunc, ndata=ndata, orig_nid=orig_nid)
    return ndata

def invoke_udf_reduce(wholegraph,graph, func, msgdata, *, orig_nid=None):
    degs = graph.in_degrees()
    nodes = graph.dstnodes()
    if orig_nid is None:
        orig_nid = nodes
    ntype = graph.dsttypes[0]
    ntid = graph.get_ntype_id_from_dst(ntype)
    dstdata = graph._node_frames[ntid]
    msgdata = Frame(msgdata)

    # degree bucketing
    unique_degs, bucketor = _bucketing(degs)
    bkt_rsts = []
    bkt_nodes = []
    for deg, node_bkt, orig_nid_bkt in zip(unique_degs, bucketor(nodes), bucketor(orig_nid)):
        if deg == 0:
            # skip reduce function for zero-degree nodes
            continue
        bkt_nodes.append(node_bkt)
        ndata_bkt = dstdata.subframe(node_bkt)

        # order the incoming edges per node by edge ID
        eid_bkt = F.zerocopy_to_numpy(graph.in_edges(node_bkt, form='eid'))
        assert len(eid_bkt) == deg * len(node_bkt)
        eid_bkt = np.sort(eid_bkt.reshape((len(node_bkt), deg)), 1)
        eid_bkt = F.zerocopy_from_numpy(eid_bkt.flatten())

        msgdata_bkt = msgdata.subframe(eid_bkt)
        # reshape all msg tensors to (num_nodes_bkt, degree, feat_size)
        maildata = {}
        for k, msg in msgdata_bkt.items():
            newshape = (len(node_bkt), deg) + F.shape(msg)[1:]
            maildata[k] = F.reshape(msg, newshape)
        # invoke udf
        nbatch = NodeBatch(graph, orig_nid_bkt, ntype, ndata_bkt, msgs=maildata)
        bkt_rsts.append(func(wholegraph, nbatch))

    # prepare a result frame
    retf = Frame(num_rows=len(nodes))
    retf._initializers = dstdata._initializers
    retf._default_initializer = dstdata._default_initializer

    # merge bucket results and write to the result frame
    if len(bkt_rsts) != 0:  # if all the nodes have zero degree, no need to merge results.
        merged_rst = {}
        for k in bkt_rsts[0].keys():
            merged_rst[k] = F.cat([rst[k] for rst in bkt_rsts], dim=0)
        merged_nodes = F.cat(bkt_nodes, dim=0)
        retf.update_row(merged_nodes, merged_rst)

    return retf
