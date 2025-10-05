// Source-based slice around line 166
// Method: <com.google.common.graph.StandardMutableNetwork: NetworkConnections newConnections()>

    NetworkConnections<N, E> connectionsU = requireNonNull(nodeConnections.get(nodeU));
    N nodeV = connectionsU.adjacentNode(edge);
    NetworkConnections<N, E> connectionsV = requireNonNull(nodeConnections.get(nodeV));
    connectionsU.removeOutEdge(edge);
    connectionsV.removeInEdge(edge, allowsSelfLoops() && nodeU.equals(nodeV));
    edgeToReferenceNode.remove(edge);
    return true;
  }

  private NetworkConnections<N, E> newConnections() {
    return isDirected()
        ? allowsParallelEdges()
            ? DirectedMultiNetworkConnections.<N, E>of()
            : DirectedNetworkConnections.<N, E>of()
        : allowsParallelEdges()
            ? UndirectedMultiNetworkConnections.<N, E>of()
            : UndirectedNetworkConnections.<N, E>of();
  }
}
