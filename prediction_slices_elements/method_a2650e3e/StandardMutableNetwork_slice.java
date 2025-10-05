// Source-based slice around line 122
// Method: <com.google.common.graph.StandardMutableNetwork: boolean addEdge(EndpointPair,E)>

      connectionsV = addNodeInternal(nodeV);
    }
    connectionsV.addInEdge(edge, nodeU, isSelfLoop);
    edgeToReferenceNode.put(edge, nodeU);
    return true;
  }

  @Override
  @CanIgnoreReturnValue
  public boolean addEdge(EndpointPair<N> endpoints, E edge) {
    validateEndpoints(endpoints);
    return addEdge(endpoints.nodeU(), endpoints.nodeV(), edge);
  }

  @Override
  @CanIgnoreReturnValue
  public boolean removeNode(N node) {
    checkNotNull(node, "node");

    NetworkConnections<N, E> connections = nodeConnections.get(node);
