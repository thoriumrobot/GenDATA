// Source-based slice around line 174
// Method: <com.google.common.graph.StandardNetwork: Set successors(N)>

    return nodeInvalidatableSet(checkedConnections(node).outEdges(), node);
  }

  @Override
  public Set<N> predecessors(N node) {
    return nodeInvalidatableSet(checkedConnections(node).predecessors(), node);
  }

  @Override
  public Set<N> successors(N node) {
    return nodeInvalidatableSet(checkedConnections(node).successors(), node);
  }

  final NetworkConnections<N, E> checkedConnections(N node) {
    NetworkConnections<N, E> connections = nodeConnections.get(node);
    if (connections == null) {
      checkNotNull(node);
      throw new IllegalArgumentException(String.format(NODE_NOT_IN_GRAPH, node));
    }
    return connections;
