// Source-based slice around line 178
// Method: <com.google.common.graph.StandardNetwork: NetworkConnections checkedConnections(N)>

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
  }

  final N checkedReferenceNode(E edge) {
    N referenceNode = edgeToReferenceNode.get(edge);
