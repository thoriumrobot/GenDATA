// Source-based slice around line 169
// Method: <com.google.common.graph.StandardNetwork: Set predecessors(N)>

    return nodeInvalidatableSet(checkedConnections(node).inEdges(), node);
  }

  @Override
  public Set<E> outEdges(N node) {
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
