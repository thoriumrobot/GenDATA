// Source-based slice around line 114
// Method: <com.google.common.graph.StandardValueGraph: Set successors(N)>

    return nodeInvalidatableSet(checkedConnections(node).adjacentNodes(), node);
  }

  @Override
  public Set<N> predecessors(N node) {
    return nodeInvalidatableSet(checkedConnections(node).predecessors(), node);
  }

  @Override
  public Set<N> successors(N node) {
    return nodeInvalidatableSet(checkedConnections(node).successors(), node);
  }

  @Override
  public Set<EndpointPair<N>> incidentEdges(N node) {
    GraphConnections<N, V> connections = checkedConnections(node);
    IncidentEdgeSet<N> incident =
        new IncidentEdgeSet<N>(this, node) {
          @Override
          public Iterator<EndpointPair<N>> iterator() {
