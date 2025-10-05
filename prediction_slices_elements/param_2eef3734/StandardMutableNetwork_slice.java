// Source-based slice around line 78
// Method: <com.google.common.graph.StandardMutableNetwork: boolean addEdge(N,N,E)>

  @CanIgnoreReturnValue
  private NetworkConnections<N, E> addNodeInternal(N node) {
    NetworkConnections<N, E> connections = newConnections();
    checkState(nodeConnections.put(node, connections) == null);
    return connections;
  }

  @Override
  @CanIgnoreReturnValue
  public boolean addEdge(N nodeU, N nodeV, E edge) {
    checkNotNull(nodeU, "nodeU");
    checkNotNull(nodeV, "nodeV");
    checkNotNull(edge, "edge");

    if (containsEdge(edge)) {
      EndpointPair<N> existingIncidentNodes = incidentNodes(edge);
      EndpointPair<N> newIncidentNodes = EndpointPair.of(this, nodeU, nodeV);
      checkArgument(
          existingIncidentNodes.equals(newIncidentNodes),
          REUSING_EDGE,
