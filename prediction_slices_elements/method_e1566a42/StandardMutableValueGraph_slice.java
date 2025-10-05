// Source-based slice around line 182
// Method: <com.google.common.graph.StandardMutableValueGraph: V removeEdge(EndpointPair)>

    if (previousValue != null) {
      connectionsV.removePredecessor(nodeU);
      checkNonNegative(--edgeCount);
    }
    return previousValue;
  }

  @Override
  @CanIgnoreReturnValue
  public @Nullable V removeEdge(EndpointPair<N> endpoints) {
    validateEndpoints(endpoints);
    return removeEdge(endpoints.nodeU(), endpoints.nodeV());
  }

  private GraphConnections<N, V> newConnections() {
    return isDirected()
        ? DirectedGraphConnections.<N, V>of(incidentEdgeOrder)
        : UndirectedGraphConnections.<N, V>of(incidentEdgeOrder);
  }
}
