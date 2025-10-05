// Source-based slice around line 187
// Method: <com.google.common.graph.StandardMutableValueGraph: GraphConnections newConnections()>

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
