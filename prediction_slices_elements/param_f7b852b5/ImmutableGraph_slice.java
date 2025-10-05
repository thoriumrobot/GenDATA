// Source-based slice around line 97
// Method: <com.google.common.graph.ImmutableGraph: GraphConnections connectionsOf(Graph,N)>

    // input nodes are sorted.
    ImmutableMap.Builder<N, GraphConnections<N, Presence>> nodeConnections = ImmutableMap.builder();
    for (N node : graph.nodes()) {
      nodeConnections.put(node, connectionsOf(graph, node));
    }
    return nodeConnections.buildOrThrow();
  }

  @SuppressWarnings("unchecked")
  private static <N> GraphConnections<N, Presence> connectionsOf(Graph<N> graph, N node) {
    Function<N, Presence> edgeValueFn =
        (Function<N, Presence>) Functions.constant(Presence.EDGE_EXISTS);
    return graph.isDirected()
        ? DirectedGraphConnections.ofImmutable(node, graph.incidentEdges(node), edgeValueFn)
        : UndirectedGraphConnections.ofImmutable(
            Maps.asMap(graph.adjacentNodes(node), edgeValueFn));
  }

  @Override
  BaseGraph<N> delegate() {
