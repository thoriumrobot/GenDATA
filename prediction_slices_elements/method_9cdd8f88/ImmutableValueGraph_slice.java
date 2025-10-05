// Source-based slice around line 96
// Method: <com.google.common.graph.ImmutableValueGraph: GraphConnections connectionsOf(ValueGraph,N)>

    // whatever ordering the graph's nodes do, so ImmutableSortedMap is unnecessary even if the
    // input nodes are sorted.
    ImmutableMap.Builder<N, GraphConnections<N, V>> nodeConnections = ImmutableMap.builder();
    for (N node : graph.nodes()) {
      nodeConnections.put(node, connectionsOf(graph, node));
    }
    return nodeConnections.buildOrThrow();
  }

  private static <N, V> GraphConnections<N, V> connectionsOf(ValueGraph<N, V> graph, N node) {
    Function<N, V> successorNodeToValueFn =
        (N successorNode) ->
            // requireNonNull is safe because the endpoint pair comes from the graph.
            requireNonNull(graph.edgeValueOrDefault(node, successorNode, null));
    return graph.isDirected()
        ? DirectedGraphConnections.ofImmutable(
            node, graph.incidentEdges(node), successorNodeToValueFn)
        : UndirectedGraphConnections.ofImmutable(
            Maps.asMap(graph.adjacentNodes(node), successorNodeToValueFn));
  }
