// Source-based slice around line 632
// Method: <com.google.common.graph.Graphs: MutableNetwork copyOf(Network)>

      copy.putEdgeValue(
          edge.nodeU(),
          edge.nodeV(),
          requireNonNull(graph.edgeValueOrDefault(edge.nodeU(), edge.nodeV(), null)));
    }
    return copy;
  }

  /** Creates a mutable copy of {@code network} with the same nodes and edges. */
  public static <N, E> MutableNetwork<N, E> copyOf(Network<N, E> network) {
    MutableNetwork<N, E> copy =
        NetworkBuilder.from(network)
            .expectedNodeCount(network.nodes().size())
            .expectedEdgeCount(network.edges().size())
            .build();
    for (N node : network.nodes()) {
      copy.addNode(node);
    }
    for (E edge : network.edges()) {
      EndpointPair<N> endpointPair = network.incidentNodes(edge);
