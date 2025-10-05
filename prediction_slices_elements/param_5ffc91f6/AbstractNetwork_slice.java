// Source-based slice around line 316
// Method: <com.google.common.graph.AbstractNetwork: Set nodePairInvalidatableSet(Set,N,N)>

        set, () -> nodes().contains(node), () -> String.format(NODE_REMOVED_FROM_GRAPH, node));
  }

  /**
   * Returns a {@link Set} whose methods throw {@link IllegalStateException} when either of the
   * given nodes is not present in this network.
   *
   * @since 33.1.0
   */
  protected final <T> Set<T> nodePairInvalidatableSet(Set<T> set, N nodeU, N nodeV) {
    return InvalidatableSet.of(
        set,
        () -> nodes().contains(nodeU) && nodes().contains(nodeV),
        () -> String.format(NODE_PAIR_REMOVED_FROM_GRAPH, nodeU, nodeV));
  }

  private static <N, E> Map<E, EndpointPair<N>> edgeIncidentNodesMap(Network<N, E> network) {
    return Maps.asMap(network.edges(), network::incidentNodes);
  }
}
