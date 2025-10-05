// Source-based slice around line 197
// Method: <com.google.common.graph.AbstractBaseGraph: Set nodePairInvalidatableSet(Set,N,N)>

  protected final boolean isOrderingCompatible(EndpointPair<?> endpoints) {
    return endpoints.isOrdered() == this.isDirected();
  }

  protected final <T> Set<T> nodeInvalidatableSet(Set<T> set, N node) {
    return InvalidatableSet.of(
        set, () -> nodes().contains(node), () -> String.format(NODE_REMOVED_FROM_GRAPH, node));
  }

  protected final <T> Set<T> nodePairInvalidatableSet(Set<T> set, N nodeU, N nodeV) {
    return InvalidatableSet.of(
        set,
        () -> nodes().contains(nodeU) && nodes().contains(nodeV),
        () -> String.format(NODE_PAIR_REMOVED_FROM_GRAPH, nodeU, nodeV));
  }
}
