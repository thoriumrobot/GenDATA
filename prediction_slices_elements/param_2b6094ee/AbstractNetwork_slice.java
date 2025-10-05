// Source-based slice around line 305
// Method: <com.google.common.graph.AbstractNetwork: Set nodeInvalidatableSet(Set,N)>

        set, () -> edges().contains(edge), () -> String.format(EDGE_REMOVED_FROM_GRAPH, edge));
  }

  /**
   * Returns a {@link Set} whose methods throw {@link IllegalStateException} when the given node is
   * not present in this network.
   *
   * @since 33.1.0
   */
  protected final <T> Set<T> nodeInvalidatableSet(Set<T> set, N node) {
    return InvalidatableSet.of(
        set, () -> nodes().contains(node), () -> String.format(NODE_REMOVED_FROM_GRAPH, node));
  }

  /**
   * Returns a {@link Set} whose methods throw {@link IllegalStateException} when either of the
   * given nodes is not present in this network.
   *
   * @since 33.1.0
   */
