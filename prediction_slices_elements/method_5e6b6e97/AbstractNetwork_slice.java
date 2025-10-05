// Source-based slice around line 294
// Method: <com.google.common.graph.AbstractNetwork: Set edgeInvalidatableSet(Set,E)>

        + edgeIncidentNodesMap(this);
  }

  /**
   * Returns a {@link Set} whose methods throw {@link IllegalStateException} when the given edge is
   * not present in this network.
   *
   * @since 33.1.0
   */
  protected final <T> Set<T> edgeInvalidatableSet(Set<T> set, E edge) {
    return InvalidatableSet.of(
        set, () -> edges().contains(edge), () -> String.format(EDGE_REMOVED_FROM_GRAPH, edge));
  }

  /**
   * Returns a {@link Set} whose methods throw {@link IllegalStateException} when the given node is
   * not present in this network.
   *
   * @since 33.1.0
   */
