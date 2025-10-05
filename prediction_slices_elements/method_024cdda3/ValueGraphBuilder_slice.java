// Source-based slice around line 101
// Method: <com.google.common.graph.ValueGraphBuilder: ValueGraphBuilder from(ValueGraph)>


  /**
   * Returns a {@link ValueGraphBuilder} initialized with all properties queryable from {@code
   * graph}.
   *
   * <p>The "queryable" properties are those that are exposed through the {@link ValueGraph}
   * interface, such as {@link ValueGraph#isDirected()}. Other properties, such as {@link
   * #expectedNodeCount(int)}, are not set in the new builder.
   */
  public static <N, V> ValueGraphBuilder<N, V> from(ValueGraph<N, V> graph) {
    return new ValueGraphBuilder<N, V>(graph.isDirected())
        .allowsSelfLoops(graph.allowsSelfLoops())
        .nodeOrder(graph.nodeOrder())
        .incidentEdgeOrder(graph.incidentEdgeOrder());
  }

  /**
   * Returns an {@link ImmutableValueGraph.Builder} with the properties of this {@link
   * ValueGraphBuilder}.
   *
