// Source-based slice around line 98
// Method: <com.google.common.graph.GraphBuilder: GraphBuilder from(Graph)>

  }

  /**
   * Returns a {@link GraphBuilder} initialized with all properties queryable from {@code graph}.
   *
   * <p>The "queryable" properties are those that are exposed through the {@link Graph} interface,
   * such as {@link Graph#isDirected()}. Other properties, such as {@link #expectedNodeCount(int)},
   * are not set in the new builder.
   */
  public static <N> GraphBuilder<N> from(Graph<N> graph) {
    return new GraphBuilder<N>(graph.isDirected())
        .allowsSelfLoops(graph.allowsSelfLoops())
        .nodeOrder(graph.nodeOrder())
        .incidentEdgeOrder(graph.incidentEdgeOrder());
  }

  /**
   * Returns an {@link ImmutableGraph.Builder} with the properties of this {@link GraphBuilder}.
   *
   * <p>The returned builder can be used for populating an {@link ImmutableGraph}.
