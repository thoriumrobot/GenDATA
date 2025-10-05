// Source-based slice around line 87
// Method: <com.google.common.graph.GraphBuilder: GraphBuilder undirected()>

    super(directed);
  }

  /** Returns a {@link GraphBuilder} for building directed graphs. */
  public static GraphBuilder<Object> directed() {
    return new GraphBuilder<>(true);
  }

  /** Returns a {@link GraphBuilder} for building undirected graphs. */
  public static GraphBuilder<Object> undirected() {
    return new GraphBuilder<>(false);
  }

  /**
   * Returns a {@link GraphBuilder} initialized with all properties queryable from {@code graph}.
   *
   * <p>The "queryable" properties are those that are exposed through the {@link Graph} interface,
   * such as {@link Graph#isDirected()}. Other properties, such as {@link #expectedNodeCount(int)},
   * are not set in the new builder.
   */
