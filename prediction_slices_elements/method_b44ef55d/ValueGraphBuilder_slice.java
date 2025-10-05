// Source-based slice around line 89
// Method: <com.google.common.graph.ValueGraphBuilder: ValueGraphBuilder undirected()>

    super(directed);
  }

  /** Returns a {@link ValueGraphBuilder} for building directed graphs. */
  public static ValueGraphBuilder<Object, Object> directed() {
    return new ValueGraphBuilder<>(true);
  }

  /** Returns a {@link ValueGraphBuilder} for building undirected graphs. */
  public static ValueGraphBuilder<Object, Object> undirected() {
    return new ValueGraphBuilder<>(false);
  }

  /**
   * Returns a {@link ValueGraphBuilder} initialized with all properties queryable from {@code
   * graph}.
   *
   * <p>The "queryable" properties are those that are exposed through the {@link ValueGraph}
   * interface, such as {@link ValueGraph#isDirected()}. Other properties, such as {@link
   * #expectedNodeCount(int)}, are not set in the new builder.
