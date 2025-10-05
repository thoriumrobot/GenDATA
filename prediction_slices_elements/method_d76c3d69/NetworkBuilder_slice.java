// Source-based slice around line 92
// Method: <com.google.common.graph.NetworkBuilder: NetworkBuilder undirected()>

    super(directed);
  }

  /** Returns a {@link NetworkBuilder} for building directed networks. */
  public static NetworkBuilder<Object, Object> directed() {
    return new NetworkBuilder<>(true);
  }

  /** Returns a {@link NetworkBuilder} for building undirected networks. */
  public static NetworkBuilder<Object, Object> undirected() {
    return new NetworkBuilder<>(false);
  }

  /**
   * Returns a {@link NetworkBuilder} initialized with all properties queryable from {@code
   * network}.
   *
   * <p>The "queryable" properties are those that are exposed through the {@link Network} interface,
   * such as {@link Network#isDirected()}. Other properties, such as {@link
   * #expectedNodeCount(int)}, are not set in the new builder.
