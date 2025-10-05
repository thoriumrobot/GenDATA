// Source-based slice around line 87
// Method: <com.google.common.graph.NetworkBuilder: NetworkBuilder directed()>

  ElementOrder<? super E> edgeOrder = ElementOrder.insertion();
  Optional<Integer> expectedEdgeCount = Optional.absent();

  /** Creates a new instance with the specified edge directionality. */
  private NetworkBuilder(boolean directed) {
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
