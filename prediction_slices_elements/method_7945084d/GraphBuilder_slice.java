// Source-based slice around line 82
// Method: <com.google.common.graph.GraphBuilder: GraphBuilder directed()>

@DoNotMock
public final class GraphBuilder<N> extends AbstractGraphBuilder<N> {

  /** Creates a new instance with the specified edge directionality. */
  private GraphBuilder(boolean directed) {
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
