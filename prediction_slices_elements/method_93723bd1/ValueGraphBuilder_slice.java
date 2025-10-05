// Source-based slice around line 84
// Method: <com.google.common.graph.ValueGraphBuilder: ValueGraphBuilder directed()>

@Beta
public final class ValueGraphBuilder<N, V> extends AbstractGraphBuilder<N> {

  /** Creates a new instance with the specified edge directionality. */
  private ValueGraphBuilder(boolean directed) {
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
