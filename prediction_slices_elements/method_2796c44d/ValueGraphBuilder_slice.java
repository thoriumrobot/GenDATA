// Source-based slice around line 119
// Method: <com.google.common.graph.ValueGraphBuilder: ImmutableValueGraph immutable()>

   * ValueGraphBuilder}.
   *
   * <p>The returned builder can be used for populating an {@link ImmutableValueGraph}.
   *
   * <p>Note that the returned builder will always have {@link #incidentEdgeOrder} set to {@link
   * ElementOrder#stable()}, regardless of the value that was set in this builder.
   *
   * @since 28.0
   */
  public <N1 extends N, V1 extends V> ImmutableValueGraph.Builder<N1, V1> immutable() {
    ValueGraphBuilder<N1, V1> castBuilder = cast();
    return new ImmutableValueGraph.Builder<>(castBuilder);
  }

  /**
   * Specifies whether the graph will allow self-loops (edges that connect a node to itself).
   * Attempting to add a self-loop to a graph that does not allow them will throw an {@link
   * UnsupportedOperationException}.
   *
   * <p>The default value is {@code false}.
