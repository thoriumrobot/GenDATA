// Source-based slice around line 115
// Method: <com.google.common.graph.GraphBuilder: ImmutableGraph immutable()>

   * Returns an {@link ImmutableGraph.Builder} with the properties of this {@link GraphBuilder}.
   *
   * <p>The returned builder can be used for populating an {@link ImmutableGraph}.
   *
   * <p>Note that the returned builder will always have {@link #incidentEdgeOrder} set to {@link
   * ElementOrder#stable()}, regardless of the value that was set in this builder.
   *
   * @since 28.0
   */
  public <N1 extends N> ImmutableGraph.Builder<N1> immutable() {
    GraphBuilder<N1> castBuilder = cast();
    return new ImmutableGraph.Builder<>(castBuilder);
  }

  /**
   * Specifies whether the graph will allow self-loops (edges that connect a node to itself).
   * Attempting to add a self-loop to a graph that does not allow them will throw an {@link
   * UnsupportedOperationException}.
   *
   * <p>The default value is {@code false}.
