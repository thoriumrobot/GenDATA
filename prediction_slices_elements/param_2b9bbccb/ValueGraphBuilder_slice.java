// Source-based slice around line 173
// Method: <com.google.common.graph.ValueGraphBuilder: ValueGraphBuilder incidentEdgeOrder(ElementOrder)>

   * <p>The default value is {@link ElementOrder#unordered() unordered} for mutable graphs. For
   * immutable graphs, this value is ignored; they always have a {@link ElementOrder#stable()
   * stable} order.
   *
   * @throws IllegalArgumentException if {@code incidentEdgeOrder} is not either {@code
   *     ElementOrder.unordered()} or {@code ElementOrder.stable()}.
   * @since 29.0
   */
  public <N1 extends N> ValueGraphBuilder<N1, V> incidentEdgeOrder(
      ElementOrder<N1> incidentEdgeOrder) {
    checkArgument(
        incidentEdgeOrder.type() == ElementOrder.Type.UNORDERED
            || incidentEdgeOrder.type() == ElementOrder.Type.STABLE,
        "The given elementOrder (%s) is unsupported. incidentEdgeOrder() only supports"
            + " ElementOrder.unordered() and ElementOrder.stable().",
        incidentEdgeOrder);
    ValueGraphBuilder<N1, V> newBuilder = cast();
    newBuilder.incidentEdgeOrder = checkNotNull(incidentEdgeOrder);
    return newBuilder;
  }
