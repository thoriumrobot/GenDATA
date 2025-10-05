// Source-based slice around line 168
// Method: <com.google.common.graph.GraphBuilder: GraphBuilder incidentEdgeOrder(ElementOrder)>

   *
   * <p>The default value is {@link ElementOrder#unordered() unordered} for mutable graphs. For
   * immutable graphs, this value is ignored; they always have a {@link ElementOrder#stable()
   * stable} order.
   *
   * @throws IllegalArgumentException if {@code incidentEdgeOrder} is not either {@code
   *     ElementOrder.unordered()} or {@code ElementOrder.stable()}.
   * @since 29.0
   */
  public <N1 extends N> GraphBuilder<N1> incidentEdgeOrder(ElementOrder<N1> incidentEdgeOrder) {
    checkArgument(
        incidentEdgeOrder.type() == ElementOrder.Type.UNORDERED
            || incidentEdgeOrder.type() == ElementOrder.Type.STABLE,
        "The given elementOrder (%s) is unsupported. incidentEdgeOrder() only supports"
            + " ElementOrder.unordered() and ElementOrder.stable().",
        incidentEdgeOrder);
    GraphBuilder<N1> newBuilder = cast();
    newBuilder.incidentEdgeOrder = checkNotNull(incidentEdgeOrder);
    return newBuilder;
  }
