// Source-based slice around line 153
// Method: <com.google.common.graph.ValueGraphBuilder: ValueGraphBuilder nodeOrder(ElementOrder)>

    this.expectedNodeCount = Optional.of(checkNonNegative(expectedNodeCount));
    return this;
  }

  /**
   * Specifies the order of iteration for the elements of {@link Graph#nodes()}.
   *
   * <p>The default value is {@link ElementOrder#insertion() insertion order}.
   */
  public <N1 extends N> ValueGraphBuilder<N1, V> nodeOrder(ElementOrder<N1> nodeOrder) {
    ValueGraphBuilder<N1, V> newBuilder = cast();
    newBuilder.nodeOrder = checkNotNull(nodeOrder);
    return newBuilder;
  }

  /**
   * Specifies the order of iteration for the elements of {@link ValueGraph#edges()}, {@link
   * ValueGraph#adjacentNodes(Object)}, {@link ValueGraph#predecessors(Object)}, {@link
   * ValueGraph#successors(Object)} and {@link ValueGraph#incidentEdges(Object)}.
   *
