// Source-based slice around line 149
// Method: <com.google.common.graph.GraphBuilder: GraphBuilder nodeOrder(ElementOrder)>

    this.expectedNodeCount = Optional.of(checkNonNegative(expectedNodeCount));
    return this;
  }

  /**
   * Specifies the order of iteration for the elements of {@link Graph#nodes()}.
   *
   * <p>The default value is {@link ElementOrder#insertion() insertion order}.
   */
  public <N1 extends N> GraphBuilder<N1> nodeOrder(ElementOrder<N1> nodeOrder) {
    GraphBuilder<N1> newBuilder = cast();
    newBuilder.nodeOrder = checkNotNull(nodeOrder);
    return newBuilder;
  }

  /**
   * Specifies the order of iteration for the elements of {@link Graph#edges()}, {@link
   * Graph#adjacentNodes(Object)}, {@link Graph#predecessors(Object)}, {@link
   * Graph#successors(Object)} and {@link Graph#incidentEdges(Object)}.
   *
