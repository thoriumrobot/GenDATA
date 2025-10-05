// Source-based slice around line 155
// Method: <com.google.common.graph.DirectedGraphConnections: DirectedGraphConnections of(ElementOrder)>

    this.adjacentNodeValues = checkNotNull(adjacentNodeValues);
    this.orderedNodeConnections = orderedNodeConnections;
    this.predecessorCount = checkNonNegative(predecessorCount);
    this.successorCount = checkNonNegative(successorCount);
    checkState(
        predecessorCount <= adjacentNodeValues.size()
            && successorCount <= adjacentNodeValues.size());
  }

  static <N, V> DirectedGraphConnections<N, V> of(ElementOrder<N> incidentEdgeOrder) {
    // We store predecessors and successors in the same map, so double the initial capacity.
    int initialCapacity = INNER_CAPACITY * 2;

    List<NodeConnection<N>> orderedNodeConnections;
    switch (incidentEdgeOrder.type()) {
      case UNORDERED:
        orderedNodeConnections = null;
        break;
      case STABLE:
        orderedNodeConnections = new ArrayList<>();
