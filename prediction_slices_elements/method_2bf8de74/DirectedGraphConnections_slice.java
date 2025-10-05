// Source-based slice around line 234
// Method: <com.google.common.graph.DirectedGraphConnections: Set adjacentNodes()>


    return new DirectedGraphConnections<>(
        adjacentNodeValues,
        orderedNodeConnectionsBuilder.build(),
        predecessorCount,
        successorCount);
  }

  @Override
  public Set<N> adjacentNodes() {
    if (orderedNodeConnections == null) {
      return Collections.unmodifiableSet(adjacentNodeValues.keySet());
    } else {
      return new AbstractSet<N>() {
        @Override
        public UnmodifiableIterator<N> iterator() {
          Iterator<NodeConnection<N>> nodeConnections = orderedNodeConnections.iterator();
          Set<N> seenNodes = new HashSet<>();
          return new AbstractIterator<N>() {
            @Override
