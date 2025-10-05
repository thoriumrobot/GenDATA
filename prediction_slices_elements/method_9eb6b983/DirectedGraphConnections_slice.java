// Source-based slice around line 368
// Method: <com.google.common.graph.DirectedGraphConnections: Iterator incidentEdgeIterator(N)>


      @Override
      public boolean contains(@Nullable Object obj) {
        return isSuccessor(adjacentNodeValues.get(obj));
      }
    };
  }

  @Override
  public Iterator<EndpointPair<N>> incidentEdgeIterator(N thisNode) {
    checkNotNull(thisNode);

    Iterator<EndpointPair<N>> resultWithDoubleSelfLoop;
    if (orderedNodeConnections == null) {
      resultWithDoubleSelfLoop =
          Iterators.concat(
              Iterators.transform(
                  predecessors().iterator(),
                  (N predecessor) -> EndpointPair.ordered(predecessor, thisNode)),
              Iterators.transform(
