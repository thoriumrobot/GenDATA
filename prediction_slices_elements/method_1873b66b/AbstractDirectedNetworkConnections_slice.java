// Source-based slice around line 60
// Method: <com.google.common.graph.AbstractDirectedNetworkConnections: Set adjacentNodes()>


  AbstractDirectedNetworkConnections(Map<E, N> inEdgeMap, Map<E, N> outEdgeMap, int selfLoopCount) {
    this.inEdgeMap = checkNotNull(inEdgeMap);
    this.outEdgeMap = checkNotNull(outEdgeMap);
    this.selfLoopCount = checkNonNegative(selfLoopCount);
    checkState(selfLoopCount <= inEdgeMap.size() && selfLoopCount <= outEdgeMap.size());
  }

  @Override
  public Set<N> adjacentNodes() {
    return Sets.union(predecessors(), successors());
  }

  @Override
  public Set<E> incidentEdges() {
    return new AbstractSet<E>() {
      @Override
      public UnmodifiableIterator<E> iterator() {
        Iterable<E> incidentEdges =
            (selfLoopCount == 0)
