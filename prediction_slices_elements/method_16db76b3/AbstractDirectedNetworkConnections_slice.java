// Source-based slice around line 65
// Method: <com.google.common.graph.AbstractDirectedNetworkConnections: Set incidentEdges()>

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
                ? Iterables.concat(inEdgeMap.keySet(), outEdgeMap.keySet())
                : Sets.union(inEdgeMap.keySet(), outEdgeMap.keySet());
        return Iterators.unmodifiableIterator(incidentEdges.iterator());
      }

