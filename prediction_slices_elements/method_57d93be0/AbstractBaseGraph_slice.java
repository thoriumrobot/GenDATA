// Source-based slice around line 107
// Method: <com.google.common.graph.AbstractBaseGraph: Set incidentEdges(N)>

    };
  }

  @Override
  public ElementOrder<N> incidentEdgeOrder() {
    return ElementOrder.unordered();
  }

  @Override
  public Set<EndpointPair<N>> incidentEdges(N node) {
    checkNotNull(node);
    checkArgument(nodes().contains(node), "Node %s is not an element of this graph.", node);
    IncidentEdgeSet<N> incident =
        new IncidentEdgeSet<N>(this, node) {
          @Override
          public UnmodifiableIterator<EndpointPair<N>> iterator() {
            if (graph.isDirected()) {
              return Iterators.unmodifiableIterator(
                  Iterators.concat(
                      Iterators.transform(
