// Source-based slice around line 102
// Method: <com.google.common.graph.AbstractBaseGraph: ElementOrder incidentEdgeOrder()>

        EndpointPair<?> endpointPair = (EndpointPair<?>) obj;
        return isOrderingCompatible(endpointPair)
            && nodes().contains(endpointPair.nodeU())
            && successors((N) endpointPair.nodeU()).contains(endpointPair.nodeV());
      }
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
