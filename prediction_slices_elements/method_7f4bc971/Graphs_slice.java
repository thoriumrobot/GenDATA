// Source-based slice around line 649
// Method: <com.google.common.graph.Graphs: int checkNonNegative(int)>

    }
    for (E edge : network.edges()) {
      EndpointPair<N> endpointPair = network.incidentNodes(edge);
      copy.addEdge(endpointPair.nodeU(), endpointPair.nodeV(), edge);
    }
    return copy;
  }

  @CanIgnoreReturnValue
  static int checkNonNegative(int value) {
    checkArgument(value >= 0, "Not true that %s is non-negative.", value);
    return value;
  }

  @CanIgnoreReturnValue
  static long checkNonNegative(long value) {
    checkArgument(value >= 0, "Not true that %s is non-negative.", value);
    return value;
  }

