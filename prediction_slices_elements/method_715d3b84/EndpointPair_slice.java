// Source-based slice around line 108
// Method: <com.google.common.graph.EndpointPair: N adjacentNode(N)>

    return nodeV;
  }

  /**
   * Returns the node that is adjacent to {@code node} along the origin edge.
   *
   * @throws IllegalArgumentException if this {@link EndpointPair} does not contain {@code node}
   * @since 20.0 (but the argument type was changed from {@code Object} to {@code N} in 31.0)
   */
  public final N adjacentNode(N node) {
    if (node.equals(nodeU)) {
      return nodeV;
    } else if (node.equals(nodeV)) {
      return nodeU;
    } else {
      throw new IllegalArgumentException("EndpointPair " + this + " does not contain node " + node);
    }
  }

  /**
