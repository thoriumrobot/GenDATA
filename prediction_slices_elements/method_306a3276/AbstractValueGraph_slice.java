// Source-based slice around line 143
// Method: <com.google.common.graph.AbstractValueGraph: String toString()>

  }

  @Override
  public final int hashCode() {
    return edgeValueMap(this).hashCode();
  }

  /** Returns a string representation of this graph. */
  @Override
  public String toString() {
    return "isDirected: "
        + isDirected()
        + ", allowsSelfLoops: "
        + allowsSelfLoops()
        + ", nodes: "
        + nodes()
        + ", edges: "
        + edgeValueMap(this);
  }

