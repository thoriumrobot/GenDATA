// Source-based slice around line 57
// Method: <com.google.common.graph.AbstractGraph: String toString()>

  }

  @Override
  public final int hashCode() {
    return edges().hashCode();
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
        + edges();
  }
}
