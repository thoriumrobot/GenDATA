// Source-based slice around line 51
// Method: <com.google.common.graph.AbstractGraph: int hashCode()>

    }
    Graph<?> other = (Graph<?>) obj;

    return isDirected() == other.isDirected()
        && nodes().equals(other.nodes())
        && edges().equals(other.edges());
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
