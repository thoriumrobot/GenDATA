// Source-based slice around line 36
// Method: <com.google.common.graph.AbstractGraph: boolean equals(Object)>

 * @param <N> Node parameter type
 * @since 20.0
 */
@Beta
public abstract class AbstractGraph<N> extends AbstractBaseGraph<N> implements Graph<N> {
  /** Constructor for use by subclasses. */
  public AbstractGraph() {}

  @Override
  public final boolean equals(@Nullable Object obj) {
    if (obj == this) {
      return true;
    }
    if (!(obj instanceof Graph)) {
      return false;
    }
    Graph<?> other = (Graph<?>) obj;

    return isDirected() == other.isDirected()
        && nodes().equals(other.nodes())
