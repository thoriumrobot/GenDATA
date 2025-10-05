// Source-based slice around line 32
// Method: <com.google.common.graph.ForwardingValueGraph: ValueGraph delegate()>

/**
 * A class to allow {@link ValueGraph} implementations to be backed by a provided delegate. This is
 * not currently planned to be released as a general-purpose forwarding class.
 *
 * @author James Sexton
 * @author Joshua O'Madadhain
 */
abstract class ForwardingValueGraph<N, V> extends AbstractValueGraph<N, V> {

  abstract ValueGraph<N, V> delegate();

  @Override
  public Set<N> nodes() {
    return delegate().nodes();
  }

  /**
   * Defer to {@link AbstractValueGraph#edges()} (based on {@link #successors(Object)}) for full
   * edges() implementation.
   */
