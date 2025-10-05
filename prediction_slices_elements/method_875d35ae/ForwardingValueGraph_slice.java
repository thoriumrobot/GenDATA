// Source-based slice around line 35
// Method: <com.google.common.graph.ForwardingValueGraph: Set nodes()>

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
  @Override
  protected long edgeCount() {
    return delegate().edges().size();
