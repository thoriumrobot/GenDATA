// Source-based slice around line 32
// Method: <com.google.common.graph.ForwardingGraph: Set nodes()>

 * currently planned to be released as a general-purpose forwarding class.
 *
 * @author James Sexton
 */
abstract class ForwardingGraph<N> extends AbstractGraph<N> {

  abstract BaseGraph<N> delegate();

  @Override
  public Set<N> nodes() {
    return delegate().nodes();
  }

  /**
   * Defer to {@link AbstractGraph#edges()} (based on {@link #successors(Object)}) for full edges()
   * implementation.
   */
  @Override
  protected long edgeCount() {
    return delegate().edges().size();
