// Source-based slice around line 29
// Method: <com.google.common.graph.ForwardingGraph: BaseGraph delegate()>


/**
 * A class to allow {@link Graph} implementations to be backed by a {@link BaseGraph}. This is not
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
