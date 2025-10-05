// Source-based slice around line 120
// Method: com.google.common.collect.Cut.serialVersionUID

  /*
   * The implementation neither produces nor consumes any non-null instance of type C, so
   * casting the type parameter is safe.
   */
  @SuppressWarnings("unchecked")
  static <C extends Comparable> Cut<C> belowAll() {
    return (Cut<C>) BelowAll.INSTANCE;
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;

  private static final class BelowAll extends Cut<Comparable<?>> {
    private static final BelowAll INSTANCE = new BelowAll();

    private BelowAll() {
      /*
       * No code ever sees this bogus value for `endpoint`: This class overrides both methods that
       * use the `endpoint` field, compareTo() and endpoint(). Additionally, the main implementation
       * of Cut.compareTo checks for belowAll before reading accessing `endpoint` on another Cut
       * instance.
