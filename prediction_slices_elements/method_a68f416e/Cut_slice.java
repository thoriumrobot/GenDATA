// Source-based slice around line 116
// Method: <com.google.common.collect.Cut: Cut belowAll()>

  // Prevent "missing hashCode" warning by explicitly forcing subclasses implement it
  @Override
  public abstract int hashCode();

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
