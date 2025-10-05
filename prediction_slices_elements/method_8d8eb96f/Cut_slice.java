// Source-based slice around line 223
// Method: <com.google.common.collect.Cut: Cut aboveAll()>


    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
  }

  /*
   * The implementation neither produces nor consumes any non-null instance of
   * type C, so casting the type parameter is safe.
   */
  @SuppressWarnings("unchecked")
  static <C extends Comparable> Cut<C> aboveAll() {
    return (Cut<C>) AboveAll.INSTANCE;
  }

  private static final class AboveAll extends Cut<Comparable<?>> {
    private static final AboveAll INSTANCE = new AboveAll();

    private AboveAll() {
      // For discussion of "", see BelowAll.
      super("");
    }
