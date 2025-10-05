// Source-based slice around line 182
// Method: com.google.common.base.FinalizableReferenceQueue.logger

   *
   * If any of this fails along the way, we fall back to loading Finalizer directly in the
   * application class loader.
   *
   * NOTE: The tests for this behavior (FinalizableReferenceQueueClassLoaderUnloadingTest) fail
   * strangely when run in JDK 9. We are considering this a known issue. Please see
   * https://github.com/google/guava/issues/3086 for more information.
   */

  private static final Logger logger = Logger.getLogger(FinalizableReferenceQueue.class.getName());

  private static final String FINALIZER_CLASS_NAME = "com.google.common.base.internal.Finalizer";

  /** Reference to Finalizer.startFinalizer(). */
  private static final Method startFinalizer;

  static {
    Class<?> finalizer =
        loadFinalizer(new SystemLoader(), new DecoupledLoader(), new DirectLoader());
    startFinalizer = getStartFinalizer(finalizer);
