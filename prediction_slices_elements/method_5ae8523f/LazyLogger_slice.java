// Source-based slice around line 33
// Method: <com.google.common.util.concurrent.LazyLogger: Logger get()>

  private final Object lock = new Object();

  private final String loggerName;
  private volatile @Nullable Logger logger;

  LazyLogger(Class<?> ownerOfLogger) {
    this.loggerName = ownerOfLogger.getName();
  }

  Logger get() {
    /*
     * We use double-checked locking. We could the try racy single-check idiom, but that would
     * depend on Logger to not contain mutable state.
     *
     * We could use Suppliers.memoizingSupplier here, but I micro-optimized to this implementation
     * to avoid the extra class for the lambda (and maybe more for memoizingSupplier itself) and the
     * indirection.
     *
     * One thing to *avoid* is a change to make each Logger user use memoizingSupplier directly:
     * That may introduce an extra class for each lambda (currently a dozen).
