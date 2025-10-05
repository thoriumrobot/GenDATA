// Source-based slice around line 26
// Method: com.google.common.util.concurrent.LazyLogger.loggerName

import com.google.common.annotations.GwtCompatible;
import java.util.logging.Logger;
import org.jspecify.annotations.Nullable;

/** A holder for a {@link Logger} that is initialized only when requested. */
@GwtCompatible
final class LazyLogger {
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
