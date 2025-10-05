// Source-based slice around line 24
// Method: com.google.common.util.concurrent.LazyLogger.lock

package com.google.common.util.concurrent;

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
