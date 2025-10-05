/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.util.logging;

    @Positive
import org.checkerframework.checker.initialization.qual.UnderInitialization;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signature.qual.BinaryName;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.time.Instant;
    @Positive
import java.util.*;
    @Positive
import java.util.concurrent.atomic.AtomicLong;
    @Positive
import java.io.*;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.time.Clock;
    @Positive
import java.util.function.Predicate;
    @Positive
import static jdk.internal.logger.SurrogateLogger.isFilteredFrame;

    @Positive
@AnnotatedFor({ "index", "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class LogRecord implements java.io.Serializable {

    @Positive
    public LogRecord(Level level, @Nullable String msg) {
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public String getLoggerName();

    @Positive
    public void setLoggerName(@Nullable String name);

    @Positive
    @Nullable
    @Positive
    public ResourceBundle getResourceBundle();

    @Positive
    public void setResourceBundle(@Nullable ResourceBundle bundle);

    @Positive
    @Nullable
    @Positive
    @BinaryName
    @Positive
    public String getResourceBundleName();

    @Positive
    public void setResourceBundleName(@Nullable String name);

    @Positive
    public Level getLevel();

    @Positive
    public void setLevel(Level level);

    @Positive
    public long getSequenceNumber();

    @Positive
    public void setSequenceNumber(long seq);

    @Positive
    @Nullable
    @Positive
    public String getSourceClassName();

    @Positive
    public void setSourceClassName(@Nullable String sourceClassName);

    @Positive
    @Nullable
    @Positive
    public String getSourceMethodName();

    @Positive
    public void setSourceMethodName(@Nullable String sourceMethodName);

    @Positive
    @Nullable
    @Positive
    public String getMessage();

    @Positive
    public void setMessage(@Nullable String message);

    @Positive
    @Nullable
    @Positive
    public Object @Nullable [] getParameters();

    @Positive
    public void setParameters(@Nullable Object @Nullable [] parameters);

    @Positive
    @Deprecated()
    @Positive
    public int getThreadID();

    @Positive
    @Deprecated()
    @Positive
    public void setThreadID(int threadID);

    @Positive
    public long getLongThreadID();

    @Positive
    public LogRecord setLongThreadID(long longThreadID);

    @Positive
    public long getMillis();

    @Positive
    @Deprecated
    @Positive
    public void setMillis(long millis);

    @Positive
    public Instant getInstant();

    @Positive
    public void setInstant(Instant instant);

    @Positive
    @Nullable
    @Positive
    public Throwable getThrown();

    @Positive
    public void setThrown(@Nullable Throwable thrown);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    static final class CallerFinder implements Predicate<StackWalker.StackFrame> {

    @Positive
        Optional<StackWalker.StackFrame> get();

    @Positive
        @Override
    @Positive
        public boolean test(StackWalker.StackFrame t);
    @Positive
    }
    @Positive
}
