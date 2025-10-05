/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.util.logging;

import org.checkerframework.checker.initialization.qual.UnderInitialization;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signature.qual.BinaryName;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.io.*;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.time.Clock;
import java.util.function.Predicate;
import static jdk.internal.logger.SurrogateLogger.isFilteredFrame;

@AnnotatedFor({ "index", "interning", "nullness" })
@UsesObjectEquals
public class LogRecord implements java.io.Serializable {

    public LogRecord(Level level, @Nullable String msg) {
    }

    @Nullable
    public String getLoggerName();

    public void setLoggerName(@Nullable String name);

    @Nullable
    public ResourceBundle getResourceBundle();

    public void setResourceBundle(@Nullable ResourceBundle bundle);

    @Nullable
    @BinaryName
    public String getResourceBundleName();

    public void setResourceBundleName(@Nullable String name);

    public Level getLevel();

    public void setLevel(Level level);

    public long getSequenceNumber();

    public void setSequenceNumber(long seq);

    @Nullable
    public String getSourceClassName();

    public void setSourceClassName(@Nullable String sourceClassName);

    @Nullable
    public String getSourceMethodName();

    public void setSourceMethodName(@Nullable String sourceMethodName);

    @Nullable
    public String getMessage();

    public void setMessage(@Nullable String message);

    @Nullable
    public Object @Nullable [] getParameters();

    public void setParameters(@Nullable Object @Nullable [] parameters);

    @Deprecated()
    public int getThreadID();

    @Deprecated()
    public void setThreadID(int threadID);

    public long getLongThreadID();

    public LogRecord setLongThreadID(long longThreadID);

    public long getMillis();

    @Deprecated
    public void setMillis(long millis);

    public Instant getInstant();

    public void setInstant(Instant instant);

    @Nullable
    public Throwable getThrown();

    public void setThrown(@Nullable Throwable thrown);

    @SuppressWarnings("removal")
    static final class CallerFinder implements Predicate<StackWalker.StackFrame> {

        Optional<StackWalker.StackFrame> get();

        @Override
        public boolean test(StackWalker.StackFrame t);
    }
}
