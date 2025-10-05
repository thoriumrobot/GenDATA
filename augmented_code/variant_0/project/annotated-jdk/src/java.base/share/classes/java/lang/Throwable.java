/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1994, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.lang;

    @Positive
import org.checkerframework.checker.initialization.qual.UnknownInitialization;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.*;
    @Positive
import java.util.*;

    @Positive
@AnnotatedFor({ "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class Throwable implements Serializable {

    @Positive
    private static class SentinelHolder {

    @Positive
        public static final StackTraceElement STACK_TRACE_ELEMENT_SENTINEL;

    @Positive
        public static final StackTraceElement[] STACK_TRACE_SENTINEL;
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Throwable() {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Throwable(@Nullable String message) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Throwable(@Nullable String message, @Nullable Throwable cause) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    public Throwable(@Nullable Throwable cause) {
    @Positive
    }

    @Positive
    @SideEffectFree
    @Positive
    protected Throwable(@Nullable String message, @Nullable Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getMessage(@GuardSatisfied Throwable this);

    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    public String getLocalizedMessage(@GuardSatisfied Throwable this);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public synchronized Throwable getCause(@GuardSatisfied Throwable this);

    @Positive
    @UnknownInitialization
    @Positive
    public synchronized Throwable initCause(@UnknownInitialization Throwable this, @Nullable Throwable cause);

    @Positive
    final void setCause(Throwable t);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Throwable this);

    @Positive
    public void printStackTrace();

    @Positive
    public void printStackTrace(PrintStream s);

    @Positive
    public void printStackTrace(PrintWriter s);

    @Positive
    private abstract static class PrintStreamOrWriter {

    @Positive
        abstract Object lock();

    @Positive
        abstract void println(Object o);
    @Positive
    }

    @Positive
    private static class WrappedPrintStream extends PrintStreamOrWriter {

    @Positive
        Object lock();

    @Positive
        void println(Object o);
    @Positive
    }

    @Positive
    private static class WrappedPrintWriter extends PrintStreamOrWriter {

    @Positive
        Object lock();

    @Positive
        void println(Object o);
    @Positive
    }

    @Positive
    public synchronized Throwable fillInStackTrace();

    @Positive
    public StackTraceElement[] getStackTrace();

    @Positive
    public void setStackTrace(StackTraceElement[] stackTrace);

    @Positive
    public final synchronized void addSuppressed(Throwable exception);

    @Positive
    public final synchronized Throwable[] getSuppressed();
    @Positive
}
