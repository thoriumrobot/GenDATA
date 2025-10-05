/*
 * Copyright (c) 1994, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.initialization.qual.UnknownInitialization;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.*;
import java.util.*;

@AnnotatedFor({ "interning", "lock", "nullness" })
@UsesObjectEquals
public class Throwable implements Serializable {

    private static class SentinelHolder {

        public static final StackTraceElement STACK_TRACE_ELEMENT_SENTINEL;

        public static final StackTraceElement[] STACK_TRACE_SENTINEL;
    }

    @SideEffectFree
    public Throwable() {
    }

    @SideEffectFree
    public Throwable(@Nullable String message) {
    }

    @SideEffectFree
    public Throwable(@Nullable String message, @Nullable Throwable cause) {
    }

    @SideEffectFree
    public Throwable(@Nullable Throwable cause) {
    }

    @SideEffectFree
    protected Throwable(@Nullable String message, @Nullable Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
    }

    @Pure
    @Nullable
    public String getMessage(@GuardSatisfied Throwable this);

    @SideEffectFree
    @Nullable
    public String getLocalizedMessage(@GuardSatisfied Throwable this);

    @Pure
    @Nullable
    public synchronized Throwable getCause(@GuardSatisfied Throwable this);

    @UnknownInitialization
    public synchronized Throwable initCause(@UnknownInitialization Throwable this, @Nullable Throwable cause);

    final void setCause(Throwable t);

    @SideEffectFree
    public String toString(@GuardSatisfied Throwable this);

    public void printStackTrace();

    public void printStackTrace(PrintStream s);

    public void printStackTrace(PrintWriter s);

    private abstract static class PrintStreamOrWriter {

        abstract Object lock();

        abstract void println(Object o);
    }

    private static class WrappedPrintStream extends PrintStreamOrWriter {

        Object lock();

        void println(Object o);
    }

    private static class WrappedPrintWriter extends PrintStreamOrWriter {

        Object lock();

        void println(Object o);
    }

    public synchronized Throwable fillInStackTrace();

    public StackTraceElement[] getStackTrace();

    public void setStackTrace(StackTraceElement[] stackTrace);

    public final synchronized void addSuppressed(Throwable exception);

    public final synchronized Throwable[] getSuppressed();
}
