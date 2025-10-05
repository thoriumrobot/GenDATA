/*
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.lock.qual.EnsuresLockHeldIf;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.security.AccessController;
import java.security.AccessControlContext;
import java.security.PrivilegedAction;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;
import jdk.internal.misc.TerminatingThreadLocal;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;
import jdk.internal.vm.annotation.IntrinsicCandidate;
import sun.nio.ch.Interruptible;
import sun.security.util.SecurityConstants;

@AnnotatedFor({ "interning", "lock", "nullness" })
@UsesObjectEquals
public class Thread implements Runnable {

    static void blockedOn(Interruptible b);

    public static final int MIN_PRIORITY;

    public static final int NORM_PRIORITY;

    public static final int MAX_PRIORITY;

    @IntrinsicCandidate
    public static native Thread currentThread();

    public static native void yield();

    public static native void sleep(long millis) throws InterruptedException;

    public static void sleep(long millis, int nanos) throws InterruptedException;

    @IntrinsicCandidate
    public static void onSpinWait();

    @Override
    protected Object clone() throws CloneNotSupportedException;

    public Thread() {
    }

    public Thread(@Nullable Runnable target) {
    }

    public Thread(@Nullable ThreadGroup group, @Nullable Runnable target) {
    }

    public Thread(String name) {
    }

    public Thread(@Nullable ThreadGroup group, String name) {
    }

    public Thread(@Nullable Runnable target, String name) {
    }

    public Thread(@Nullable ThreadGroup group, @Nullable Runnable target, String name) {
    }

    public Thread(@Nullable ThreadGroup group, @Nullable Runnable target, String name, long stackSize) {
    }

    public Thread(ThreadGroup group, Runnable target, String name, long stackSize, boolean inheritThreadLocals) {
    }

    public synchronized void start();

    @Override
    public void run();

    @Deprecated()
    public final void stop();

    public void interrupt();

    public static boolean interrupted();

    @Pure
    public boolean isInterrupted(@GuardSatisfied Thread this);

    @Pure
    public final native boolean isAlive(@GuardSatisfied Thread this);

    @Deprecated()
    public final void suspend();

    @Deprecated()
    public final void resume();

    public final void setPriority(@UnknownInitialization(java.lang.Thread.class) Thread this, int newPriority);

    public final int getPriority();

    public final synchronized void setName(String name);

    public final String getName();

    @Nullable
    public final ThreadGroup getThreadGroup();

    public static int activeCount();

    public static int enumerate(Thread[] tarray);

    @Deprecated()
    public int countStackFrames();

    public final synchronized void join(final long millis) throws InterruptedException;

    public final synchronized void join(long millis, int nanos) throws InterruptedException;

    public final void join() throws InterruptedException;

    public static void dumpStack();

    public final void setDaemon(@UnknownInitialization Thread this, boolean on);

    @Pure
    public final boolean isDaemon(@GuardSatisfied Thread this);

    @Deprecated()
    public final void checkAccess();

    @SideEffectFree
    public String toString(@GuardSatisfied Thread this);

    @CallerSensitive
    @Nullable
    public ClassLoader getContextClassLoader();

    public void setContextClassLoader(@Nullable ClassLoader cl);

    @EnsuresLockHeldIf(expression = { "#1" }, result = true)
    @ReleasesNoLocks
    public static native boolean holdsLock(Object obj);

    public StackTraceElement[] getStackTrace();

    public static Map<Thread, StackTraceElement[]> getAllStackTraces();

    private static class Caches {
    }

    public long getId();

    public enum State {

        NEW,
        RUNNABLE,
        BLOCKED,
        WAITING,
        TIMED_WAITING,
        TERMINATED
    }

    public State getState();

    @FunctionalInterface
    public interface UncaughtExceptionHandler {

        void uncaughtException(Thread t, Throwable e);
    }

    public static void setDefaultUncaughtExceptionHandler(@Nullable UncaughtExceptionHandler eh);

    @Nullable
    public static UncaughtExceptionHandler getDefaultUncaughtExceptionHandler();

    @Nullable
    public UncaughtExceptionHandler getUncaughtExceptionHandler();

    public void setUncaughtExceptionHandler(@Nullable UncaughtExceptionHandler eh);

    static void processQueue(ReferenceQueue<Class<?>> queue, ConcurrentMap<? extends WeakReference<Class<?>>, ?> map);

    static class WeakClassKey extends WeakReference<Class<?>> {

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object obj);
    }
}
