/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.lock.qual.EnsuresLockHeldIf;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Map;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import java.util.concurrent.TimeUnit;
    @Positive
import java.util.concurrent.locks.LockSupport;
    @Positive
import jdk.internal.misc.TerminatingThreadLocal;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import sun.nio.ch.Interruptible;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class Thread implements Runnable {

    @Positive
    static void blockedOn(Interruptible b);

    @Positive
    public static final int MIN_PRIORITY;

    @Positive
    public static final int NORM_PRIORITY;

    @Positive
    public static final int MAX_PRIORITY;

    @Positive
    @IntrinsicCandidate
    @Positive
    public static native Thread currentThread();

    @Positive
    public static native void yield();

    @Positive
    public static native void sleep(long millis) throws InterruptedException;

    @Positive
    public static void sleep(long millis, int nanos) throws InterruptedException;

    @Positive
    @IntrinsicCandidate
    @Positive
    public static void onSpinWait();

    @Positive
    @Override
    @Positive
    protected Object clone() throws CloneNotSupportedException;

    @Positive
    public Thread() {
    @Positive
    }

    @Positive
    public Thread(@Nullable Runnable target) {
    @Positive
    }

    @Positive
    public Thread(@Nullable ThreadGroup group, @Nullable Runnable target) {
    @Positive
    }

    @Positive
    public Thread(String name) {
    @Positive
    }

    @Positive
    public Thread(@Nullable ThreadGroup group, String name) {
    @Positive
    }

    @Positive
    public Thread(@Nullable Runnable target, String name) {
    @Positive
    }

    @Positive
    public Thread(@Nullable ThreadGroup group, @Nullable Runnable target, String name) {
    @Positive
    }

    @Positive
    public Thread(@Nullable ThreadGroup group, @Nullable Runnable target, String name, long stackSize) {
    @Positive
    }

    @Positive
    public Thread(ThreadGroup group, Runnable target, String name, long stackSize, boolean inheritThreadLocals) {
    @Positive
    }

    @Positive
    public synchronized void start();

    @Positive
    @Override
    @Positive
    public void run();

    @Positive
    @Deprecated()
    @Positive
    public final void stop();

    @Positive
    public void interrupt();

    @Positive
    public static boolean interrupted();

    @Positive
    @Pure
    @Positive
    public boolean isInterrupted(@GuardSatisfied Thread this);

    @Positive
    @Pure
    @Positive
    public final native boolean isAlive(@GuardSatisfied Thread this);

    @Positive
    @Deprecated()
    @Positive
    public final void suspend();

    @Positive
    @Deprecated()
    @Positive
    public final void resume();

    @Positive
    public final void setPriority(@UnknownInitialization(java.lang.Thread.class) Thread this, int newPriority);

    @Positive
    public final int getPriority();

    @Positive
    public final synchronized void setName(String name);

    @Positive
    public final String getName();

    @Positive
    @Nullable
    @Positive
    public final ThreadGroup getThreadGroup();

    @Positive
    public static int activeCount();

    @Positive
    public static int enumerate(Thread[] tarray);

    @Positive
    @Deprecated()
    @Positive
    public int countStackFrames();

    @Positive
    public final synchronized void join(final long millis) throws InterruptedException;

    @Positive
    public final synchronized void join(long millis, int nanos) throws InterruptedException;

    @Positive
    public final void join() throws InterruptedException;

    @Positive
    public static void dumpStack();

    @Positive
    public final void setDaemon(@UnknownInitialization Thread this, boolean on);

    @Positive
    @Pure
    @Positive
    public final boolean isDaemon(@GuardSatisfied Thread this);

    @Positive
    @Deprecated()
    @Positive
    public final void checkAccess();

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Thread this);

    @Positive
    @CallerSensitive
    @Positive
    @Nullable
    @Positive
    public ClassLoader getContextClassLoader();

    @Positive
    public void setContextClassLoader(@Nullable ClassLoader cl);

    @Positive
    @EnsuresLockHeldIf(expression = { "#1" }, result = true)
    @Positive
    @ReleasesNoLocks
    @Positive
    public static native boolean holdsLock(Object obj);

    @Positive
    public StackTraceElement[] getStackTrace();

    @Positive
    public static Map<Thread, StackTraceElement[]> getAllStackTraces();

    @Positive
    private static class Caches {
    @Positive
    }

    @Positive
    public long getId();

    @Positive
    public enum State {

    @Positive
        NEW,
    @Positive
        RUNNABLE,
    @Positive
        BLOCKED,
    @Positive
        WAITING,
    @Positive
        TIMED_WAITING,
    @Positive
        TERMINATED
    @Positive
    }

    @Positive
    public State getState();

    @Positive
    @FunctionalInterface
    @Positive
    public interface UncaughtExceptionHandler {

    @Positive
        void uncaughtException(Thread t, Throwable e);
    @Positive
    }

    @Positive
    public static void setDefaultUncaughtExceptionHandler(@Nullable UncaughtExceptionHandler eh);

    @Positive
    @Nullable
    @Positive
    public static UncaughtExceptionHandler getDefaultUncaughtExceptionHandler();

    @Positive
    @Nullable
    @Positive
    public UncaughtExceptionHandler getUncaughtExceptionHandler();

    @Positive
    public void setUncaughtExceptionHandler(@Nullable UncaughtExceptionHandler eh);

    @Positive
    static void processQueue(ReferenceQueue<Class<?>> queue, ConcurrentMap<? extends WeakReference<Class<?>>, ?> map);

    @Positive
    static class WeakClassKey extends WeakReference<Class<?>> {

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
