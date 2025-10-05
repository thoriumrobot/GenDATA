/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
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
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.io.PrintStream;
    @Positive
import java.util.Arrays;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class ThreadGroup implements Thread.UncaughtExceptionHandler {

    @Positive
    public ThreadGroup(@Nullable String name) {
    @Positive
    }

    @Positive
    public ThreadGroup(ThreadGroup parent, @Nullable String name) {
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public final String getName();

    @Positive
    @Nullable
    @Positive
    public final ThreadGroup getParent();

    @Positive
    public final int getMaxPriority();

    @Positive
    @Pure
    @Positive
    @Deprecated()
    @Positive
    public final boolean isDaemon(@GuardSatisfied ThreadGroup this);

    @Positive
    @Pure
    @Positive
    @Deprecated()
    @Positive
    public synchronized boolean isDestroyed(@GuardSatisfied ThreadGroup this);

    @Positive
    @Deprecated()
    @Positive
    public final void setDaemon(boolean daemon);

    @Positive
    @CFComment({ "index: groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @Positive
    @SuppressWarnings("index:array.access.unsafe.high")
    @Positive
    public final void setMaxPriority(int pri);

    @Positive
    public final boolean parentOf(ThreadGroup g);

    @Positive
    @Deprecated()
    @Positive
    public final void checkAccess();

    @Positive
    @CFComment({ "index: groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @Positive
    @SuppressWarnings("index:array.access.unsafe.high")
    @Positive
    @NonNegative
    @Positive
    public int activeCount();

    @Positive
    @NonNegative
    @Positive
    public int enumerate(Thread[] list);

    @Positive
    @NonNegative
    @Positive
    public int enumerate(Thread[] list, boolean recurse);

    @Positive
    @CFComment({ "index: groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @Positive
    @SuppressWarnings("index:array.access.unsafe.high")
    @Positive
    @NonNegative
    @Positive
    public int activeGroupCount();

    @Positive
    @NonNegative
    @Positive
    public int enumerate(ThreadGroup[] list);

    @Positive
    @NonNegative
    @Positive
    public int enumerate(ThreadGroup[] list, boolean recurse);

    @Positive
    @Deprecated()
    @Positive
    public final void stop();

    @Positive
    @CFComment({ " groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @Positive
    @SuppressWarnings("index:array.access.unsafe.high")
    @Positive
    public final void interrupt();

    @Positive
    @Deprecated()
    @Positive
    @SuppressWarnings("removal")
    @Positive
    public final void suspend();

    @Positive
    @CFComment({ "index:  // groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @Positive
    @Deprecated()
    @Positive
    @SuppressWarnings({ "removal", "index:array.access.unsafe.high" })
    @Positive
    public final void resume();

    @Positive
    @CFComment({ "index: groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @Positive
    @SuppressWarnings("index:array.access.unsafe.high")
    @Positive
    @Deprecated()
    @Positive
    public final void destroy();

    @Positive
    void addUnstarted();

    @Positive
    @CFComment({ "index: #1: If nthreads = threads.length, length of threads is doubled" })
    @Positive
    @SuppressWarnings({ "index:array.access.unsafe.high", "index:compound.assignment" })
    @Positive
    void add(Thread t);

    @Positive
    void threadStartFailed(Thread t);

    @Positive
    void threadTerminated(Thread t);

    @Positive
    @CFComment({ "index: groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @Positive
    @SuppressWarnings("index:array.access.unsafe.high")
    @Positive
    public void list();

    @Positive
    void list(PrintStream out, int indent);

    @Positive
    public void uncaughtException(Thread t, Throwable e);

    @Positive
    @Deprecated()
    @Positive
    public boolean allowThreadSuspension(boolean b);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied ThreadGroup this);
    @Positive
}

// CFWR semantic augmentation - variant 0
